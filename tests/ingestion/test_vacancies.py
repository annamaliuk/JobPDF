import logging
from datetime import timedelta

import pytest

from jobpdf.ingestion import vacancies
from jobpdf.ingestion.requirements import (
    MAX_TAG_CHARS,
    MAX_TAG_SKILLS,
    NO_REQUIREMENTS,
    TAGS_FALLBACK,
    TagsExtractor,
)
from jobpdf.ingestion.vacancies import (
    MAX_CONSECUTIVE_FAILURES,
    StageError,
    build_features,
    current_versions,
    ingest,
)
from tests.ingestion.ingest_fakes import (
    T0,
    FakeEmbedder,
    FakeExtractor,
    FakeService,
    MemoryStore,
    ServiceVersions,
    posting,
)


class World:
    """A store, a service, an embedder and an extractor, and a way to run a fetch."""

    def __init__(self, **extractor_args) -> None:
        self.store = MemoryStore()
        self.service = FakeService()
        self.embedder = FakeEmbedder()
        self.extractor = FakeExtractor(**extractor_args)

    def run(self, postings, at=T0, **kwargs):
        return ingest(postings, store=self.store, extractor=self.extractor,
                      normalizer=self.service, embedder=self.embedder, fetched_at=at, **kwargs)

    @property
    def work(self) -> tuple[int, int, int]:
        return len(self.extractor.calls), len(self.service.calls), len(self.embedder.calls)


# --- the tags fallback ------------------------------------------------------------------


def test_tags_become_required_skills_cleaned_and_deduplicated() -> None:
    p = posting(1, tags=["  Python ", "python", "SQL", "", "  ", "x" * (MAX_TAG_CHARS + 1),
                         "Machine   Learning", "machine learning"])

    result = TagsExtractor().extract(p)

    assert result.required_skills == ["Python", "SQL", "Machine Learning"]
    assert result.preferred_skills == [] and result.responsibilities == []
    assert result.min_years_experience is None and result.languages is None
    assert result.warnings == [TAGS_FALLBACK]


def test_a_tag_of_exactly_sixty_characters_is_kept_and_longer_is_not() -> None:
    assert MAX_TAG_CHARS == 60
    assert TagsExtractor().extract(posting(1, tags=["x" * 60])).required_skills == ["x" * 60]
    assert TagsExtractor().extract(posting(1, tags=["x" * 61])).required_skills == []


def test_at_most_twenty_five_tags_are_used_in_order() -> None:
    assert MAX_TAG_SKILLS == 25
    p = posting(1, tags=[f"skill{i}" for i in range(40)])

    assert TagsExtractor().extract(p).required_skills == [f"skill{i}" for i in range(25)]


def test_repeats_are_found_whatever_the_case_of_the_first_one() -> None:
    p = posting(1, tags=["python", "Python", "PYTHON", "Go"])

    assert TagsExtractor().extract(p).required_skills == ["python", "Go"]


def test_no_usable_tags_says_so() -> None:
    result = TagsExtractor().extract(posting(1, tags=[" ", ""]))

    assert result.required_skills == [] and result.warnings == [TAGS_FALLBACK, NO_REQUIREMENTS]
    assert TagsExtractor().extract(posting(1, tags=[])).warnings == [TAGS_FALLBACK,
                                                                     NO_REQUIREMENTS]


# --- building one vacancy ---------------------------------------------------------------


def test_a_vacancy_is_built_normalized_embedded_and_stamped() -> None:
    w = World()

    features, requirements = build_features(
        posting(7, tags=["Python", "Rust"]), extractor=w.extractor, normalizer=w.service,
        embedder=w.embedder)

    assert (features.id, features.source, features.title) == ("7", "remotive", "Data Engineer 7")
    assert features.company == "Acme" and features.url == "https://remotive.com/7"
    assert features.location_raw == "Worldwide" and features.posted_at == T0
    assert [r.key for r in features.required_skills] == ["c/python", "raw:rust"]
    assert [r.key for r in features.preferred_skills] == ["c/dbt", "raw:obscure tool"]
    assert features.responsibilities == ["Build pipelines"]
    assert features.min_years_experience == 2 and features.languages == ["English"]
    assert len(features.embedding) == 3 and features.embedding_version == \
        "fake/e5@rev1|query: "
    assert features.vacancy_text_version == "1"
    assert (features.taxonomy_version, features.index_version,
            features.matcher_prompt_version) == ("v1.2.1", "idx1", "p1")
    assert requirements.warnings == ["fake_warning"]


def test_the_matcher_is_told_the_skills_come_from_a_job_posting() -> None:
    w = World()

    build_features(posting(1), extractor=w.extractor, normalizer=w.service, embedder=w.embedder)

    groups, context = w.service.calls[0]
    assert context == "job posting" and set(groups) == {"required", "preferred"}


def test_empty_skills_are_dropped_not_kept_as_blank_requirements() -> None:
    w = World()
    w.extractor.extract = lambda p: vacancies.Requirements(required_skills=["  ", "Python"])

    features, _ = build_features(posting(1), extractor=w.extractor, normalizer=w.service,
                                 embedder=w.embedder)

    assert [r.key for r in features.required_skills] == ["c/python"]


@pytest.mark.parametrize(("break_it", "code"), [
    (lambda w: setattr(w.extractor, "extract", lambda p: 1 / 0), "extraction_failed"),
    (lambda w: setattr(w.service, "fail", True), "normalization_failed"),
    (lambda w: setattr(w.embedder, "fail", True), "embedding_failed"),
])
def test_a_failing_stage_is_reported_by_code_without_the_cause(break_it, code) -> None:
    w = World()
    break_it(w)

    with pytest.raises(StageError) as caught:
        build_features(posting(1), extractor=w.extractor, normalizer=w.service,
                       embedder=w.embedder)

    assert caught.value.code == code and str(caught.value) == code
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert "SecretTitleXYZ" not in repr(caught.value)


def test_the_current_versions_come_from_the_embedder_and_the_service() -> None:
    versions = current_versions(FakeService(ServiceVersions("v9", "i9", "p9")), FakeEmbedder())

    assert (versions.embedding, versions.vacancy_text, versions.taxonomy, versions.index,
            versions.matcher_prompt) == ("fake/e5@rev1|query: ", "1", "v9", "i9", "p9")


# --- incremental runs -----------------------------------------------------------------------


def test_a_first_run_stores_everything_and_counts_it() -> None:
    w = World()

    summary = w.run([posting(1), posting(2), posting(3)])

    assert (summary.received, summary.new, summary.inserted, summary.updated) == (3, 3, 3, 0)
    assert (summary.changed, summary.stale, summary.unchanged) == (0, 0, 0)
    assert summary.failed == {} and summary.deferred == 0 and not summary.aborted
    assert summary.source == "remotive" and set(w.store.rows) == {
        ("remotive", "1"), ("remotive", "2"), ("remotive", "3")}
    assert summary.warnings == {"fake_warning": 3}


def test_running_the_same_fetch_again_does_no_work_at_all() -> None:
    w = World()
    w.run([posting(1), posting(2)])
    before = w.work

    summary = w.run([posting(1), posting(2)], at=T0 + timedelta(hours=8))

    assert w.work == before  # no extraction, no normalization, no embedding
    assert (summary.unchanged, summary.new, summary.inserted, summary.updated) == (2, 0, 0, 0)
    assert w.store.rows[("remotive", "1")]["last_seen"] == T0 + timedelta(hours=8)


def test_only_the_changed_posting_is_processed() -> None:
    w = World()
    w.run([posting(1), posting(2), posting(3)])
    w.extractor.calls.clear()

    summary = w.run([posting(1), posting(2, title="Senior Data Engineer 2"), posting(3)],
                    at=T0 + timedelta(hours=8))

    assert w.extractor.calls == ["2"]
    assert (summary.changed, summary.unchanged, summary.updated, summary.inserted) == (1, 2, 1, 0)
    assert w.store.rows[("remotive", "2")]["features"].title == "Senior Data Engineer 2"


def test_a_backfill_of_known_postings_adds_no_duplicates() -> None:
    w = World()
    w.run([posting(i) for i in range(1, 6)])

    w.run([posting(i) for i in range(1, 6)], at=T0 + timedelta(days=1))
    w.run([posting(i) for i in range(1, 9)], at=T0 + timedelta(days=2))

    assert len(w.store.rows) == 8


def test_postings_repeated_inside_one_fetch_are_processed_once() -> None:
    w = World()

    summary = w.run([posting(1), posting(1, title="Same id again"), posting(2)])

    assert summary.duplicates == 1 and summary.inserted == 2 and w.extractor.calls == ["1", "2"]


def test_a_new_embedding_version_rebuilds_every_row_once() -> None:
    w = World()
    w.run([posting(1), posting(2)])
    w.embedder.revision = "rev2"

    first = w.run([posting(1), posting(2)], at=T0 + timedelta(hours=8))
    second = w.run([posting(1), posting(2)], at=T0 + timedelta(hours=16))

    assert (first.stale, first.updated, first.unchanged) == (2, 2, 0)
    assert (second.stale, second.unchanged) == (0, 2)
    assert w.store.rows[("remotive", "1")]["features"].embedding_version.endswith("rev2|query: ")


@pytest.mark.parametrize("change", [
    {"taxonomy": "v2"}, {"index": "idx2"}, {"matcher_prompt": "p2"},
])
def test_a_new_normalization_version_rebuilds_the_rows(change) -> None:
    w = World()
    w.run([posting(1)])
    for field, value in change.items():
        setattr(w.service.versions, field, value)

    summary = w.run([posting(1)], at=T0 + timedelta(hours=8))

    assert summary.stale == 1 and summary.updated == 1


def test_an_unavailable_index_does_not_make_everything_stale_forever() -> None:
    w = World()
    w.service.versions.index = "unavailable"
    w.run([posting(1)])

    summary = w.run([posting(1)], at=T0 + timedelta(hours=8))

    assert summary.unchanged == 1 and summary.stale == 0


def test_an_older_fetch_does_not_replace_newer_data() -> None:
    w = World()
    w.run([posting(1, title="Newer")], at=T0 + timedelta(days=1))

    summary = w.run([posting(1, title="Older")], at=T0)

    assert summary.changed == 1 and summary.skipped_older == 1 and summary.updated == 0
    assert w.store.rows[("remotive", "1")]["features"].title == "Newer"


def test_nothing_to_ingest_does_nothing() -> None:
    w = World()

    summary = w.run([], source="adzuna")

    assert summary.received == 0 and summary.source == "adzuna" and w.work == (0, 0, 0)


# --- the per-run limit ------------------------------------------------------------------------


def test_a_run_processes_the_newest_postings_first_and_defers_the_rest() -> None:
    w = World()
    postings = [posting(i, posted_at=T0 + timedelta(days=i)) for i in range(1, 6)]
    undated = posting(9, posted_at=None)

    summary = w.run([*postings, undated], limit=3)

    assert w.extractor.calls == ["5", "4", "3"]
    assert (summary.inserted, summary.deferred, summary.new) == (3, 3, 6)


def test_deferred_postings_are_picked_up_by_the_next_runs() -> None:
    w = World()
    postings = [posting(i, posted_at=T0 + timedelta(days=i)) for i in range(1, 6)]

    w.run(postings, limit=2)
    w.run(postings, at=T0 + timedelta(hours=8), limit=2)
    last = w.run(postings, at=T0 + timedelta(hours=16), limit=2)

    assert len(w.store.rows) == 5 and last.inserted == 1 and last.deferred == 0


def test_deferred_changed_postings_are_still_marked_as_seen() -> None:
    w = World()
    postings = [posting(i, posted_at=T0 + timedelta(days=i)) for i in range(1, 4)]
    w.run(postings)
    edited = [p.model_copy(update={"title": p.title + " (edited)"}) for p in postings]
    later = T0 + timedelta(hours=8)

    summary = w.run(edited, at=later, limit=1)

    assert summary.updated == 1 and summary.deferred == 2
    assert all(w.store.rows[("remotive", str(i))]["last_seen"] == later for i in (1, 2, 3))


def test_the_default_limit_is_a_sane_bound() -> None:
    assert 10 <= vacancies.MAX_PROCESSED_PER_RUN <= 1000 and MAX_CONSECUTIVE_FAILURES >= 3


# --- failures ---------------------------------------------------------------------------------


def test_one_failing_posting_does_not_stop_the_others_and_is_retried_next_run() -> None:
    w = World(fail_for={"2"})

    first = w.run([posting(1), posting(2), posting(3)])

    assert first.inserted == 2 and first.failed == {"extraction_failed": 1}
    assert ("remotive", "2") not in w.store.rows
    w.extractor.fail_for.clear()
    second = w.run([posting(1), posting(2), posting(3)], at=T0 + timedelta(hours=8))
    assert (second.new, second.inserted, second.unchanged) == (1, 1, 2)


@pytest.mark.parametrize(("break_it", "code"), [
    (lambda w: setattr(w.service, "fail", True), "normalization_failed"),
    (lambda w: setattr(w.embedder, "fail", True), "embedding_failed"),
    (lambda w: w.store.fail_upsert_for.update({"1", "2"}), "store_failed"),
])
def test_each_stage_failure_is_counted_under_its_code(break_it, code) -> None:
    w = World()
    break_it(w)

    summary = w.run([posting(1), posting(2)])

    assert summary.failed == {code: 2} and summary.inserted == 0 and not summary.aborted


def test_a_store_failure_rolls_the_connection_back() -> None:
    w = World()
    w.store.fail_upsert_for.add("1")

    w.run([posting(1), posting(2)])

    assert w.store.conn.rollbacks == 1 and ("remotive", "2") in w.store.rows


def test_a_run_gives_up_after_a_streak_of_failures_instead_of_hammering_a_dead_service() -> None:
    w = World()
    w.service.fail = True
    postings = [posting(i, posted_at=T0 + timedelta(days=i)) for i in range(1, 21)]

    summary = w.run(postings)

    assert summary.aborted and summary.failed == {"normalization_failed": MAX_CONSECUTIVE_FAILURES}
    assert len(w.service.calls) == MAX_CONSECUTIVE_FAILURES
    assert summary.deferred == 20 - MAX_CONSECUTIVE_FAILURES and summary.inserted == 0


def test_a_success_resets_the_failure_streak() -> None:
    n = MAX_CONSECUTIVE_FAILURES - 1
    bad = {str(i) for i in range(1, 2 * n + 2) if i != n + 1}
    w = World(fail_for=bad)
    postings = [posting(i, posted_at=T0 + timedelta(days=100 - i)) for i in range(1, 2 * n + 2)]

    summary = w.run(postings)

    assert not summary.aborted and summary.inserted == 1
    assert summary.failed == {"extraction_failed": 2 * n}


def test_an_abort_still_marks_the_unprocessed_known_postings_as_seen() -> None:
    w = World()
    known = [posting(i, posted_at=T0 + timedelta(days=i)) for i in range(1, 11)]
    w.run(known)
    w.service.fail = True
    edited = [p.model_copy(update={"title": p.title + "!"}) for p in known]
    later = T0 + timedelta(hours=8)

    summary = w.run(edited, at=later)

    assert summary.aborted
    assert all(row["last_seen"] == later for row in w.store.rows.values())


def test_exception_text_never_reaches_the_logs_or_the_summary(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    w = World(fail_for={"1"})
    w.store.fail_upsert_for.add("2")
    w.service.fail = False

    summary = w.run([posting(1, title="SecretTitleXYZ"), posting(2, title="SecretTitleXYZ"),
                     posting(3)])

    everything = caplog.text + summary.model_dump_json()
    assert "SecretTitleXYZ" not in everything and "Data Engineer" not in everything
    assert "remotive" in caplog.text and "failed" in caplog.text


def test_the_summary_is_plain_counts_and_codes() -> None:
    w = World()

    dumped = w.run([posting(1)]).model_dump()

    assert set(dumped) == {
        "source", "received", "duplicates", "new", "changed", "stale", "unchanged",
        "inserted", "updated", "skipped_older", "deferred", "failed", "warnings", "aborted"}
