import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import psycopg
import pytest

from jobpdf.ingestion import store as store_module
from jobpdf.ingestion.sources.models import SourcePosting
from jobpdf.ingestion.store import (
    HASH_VERSION,
    StoredState,
    VacancyStore,
    Versions,
    apply_schema,
    content_hash,
    is_stale,
    plan_batch,
)
from jobpdf.normalization.db import database_url
from jobpdf.ranking.models import SkillRef, VacancyFeatures

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
V = Versions(embedding="e5@1|query: ", vacancy_text="1", taxonomy="v1.2.1", index="idx1",
             matcher_prompt="p1", extraction="rules-1")


def posting(i: int = 1, **changes) -> SourcePosting:
    data = {
        "source": "remotive", "source_id": str(i), "url": f"https://remotive.com/{i}",
        "title": f"Data Engineer {i}", "company": "Acme", "location_raw": "Worldwide",
        "remote": True, "description": "<p>Build things</p>", "tags": ["python", "sql"],
        "category": "Software", "employment_type": "full_time", "salary_raw": "$50k",
        "posted_at": T0,
    }
    return SourcePosting(**{**data, **changes})


def features(p: SourcePosting, *, versions: Versions = V, vector=(0.5, -0.25, 0.125)
             ) -> VacancyFeatures:
    return VacancyFeatures(
        id=p.source_id, source=p.source, title=p.title, company=p.company, url=p.url,
        posted_at=p.posted_at, location_raw=p.location_raw,
        required_skills=[SkillRef(concept_id="c/python", raw="python", normalized="python",
                                  label="Python"),
                         SkillRef(raw="Obscure", normalized="obscure")],
        preferred_skills=[SkillRef(concept_id="c/sql", raw="sql", normalized="sql")],
        responsibilities=["Build pipelines"], min_years_experience=3,
        languages=["English"],
        embedding=list(vector), embedding_version=versions.embedding,
        vacancy_text_version=versions.vacancy_text, taxonomy_version=versions.taxonomy,
        index_version=versions.index, matcher_prompt_version=versions.matcher_prompt,
        extraction_version=versions.extraction,
    )


def state(p: SourcePosting, versions: Versions = V, fetched: datetime = T0) -> StoredState:
    return StoredState(content_hash(p), versions, fetched)


# --- the content hash ----------------------------------------------------------------


def test_equal_postings_hash_equal_and_tag_order_does_not_matter() -> None:
    assert content_hash(posting(1)) == content_hash(posting(1))
    assert content_hash(posting(1, tags=["sql", "python"])) == content_hash(posting(1))
    assert len(content_hash(posting(1))) == 64


@pytest.mark.parametrize("changes", [
    {"title": "Other"}, {"company": "Other"}, {"url": "https://remotive.com/other"},
    {"location_raw": "Europe"}, {"remote": False}, {"remote": None},
    {"description": "<p>Other</p>"}, {"description_is_excerpt": True},
    {"tags": ["python"]}, {"category": "Other"}, {"employment_type": "contract"},
    {"salary_raw": "$60k"}, {"description": None},
])
def test_every_content_field_changes_the_hash(changes) -> None:
    assert content_hash(posting(1, **changes)) != content_hash(posting(1))


def test_the_posted_time_and_identity_do_not_change_the_hash() -> None:
    base = content_hash(posting(1))

    assert content_hash(posting(1, posted_at=T0 + timedelta(days=3))) == base
    assert content_hash(posting(1, posted_at=None)) == base


def test_the_hash_is_pinned_so_an_accidental_change_is_noticed() -> None:
    # If this fails you changed what counts as "changed": bump HASH_VERSION on purpose,
    # because every stored vacancy will be reprocessed once.
    assert HASH_VERSION == "1"
    assert content_hash(posting(1)) == (
        "d810ab508a2152648262252e6b051b7313e6e162060bf42287624621587d51a6")


# --- stale versions --------------------------------------------------------------------


def test_a_row_is_stale_when_any_known_current_version_differs() -> None:
    assert not is_stale(V, V)
    for field in ("embedding", "vacancy_text", "taxonomy", "index", "matcher_prompt"):
        changed = Versions(**{**V.__dict__, field: "other"})
        assert is_stale(V, changed), field


def test_an_unknown_current_version_is_not_compared() -> None:
    unknown = Versions(embedding=None, vacancy_text="1", taxonomy=None, index="unavailable",
                       matcher_prompt="p1")

    assert not is_stale(Versions(embedding="old", vacancy_text="1", taxonomy="old",
                                 index="old", matcher_prompt="p1"), unknown)
    assert is_stale(Versions(vacancy_text="0", matcher_prompt="p1"), unknown)


def test_a_row_with_no_stamp_is_stale_once_a_version_is_known() -> None:
    assert is_stale(Versions(), V)
    assert not is_stale(Versions(), Versions())


# --- planning a batch (no database) ----------------------------------------------------


def test_a_batch_splits_into_new_changed_stale_and_unchanged() -> None:
    a, b, c, d = (posting(i) for i in (1, 2, 3, 4))
    stored = {
        ("remotive", "2"): state(posting(2, title="Old title")),
        ("remotive", "3"): state(c, Versions(embedding="old", vacancy_text="1",
                                             taxonomy="v1.2.1", index="idx1",
                                             matcher_prompt="p1")),
        ("remotive", "4"): state(d),
    }

    plan = plan_batch([a, b, c, d], stored, V)

    assert [p.source_id for p in plan.new] == ["1"]
    assert [p.source_id for p in plan.changed] == ["2"]
    assert [p.source_id for p in plan.stale] == ["3"]
    assert [p.source_id for p in plan.unchanged] == ["4"]
    assert [p.source_id for p in plan.to_process] == ["1", "2", "3"]
    assert plan.duplicates == 0


def test_the_same_id_from_different_sources_is_two_postings() -> None:
    plan = plan_batch([posting(1), posting(1, source="adzuna")], {}, V)

    assert len(plan.new) == 2


def test_repeats_in_a_batch_count_once_and_the_first_wins() -> None:
    first, again = posting(1), posting(1, title="Different")

    plan = plan_batch([first, again, posting(2)], {}, V)

    assert plan.new == [first, posting(2)] and plan.duplicates == 1


def test_a_changed_posting_is_changed_even_if_versions_are_stale_too() -> None:
    old = Versions(embedding="old")
    plan = plan_batch([posting(1)], {("remotive", "1"): state(posting(1, title="x"), old)}, V)

    assert len(plan.changed) == 1 and plan.stale == []


def test_an_empty_batch_is_an_empty_plan() -> None:
    plan = plan_batch([], {}, V)

    assert plan.to_process == [] and plan.unchanged == [] and plan.duplicates == 0


# --- the database -----------------------------------------------------------------------


@pytest.fixture
def db_url() -> str:
    url = database_url()
    if not url:
        pytest.skip("DATABASE_URL not set (environment or .env)")
    return url


@pytest.fixture
def conn(db_url: str) -> Iterator[psycopg.Connection]:
    """A connection whose search_path is a throwaway schema with the vacancies table."""
    name = f"test_{uuid.uuid4().hex[:12]}"
    try:
        admin = psycopg.connect(db_url, autocommit=True, connect_timeout=10)
    except psycopg.OperationalError as exc:
        pytest.skip(f"database not reachable: {type(exc).__name__}")
    with admin:
        admin.execute(f'CREATE SCHEMA "{name}"')
    connection = psycopg.connect(db_url, options=f"-c search_path={name}", connect_timeout=10)
    apply_schema(connection)
    try:
        yield connection
    finally:
        connection.close()
        with psycopg.connect(db_url, autocommit=True, connect_timeout=10) as admin:
            admin.execute(f'DROP SCHEMA "{name}" CASCADE')


@pytest.fixture
def store(conn) -> VacancyStore:
    return VacancyStore(conn)


def rows(conn) -> list[tuple]:
    out = conn.execute("SELECT source, source_id FROM vacancies ORDER BY 1, 2").fetchall()
    conn.rollback()
    return out


@pytest.mark.db
def test_the_schema_can_be_applied_twice(conn) -> None:
    apply_schema(conn)
    apply_schema(conn)

    assert rows(conn) == []


@pytest.mark.db
def test_a_new_vacancy_round_trips_with_everything_ranking_needs(store) -> None:
    p = posting(1)

    result = store.upsert(p, features(p), T0)
    (loaded,) = store.load()

    assert result.inserted and not result.updated and not result.skipped_older
    original = features(p)
    assert loaded.model_dump(exclude={"embedding"}) == original.model_dump(exclude={"embedding"})
    assert loaded.embedding == pytest.approx(original.embedding)
    assert loaded.required_skills[0].key == "c/python"
    assert loaded.required_skills[1].key == "raw:obscure"
    assert loaded.posted_at == T0 and loaded.url == p.url


@pytest.mark.db
def test_the_same_posting_twice_is_one_row(store, conn) -> None:
    p = posting(1)

    first = store.upsert(p, features(p), T0)
    second = store.upsert(p, features(p), T0 + timedelta(hours=8))

    assert first.inserted and second.updated and not second.inserted
    assert rows(conn) == [("remotive", "1")] and store.count() == 1


@pytest.mark.db
def test_a_changed_posting_replaces_its_row_and_keeps_when_it_was_first_seen(store, conn) -> None:
    p = posting(1)
    store.upsert(p, features(p), T0)
    changed = posting(1, title="Senior Data Engineer")

    store.upsert(changed, features(changed, vector=(1.0, 2.0, 3.0)), T0 + timedelta(days=1))

    (loaded,) = store.load()
    first, last, h = conn.execute(
        "SELECT first_seen_at, last_seen_at, content_hash FROM vacancies").fetchone()
    assert loaded.title == "Senior Data Engineer" and loaded.embedding == [1.0, 2.0, 3.0]
    assert first == T0 and last == T0 + timedelta(days=1) and h == content_hash(changed)


@pytest.mark.db
def test_an_older_fetch_never_overwrites_a_newer_one(store) -> None:
    newer, older = posting(1, title="Newer"), posting(1, title="Older")
    store.upsert(newer, features(newer), T0 + timedelta(days=1))

    result = store.upsert(older, features(older), T0)

    assert result.skipped_older and not result.inserted and not result.updated
    assert store.load()[0].title == "Newer"


@pytest.mark.db
def test_a_fetch_at_the_same_instant_is_applied(store) -> None:
    store.upsert(posting(1, title="A"), features(posting(1, title="A")), T0)

    result = store.upsert(posting(1, title="B"), features(posting(1, title="B")), T0)

    assert result.updated and store.load()[0].title == "B"


@pytest.mark.db
def test_last_seen_only_moves_forward(store, conn) -> None:
    p = posting(1)
    store.upsert(p, features(p), T0 + timedelta(days=2))
    store.upsert(p, features(p), T0 + timedelta(days=1))  # skipped: older

    last = conn.execute("SELECT last_seen_at FROM vacancies").fetchone()[0]
    conn.rollback()
    assert last == T0 + timedelta(days=2)


@pytest.mark.db
def test_classify_reads_what_is_stored(store) -> None:
    a, b, c = posting(1), posting(2), posting(3)
    store.upsert(a, features(a), T0)
    store.upsert(b, features(b, versions=Versions(embedding="old")), T0)

    plan = store.classify([a, b, posting(1, title="changed"), c], V)

    assert [p.source_id for p in plan.new] == ["3"]
    assert [p.source_id for p in plan.unchanged] == ["1"]
    assert [p.source_id for p in plan.stale] == ["2"]
    assert plan.changed == [] and plan.duplicates == 1


@pytest.mark.db
def test_classify_sees_a_change_after_it_is_stored(store) -> None:
    p = posting(1)
    store.upsert(p, features(p), T0)
    edited = posting(1, description="<p>New text</p>")

    assert [x.source_id for x in store.classify([edited], V).changed] == ["1"]
    store.upsert(edited, features(edited), T0 + timedelta(hours=8))
    assert [x.source_id for x in store.classify([edited], V).unchanged] == ["1"]


@pytest.mark.db
def test_classify_with_nothing_asks_nothing(store) -> None:
    assert store.classify([], V).to_process == []
    assert store.states([]) == {}


@pytest.mark.db
def test_the_same_id_in_two_sources_are_two_rows(store) -> None:
    for source in ("remotive", "adzuna"):
        p = posting(1, source=source)
        store.upsert(p, features(p), T0)

    assert store.count() == 2 and store.count("adzuna") == 1
    assert [v.source for v in store.load(source="adzuna")] == ["adzuna"]


@pytest.mark.db
def test_touch_notes_that_unchanged_postings_were_seen_again(store, conn) -> None:
    a, b = posting(1), posting(2)
    store.upsert(a, features(a), T0)
    store.upsert(b, features(b), T0)

    updated = store.touch([a], T0 + timedelta(hours=8))

    rows_ = dict(((r[0]), r[1]) for r in conn.execute(
        "SELECT source_id, last_seen_at FROM vacancies").fetchall())
    conn.rollback()
    assert updated == 1
    assert rows_["1"] == T0 + timedelta(hours=8) and rows_["2"] == T0


@pytest.mark.db
def test_touch_never_moves_time_backwards_or_touches_content(store, conn) -> None:
    p = posting(1)
    store.upsert(p, features(p), T0 + timedelta(days=1))

    assert store.touch([p], T0) == 0
    assert store.touch([], T0) == 0
    assert store.touch([posting(99)], T0 + timedelta(days=5)) == 0  # not stored: no row made
    assert store.count() == 1
    assert store.load()[0].title == p.title


@pytest.mark.db
def test_touch_moves_fetched_at_too_so_an_older_fetch_cannot_overwrite_later(store) -> None:
    p = posting(1)
    store.upsert(p, features(p), T0)
    store.touch([p], T0 + timedelta(days=1))
    older = posting(1, title="Older content")

    result = store.upsert(older, features(older), T0 + timedelta(hours=12))

    assert result.skipped_older


@pytest.mark.db
def test_load_filters_by_last_seen_and_orders_newest_posting_first(store) -> None:
    old, mid, new, undated = posting(1), posting(2), posting(3), posting(4, posted_at=None)
    store.upsert(old, features(old), T0 - timedelta(days=40))
    store.upsert(mid, features(mid), T0)
    store.upsert(new, features(new), T0)
    store.upsert(undated, features(undated), T0)
    mid_later = posting(2, posted_at=T0 + timedelta(days=1))
    store.upsert(mid_later, features(mid_later), T0)
    new_later = posting(3, posted_at=T0 + timedelta(days=2))
    store.upsert(new_later, features(new_later), T0)

    recent = store.load(seen_since=T0 - timedelta(days=30))

    assert [v.id for v in recent] == ["3", "2", "4"]
    assert [v.id for v in store.load()] == ["3", "2", "1", "4"]


@pytest.mark.db
def test_features_that_belong_to_another_posting_are_refused(store) -> None:
    with pytest.raises(ValueError):
        store.upsert(posting(1), features(posting(2)), T0)
    with pytest.raises(ValueError):
        store.upsert(posting(1), features(posting(1, source="adzuna")), T0)
    assert store.count() == 0


@pytest.mark.db
def test_a_vacancy_without_an_embedding_is_stored_as_such(store) -> None:
    p = posting(1)
    bare = features(p).model_copy(update={"embedding": None, "embedding_version": None})

    store.upsert(p, bare, T0)

    (loaded,) = store.load()
    assert loaded.embedding is None and loaded.embedding_version is None


@pytest.mark.db
def test_blank_identity_is_refused_by_the_table(conn) -> None:
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            """INSERT INTO vacancies (source, source_id, content_hash, features, url,
                   fetched_at, first_seen_at, last_seen_at)
               VALUES ('', 'x', 'h', '{}', 'u', now(), now(), now())""")


@pytest.mark.db
def test_no_description_text_is_stored(store, conn) -> None:
    p = posting(1, description="<p>SecretDescriptionXYZ</p>")
    store.upsert(p, features(p), T0)

    dump = " ".join(str(v) for v in conn.execute("SELECT * FROM vacancies").fetchone())
    conn.rollback()
    assert "SecretDescriptionXYZ" not in dump


@pytest.mark.db
def test_a_nul_character_fails_that_one_write_and_leaves_the_connection_usable(store) -> None:
    bad = posting(1, title="bad\x00title")

    with pytest.raises(psycopg.Error):
        store.upsert(bad, features(bad), T0)
    store.conn.rollback()
    good = posting(2)
    assert store.upsert(good, features(good), T0).inserted


def test_the_schema_file_is_the_one_the_store_applies() -> None:
    assert store_module.SCHEMA_FILE.name == "004_vacancies.sql"
    assert store_module.SCHEMA_FILE.exists()


@pytest.mark.db
def test_reprocessing_a_stale_row_updates_every_version_stamp(store) -> None:
    p = posting(1)
    old = Versions("old-e", "old-t", "old-x", "old-i", "old-p", "old-r")
    store.upsert(p, features(p, versions=old), T0)
    assert [x.source_id for x in store.classify([p], V).stale] == ["1"]

    store.upsert(p, features(p, versions=V), T0 + timedelta(hours=8))

    assert [x.source_id for x in store.classify([p], V).unchanged] == ["1"]
    (loaded,) = store.load()
    assert (loaded.embedding_version, loaded.vacancy_text_version, loaded.taxonomy_version,
            loaded.index_version, loaded.matcher_prompt_version, loaded.extraction_version) == (
        V.embedding, V.vacancy_text, V.taxonomy, V.index, V.matcher_prompt, V.extraction)


@pytest.mark.db
def test_an_update_refreshes_the_columns_readers_filter_on(store, conn) -> None:
    p = posting(1)
    store.upsert(p, features(p), T0)
    moved = posting(1, url="https://remotive.com/moved", posted_at=T0 + timedelta(days=1))

    store.upsert(moved, features(moved), T0 + timedelta(days=2))

    url, posted, fetched = conn.execute(
        "SELECT url, posted_at, fetched_at FROM vacancies").fetchone()
    conn.rollback()
    assert (url, posted, fetched) == (moved.url, moved.posted_at, T0 + timedelta(days=2))


@pytest.mark.db
def test_the_json_does_not_repeat_the_embedding(store, conn) -> None:
    p = posting(1)
    store.upsert(p, features(p), T0)

    keys = conn.execute("SELECT jsonb_object_keys(features) FROM vacancies").fetchall()
    conn.rollback()
    assert "embedding" not in {k[0] for k in keys} and "title" in {k[0] for k in keys}


@pytest.mark.db
def test_classify_tells_sources_apart(store) -> None:
    store.upsert(posting(1), features(posting(1)), T0)
    other = posting(1, source="adzuna")

    plan = store.classify([other], V)

    assert [x.source for x in plan.new] == ["adzuna"] and plan.unchanged == []


@pytest.mark.db
def test_touch_is_committed_not_left_in_an_open_transaction(store, conn) -> None:
    p = posting(1)
    store.upsert(p, features(p), T0)
    store.touch([p], T0 + timedelta(hours=8))
    conn.rollback()  # whatever is still uncommitted is lost here

    last = conn.execute("SELECT last_seen_at FROM vacancies").fetchone()[0]
    conn.rollback()
    assert last == T0 + timedelta(hours=8)


@pytest.mark.db
def test_seen_since_includes_the_boundary(store) -> None:
    p = posting(1)
    store.upsert(p, features(p), T0)

    assert [v.id for v in store.load(seen_since=T0)] == ["1"]
    assert store.load(seen_since=T0 + timedelta(seconds=1)) == []


@pytest.mark.db
def test_rows_built_by_another_extractor_are_stale_once(store) -> None:
    p = posting(1)
    store.upsert(p, features(p, versions=Versions(**{**V.__dict__, "extraction": "tags"})), T0)
    assert [x.source_id for x in store.classify([p], V).stale] == ["1"]

    store.upsert(p, features(p), T0 + timedelta(hours=8))

    assert [x.source_id for x in store.classify([p], V).unchanged] == ["1"]


@pytest.mark.db
def test_a_table_from_before_jm48_gets_the_extraction_column(conn) -> None:
    conn.execute("DROP TABLE IF EXISTS vacancies")
    conn.execute(store_module.SCHEMA_FILE.read_text(encoding="utf-8")
                 .replace("    extraction_version     text,\n", "")
                 .replace("ALTER TABLE vacancies ADD COLUMN IF NOT EXISTS extraction_version text;",
                          ""))
    conn.commit()
    store_module.apply_schema(conn)
    columns = {r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'vacancies'")}
    assert "extraction_version" in columns
