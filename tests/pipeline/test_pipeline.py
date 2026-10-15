import datetime as dt
import json
import logging
import threading
from pathlib import Path

import pytest

from jobpdf import pipeline
from jobpdf.extraction.grounding import GroundingStatus
from jobpdf.extraction.llm import FakeToolCaller, LLMResponseError, ToolCallResult
from jobpdf.extraction.models import ParseError
from jobpdf.normalization.match_cache import MatchTier, SkillMatch
from jobpdf.normalization.service import ProfileSkills, ServiceVersions, profile_skill_paths
from jobpdf.pipeline import (
    PipelineError,
    analyze_cv,
    classify_extraction_warning,
)
from jobpdf.ranking.models import SkillRef, VacancyFeatures
from jobpdf.ranking.score import LinearScorer, VacancyAdjustment
from tests.normalization.fakes import FakeEmbedder

FIXTURES = Path(__file__).parents[1] / "extraction" / "fixtures"
TODAY = dt.date(2026, 10, 3)
PYTHON = "http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d"
SQL = "http://data.europa.eu/esco/skill/sql"
VERSIONS = ServiceVersions(taxonomy="v1.2.1", index="abc123", matcher_prompt="1")
# Text that is only in the CV: it must never reach a log line, a warning or an error.
SECRETS = ("Olena", "Testenko", "Acme Widgets", "Globex", "olena.testenko", "single_column")


def cv_bytes(name: str = "single_column.pdf") -> bytes:
    return (FIXTURES / name).read_bytes()


def reply(tool_input: dict, tokens: tuple = (1000, 200)) -> ToolCallResult:
    return ToolCallResult(tool_input=tool_input, stop_reason="tool_use",
                          input_tokens=tokens[0], output_tokens=tokens[1], model="fake-model")


def extracted(**changes) -> dict:
    data = {
        "summary": None,
        "skills": [
            {"name": "Python", "kind": "hard", "found_in": "skills_list",
             "quote": "Python, SQL, Apache Airflow"},
            {"name": "SQL", "kind": "hard", "found_in": "skills_list",
             "quote": "Python, SQL, Apache Airflow"},
        ],
        "experience": [
            {"title": "Data Engineer", "company": "Acme Widgets LLC", "location": None,
             "start": "2021", "end": "2024", "description": None, "skills_used": ["Python"],
             "quote": "Data Engineer\nAcme Widgets LLC"},
        ],
        "education": [], "languages": [], "certifications": [],
    }
    data.update(changes)
    return data


def caller(*responses) -> FakeToolCaller:
    return FakeToolCaller(list(responses) or [reply(extracted())])


class FakeNormalizer:
    """Same call as NormalizationService.normalize_profile; Python and SQL are known."""

    known = {"python": PYTHON, "sql": SQL}

    def __init__(self, warnings: list[str] | None = None) -> None:
        self.profiles: list = []
        self.warnings = warnings or []

    def normalize_profile(self, profile) -> ProfileSkills:
        self.profiles.append(profile)
        matches = {}
        for path, raw in profile_skill_paths(profile):
            concept = self.known.get(raw.lower())
            matches[path] = SkillMatch(
                raw=raw, normalized=raw.lower(), concept_id=concept,
                preferred_label=raw if concept else None,
                tier=MatchTier.EXACT if concept else MatchTier.UNMATCHED,
                warnings=[] if concept else list(self.warnings),
            )
        return ProfileSkills(matches=matches, versions=VERSIONS)


def vacancy(vid: str, required: list[str]) -> VacancyFeatures:
    refs = [SkillRef(concept_id=FakeNormalizer.known.get(s), raw=s, normalized=s)
            for s in required]
    return VacancyFeatures(id=vid, source="synthetic", title=f"Role {vid}", required_skills=refs)


VACANCIES = [vacancy("both", ["python", "sql"]), vacancy("one", ["python", "rust"]),
             vacancy("none", ["rust", "go"])]


def run(content: bytes | None = None, filename: str = "cv.pdf", **kwargs):
    kwargs.setdefault("caller", caller())
    kwargs.setdefault("normalizer", FakeNormalizer())
    kwargs.setdefault("embedder", FakeEmbedder())
    kwargs.setdefault("today", TODAY)
    vacancies = kwargs.pop("vacancies", VACANCIES)
    return analyze_cv(cv_bytes() if content is None else content, filename, vacancies, **kwargs)


def error_of(**kwargs) -> PipelineError:
    with pytest.raises(PipelineError) as info:
        run(**kwargs)
    return info.value


# --- the whole path ----------------------------------------------------------------


def test_a_pdf_goes_through_every_stage() -> None:
    result = run()

    assert [s.name for s in result.profile.skills] == ["Python", "SQL"]
    assert set(result.skill_matches) == {"skills[0]", "skills[1]", "experience[0].skills_used[0]"}
    assert result.candidate_skills.keys == {PYTHON, SQL}
    ranked = [r.vacancy.id for r in result.ranking.results]
    assert ranked[0] == "both" and ranked[-1] == "none" and len(ranked) == 3
    assert (result.input_tokens, result.output_tokens) == (1000, 200)
    assert set(result.timings_ms) == {*pipeline.STAGES, "total"}


def test_the_result_says_what_produced_it() -> None:
    versions = run().versions

    assert versions.extraction_model == "fake-model"
    assert versions.normalization == VERSIONS
    assert versions.embedding and versions.candidate_text
    assert versions.scorer == "linear" and versions.ranking
    assert versions.profile_schema == "1.0"


def test_a_docx_works_and_its_parse_warning_becomes_a_code() -> None:
    result = run(cv_bytes("table_cv.docx"), "resume.docx")

    assert any(w.stage == "parse" and w.code == "text_boxes_skipped" for w in result.warnings)


def test_the_extension_is_case_insensitive() -> None:
    assert run(filename="MY CV.PDF").profile.skills


def test_the_result_is_json_serialisable() -> None:
    json.loads(run().model_dump_json())


def test_calls_may_run_in_parallel_without_sharing_anything() -> None:
    results: list = []

    def work() -> None:
        results.append(run())

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 4
    assert all([r.vacancy.id for r in x.ranking.results] == ["both", "one", "none"]
               for x in results)


# --- the upload --------------------------------------------------------------------


@pytest.mark.parametrize("name", ["cv.txt", "cv", "cv.pdf.exe", "cv.doc", ""])
def test_unsupported_files_are_refused_before_anything_is_read_or_sent(name: str) -> None:
    fake = caller()

    err = error_of(filename=name, caller=fake)

    assert (err.stage, err.code) == ("parse", "unsupported_type")
    assert fake.calls == []


def test_the_size_limit_is_inclusive() -> None:
    content = cv_bytes()

    assert run(content, max_bytes=len(content)).profile.skills
    err = error_of(content=content, max_bytes=len(content) - 1)
    assert (err.stage, err.code) == ("parse", "too_large")


def test_the_default_limit_is_five_megabytes() -> None:
    assert pipeline.MAX_UPLOAD_BYTES == 5 * 1024 * 1024
    assert error_of(content=b"%PDF" + b"0" * pipeline.MAX_UPLOAD_BYTES).code == "too_large"


def test_a_corrupt_file_is_a_parse_error() -> None:
    err = error_of(content=cv_bytes("corrupt.pdf"))

    assert (err.stage, err.code) == ("parse", "corrupt")


def test_a_scanned_pdf_has_no_text() -> None:
    fake = caller()

    err = error_of(content=cv_bytes("image_only.pdf"), caller=fake)

    assert (err.stage, err.code) == ("parse", "no_text")
    assert fake.calls == []


@pytest.mark.parametrize("reason", ["encrypted", "too_many_pages", "corrupt", "unsupported_type"])
def test_parser_reasons_pass_through_unchanged(reason, monkeypatch) -> None:
    def refuse(path):
        raise ParseError(reason, "secret detail from the parser")

    monkeypatch.setattr(pipeline, "parse", refuse)

    err = error_of()

    assert err.code == reason and "secret" not in str(err)


def test_the_upload_is_parsed_from_a_private_file_that_is_always_removed(
    monkeypatch, temp_dir
) -> None:
    seen: list[Path] = []
    real = pipeline.parse

    def spy(path):
        seen.append(Path(path))
        assert Path(path).is_file()
        return real(path)

    monkeypatch.setattr(pipeline, "parse", spy)

    run(filename="Olena Testenko CV.pdf")
    with pytest.raises(PipelineError):
        run(content=b"not a pdf at all")

    assert [p.name for p in seen] == ["cv.pdf", "cv.pdf"]  # the upload's own name is not used
    assert all(p.parent.parent == temp_dir for p in seen)  # inside the temp folder
    assert all(not p.exists() and not p.parent.exists() for p in seen)
    assert list(temp_dir.iterdir()) == []


def test_the_file_is_removed_even_when_a_stage_blows_up(monkeypatch, temp_dir) -> None:
    seen: list[Path] = []

    def explode(path):
        seen.append(Path(path))
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "parse", explode)

    with pytest.raises(RuntimeError):
        run()

    assert not seen[0].parent.exists() and list(temp_dir.iterdir()) == []


def test_nothing_is_written_to_the_working_directory(tmp_path, monkeypatch) -> None:
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    run()

    assert list(work.iterdir()) == []  # in particular, no extraction cache


def test_the_default_temp_folder_is_jobpdf_temp(monkeypatch) -> None:
    import jobpdf

    monkeypatch.undo()  # drop the fixture's redirect to see the real default

    assert pipeline.TEMP_DIR == Path(jobpdf.__file__).parent / "temp"


def test_the_temp_folder_is_created_on_demand(temp_dir) -> None:
    assert not temp_dir.exists()

    run()

    assert temp_dir.is_dir() and list(temp_dir.iterdir()) == []


def test_an_explicit_temp_dir_wins(tmp_path) -> None:
    other = tmp_path / "elsewhere"

    run(temp_dir=other)

    assert other.is_dir()


def test_stale_upload_directories_are_swept_but_fresh_and_foreign_ones_stay(
    temp_dir,
) -> None:
    import os
    import time

    temp_dir.mkdir()
    stale, fresh, foreign = (temp_dir / "jobpdf-cv-old", temp_dir / "jobpdf-cv-new",
                             temp_dir / "keep-me")
    for folder in (stale, fresh, foreign):
        folder.mkdir()
        (folder / "cv.pdf").write_bytes(b"x")
    (temp_dir / ".gitkeep").write_text("")
    old = time.time() - pipeline.STALE_AFTER_SECONDS - 60
    os.utime(stale, (old, old))
    os.utime(foreign, (old, old))

    run()  # every call sweeps first

    assert not stale.exists()
    assert fresh.exists() and foreign.exists() and (temp_dir / ".gitkeep").exists()


def test_remove_stale_temp_reports_how_many(temp_dir) -> None:
    import os
    import time

    temp_dir.mkdir()
    for name in ("jobpdf-cv-a", "jobpdf-cv-b"):
        (temp_dir / name).mkdir()
        old = time.time() - 7200
        os.utime(temp_dir / name, (old, old))

    assert pipeline.remove_stale_temp(temp_dir) == 2
    assert pipeline.remove_stale_temp(temp_dir) == 0


def test_the_extraction_cache_is_never_used(monkeypatch) -> None:
    seen: dict = {}
    real = pipeline.extract_profile

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "extract_profile", spy)

    run()

    assert seen["use_cache"] is False


# --- extraction failures -----------------------------------------------------------


def test_without_an_api_key_it_says_so() -> None:
    err = error_of(caller=None)

    assert (err.stage, err.code) == ("extraction", "llm_not_configured")


def test_an_sdk_error_becomes_a_code_and_never_carries_cv_text(caplog) -> None:
    class RateLimit(Exception):
        pass

    fake = caller(RateLimit("rate limited while reading Olena Testenko of Acme Widgets"))

    with caplog.at_level(logging.DEBUG):
        err = error_of(caller=fake)

    assert (err.stage, err.code) == ("extraction", "llm_error")
    assert err.__cause__ is None and err.__suppress_context__
    assert not any(s in str(err) for s in SECRETS)
    assert "RateLimit" in caplog.text  # the type is kept for the operator
    assert not any(s in caplog.text for s in SECRETS)


def test_a_reply_without_the_tool_call_is_an_llm_error() -> None:
    err = error_of(caller=caller(LLMResponseError("no tool call")))

    assert (err.stage, err.code) == ("extraction", "llm_error")


def test_invalid_output_twice_is_extraction_invalid() -> None:
    bad = extracted(experience=[{**extracted()["experience"][0], "start": "May 2021"}])
    fake = caller(reply(bad), reply(bad))

    err = error_of(caller=fake)

    assert (err.stage, err.code) == ("extraction", "extraction_invalid")
    assert len(fake.calls) == 2  # the one corrective retry


def test_one_invalid_answer_is_corrected_and_tokens_add_up() -> None:
    bad = extracted(experience=[{**extracted()["experience"][0], "start": "May 2021"}])

    result = run(caller=caller(reply(bad, (500, 50)), reply(extracted(), (600, 60))))

    assert (result.input_tokens, result.output_tokens) == (1100, 110)


# --- soft failures -----------------------------------------------------------------


def test_free_text_extraction_warnings_become_codes_without_cv_text() -> None:
    wrong = extracted(skills=[{"name": "Python", "kind": "hard", "found_in": "skills_list",
                               "quote": "Olena Testenko loves Python"}])

    result = run(caller=caller(reply(wrong)))

    dumped = json.dumps([w.model_dump() for w in result.warnings])
    assert any(w.code == "quote_not_found" for w in result.warnings)
    assert not any(s in dumped for s in SECRETS)


@pytest.mark.parametrize(("text", "code"), [
    ("quote not found: 'Olena said'", "quote_not_found"),
    ("possible personal data in output at skills[0].name", "possible_personal_data"),
    ("experience 'Engineer at Acme Widgets': no start date", "experience_no_start_date"),
    ("experience 'Engineer at Acme Widgets': no end date", "experience_no_end_date"),
    ("experience 'Engineer at Acme Widgets': ends before it starts",
     "experience_ends_before_start"),
    ("something brand new about Olena", "extraction_warning"),
    ("", "extraction_warning"),
])
def test_classify_extraction_warning(text: str, code: str) -> None:
    assert classify_extraction_warning(text) == code


def test_every_extraction_warning_the_pipeline_can_get_is_classified() -> None:
    # JM-11 owns the wording; if it changes, the fragments here must follow.
    from jobpdf.extraction import postprocess

    source = Path(postprocess.__file__).read_text(encoding="utf-8")
    for fragment, _ in pipeline.EXPERIENCE_WARNING_FRAGMENTS:
        assert fragment in source


def test_embedding_failure_degrades_to_skills_only_ranking(caplog) -> None:
    class Broken:
        def embed(self, texts, show_progress=False):
            raise OSError("model files missing for Olena Testenko")

    with caplog.at_level(logging.DEBUG):
        result = run(embedder=Broken())

    assert any((w.stage, w.code) == ("embedding", "embedding_failed") for w in result.warnings)
    assert result.versions.embedding is None and result.versions.candidate_text is None
    assert [r.vacancy.id for r in result.ranking.results][0] == "both"
    assert "OSError" in caplog.text and not any(s in caplog.text for s in SECRETS)


def test_no_vacancies_is_a_warning_not_an_error() -> None:
    result = run(vacancies=[])

    assert result.ranking.results == []
    assert any((w.stage, w.code) == ("ranking", "no_vacancies") for w in result.warnings)


def test_repeated_codes_are_counted_once_per_stage() -> None:
    cv = extracted(skills=[
        {"name": n, "kind": "hard", "found_in": "skills_list", "quote": "Python, SQL"}
        for n in ("Rust", "Go", "Zig")
    ], experience=[])

    result = run(caller=caller(reply(cv)), normalizer=FakeNormalizer(["index_unavailable"]))

    found = [w for w in result.warnings if w.code == "index_unavailable"]
    assert [(w.stage, w.count) for w in found] == [("normalization", 3)]
    assert sum(1 for m in result.skill_matches.values() if m.concept_id is None) == 3


def test_warnings_come_in_stage_order() -> None:
    result = run(cv_bytes("table_cv.docx"), "cv.docx", vacancies=[],
                 caller=caller(reply(extracted(skills=[
                     {"name": "Python", "kind": "hard", "found_in": "skills_list",
                      "quote": "not in the cv"}]))))

    order = [pipeline.STAGES.index(w.stage) for w in result.warnings]
    assert order == sorted(order) and len(set(order)) > 2


# --- plumbing ----------------------------------------------------------------------


def test_a_given_scorer_and_adjustments_reach_ranking() -> None:
    class Flat:
        name = "flat"

        def score(self, components):
            return 0.5

    result = run(scorer=Flat(), adjustments={"none": VacancyAdjustment(hidden=True)})

    assert result.versions.scorer == "flat"
    assert [r.vacancy.id for r in result.ranking.hidden] == ["none"]
    assert result.ranking.hidden_count == 1


def test_the_linear_scorer_is_the_default() -> None:
    assert run().ranking.weights == LinearScorer().weights


def test_the_shared_normalizer_is_only_built_when_none_is_given(monkeypatch) -> None:
    built: list[int] = []

    def build():
        built.append(1)
        return FakeNormalizer()

    monkeypatch.setattr(pipeline, "get_normalization_service", build)

    run(normalizer=FakeNormalizer())
    assert built == []
    run(normalizer=None)
    assert built == [1]


def test_the_vacancies_are_not_changed() -> None:
    before = [v.model_dump() for v in VACANCIES]

    run()

    assert [v.model_dump() for v in VACANCIES] == before


def test_logs_hold_counts_and_codes_only(caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        run(cv_bytes("table_cv.docx"), "Olena Testenko.docx")

    assert "analysed a CV" in caplog.text
    assert not any(s in caplog.text for s in (*SECRETS, "Python", "SQL", "Data Engineer"))


def test_quotes_are_grounded_against_the_text_the_llm_saw() -> None:
    # the LLM never sees the contact line (it is masked), so a quote may hold the mask
    masked = "█" * 26 + " | " + "█" * 16 + " | Kyiv, Ukraine"
    cv = extracted(skills=[{"name": "Python", "kind": "hard", "found_in": "other",
                            "quote": masked}], experience=[])

    result = run(caller=caller(reply(cv)))

    assert result.grounding.counts[GroundingStatus.UNGROUNDED] == 0
    assert not any(w.stage == "grounding" for w in result.warnings)
