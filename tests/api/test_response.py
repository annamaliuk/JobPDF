import json

import pytest
from pydantic import ValidationError

from jobpdf.api import response
from jobpdf.api.response import (
    GENERIC_DETAIL,
    SCHEMA_VERSION,
    WARNING_DETAILS,
    AnalysisResponse,
    Insights,
    SkillToLearn,
    WarningOut,
    WithinReach,
    build_response,
    skill_from_ref,
    unknown_codes,
    warning_out,
)
from jobpdf.extraction import grounding
from jobpdf.normalization import matcher
from jobpdf.ranking import overlap, score, semantic
from jobpdf.ranking.models import SkillRef
from jobpdf.ranking.score import Band, VacancyAdjustment
from tests.pipeline.fakes import (
    PYTHON,
    SECRETS,
    SQL,
    VACANCIES,
    FakeNormalizer,
    caller,
    extracted,
    reply,
    run,
)


@pytest.fixture
def analysed():
    return run()


@pytest.fixture
def full(analysed) -> AnalysisResponse:
    return build_response(analysed)


def match(resp: AnalysisResponse, vacancy_id: str):
    return next(m for m in resp.matches if m.vacancy.id == vacancy_id)


# --- the whole response ------------------------------------------------------------


def test_everything_is_in_one_page_by_default(full) -> None:
    assert [m.vacancy.id for m in full.matches] == ["both", "one", "none"]
    assert [m.rank for m in full.matches] == [1, 2, 3]
    assert full.pagination.model_dump() == {"page": 1, "page_size": 3, "total": 3, "pages": 1}


def test_the_response_round_trips_through_json(full) -> None:
    again = AnalysisResponse.model_validate_json(full.model_dump_json())

    assert again == full


def test_the_contract_rejects_unknown_fields_and_cannot_be_changed(full) -> None:
    with pytest.raises(ValidationError):
        WarningOut(code="x", detail="y", surprise=1)
    with pytest.raises(ValidationError):
        full.hidden_count = 5


def test_internal_quotes_and_offsets_never_leave(full) -> None:
    keys = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            keys.update(node)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(json.loads(full.model_dump_json()))

    assert not keys & {"quote", "quotes", "found_in", "start_offset", "groundings", "vectors"}


# --- the profile -------------------------------------------------------------------


def test_profile_skills_say_how_they_were_read(full) -> None:
    skills = {s.name: s for s in full.profile.skills}

    assert skills["Python"].recognised and skills["Python"].concept_id == PYTHON
    assert skills["Python"].label == "Python" and skills["Python"].matched_by == "exact"
    assert skills["SQL"].concept_id == SQL


def test_unrecognised_skills_are_marked_and_have_no_label() -> None:
    cv = extracted(skills=[{"name": "Zorblax", "kind": "hard", "found_in": "skills_list",
                            "quote": "Python, SQL, Apache Airflow"}], experience=[])

    out = build_response(run(caller=caller(reply(cv)))).profile.skills[0]

    assert (out.name, out.recognised, out.label, out.concept_id) == ("Zorblax", False, None, None)
    assert out.matched_by == "unmatched"


def test_jobs_carry_their_own_normalised_skills(full) -> None:
    job = full.profile.experience[0]

    assert (job.title, job.company, job.start, job.end) == (
        "Data Engineer", "Acme Widgets LLC", "2021", "2024")
    assert [(s.name, s.concept_id) for s in job.skills] == [("Python", PYTHON)]


def test_total_experience_and_empty_lists_come_through(full, analysed) -> None:
    assert full.profile.total_experience_months == analysed.profile.total_experience_months
    assert full.profile.education == [] and full.profile.certifications == []


# --- matches -----------------------------------------------------------------------


def test_matched_and_missing_skills_with_counts(full) -> None:
    one = match(full, "one")

    assert [s.name for s in one.required.matched] == ["python"]
    assert [s.name for s in one.required.missing] == ["rust"]
    assert (one.required.matched_count, one.required.count) == (1, 2)
    assert one.required.matched[0].recognised and not one.required.missing[0].recognised
    assert one.preferred.count == 0


def test_components_and_fit_are_copied_unchanged(full, analysed) -> None:
    original = analysed.ranking.results[0]
    out = full.matches[0]

    assert out.fit == original.fit and out.band is original.band
    assert out.components.skill == original.components.skill
    assert out.components.required_coverage == 1.0
    assert out.components.semantic == original.components.semantic


def test_the_vacancy_block_is_what_the_user_needs_to_open_it(full) -> None:
    v = match(full, "both").vacancy

    assert (v.id, v.source, v.title) == ("both", "synthetic", "Role both")


def test_a_band_cap_is_visible_and_hidden_vacancies_are_counted_not_listed() -> None:
    adjustments = {"both": VacancyAdjustment(band_cap=Band.WEAK, reasons=["location"]),
                   "none": VacancyAdjustment(hidden=True, reasons=["visa"])}

    resp = build_response(run(adjustments=adjustments))

    capped = match(resp, "both")
    assert capped.band is Band.WEAK and capped.capped_from is not None
    assert capped.adjustment_reasons == ["location"]
    assert "none" not in [m.vacancy.id for m in resp.matches]
    assert resp.hidden_count == 1 and resp.pagination.total == 2


def test_uncapped_matches_have_no_capped_from(full) -> None:
    assert all(m.capped_from is None for m in full.matches)


def test_band_counts_cover_every_visible_match(full) -> None:
    assert sum(full.band_counts.values()) == 3
    assert set(full.band_counts) == set(Band)


# --- pagination --------------------------------------------------------------------


def test_pages_slice_the_same_ordered_list(analysed) -> None:
    first = build_response(analysed, page=1, page_size=2)
    second = build_response(analysed, page=2, page_size=2)

    assert [m.vacancy.id for m in first.matches] == ["both", "one"]
    assert [(m.rank, m.vacancy.id) for m in second.matches] == [(3, "none")]
    assert first.pagination.pages == second.pagination.pages == 2
    assert second.pagination.total == 3
    assert first.band_counts == second.band_counts  # always about all matches


def test_a_page_past_the_end_is_empty_not_an_error(analysed) -> None:
    resp = build_response(analysed, page=9, page_size=2)

    assert resp.matches == [] and resp.pagination.page == 9


def test_no_vacancies_is_one_empty_page() -> None:
    resp = build_response(run(vacancies=[]))

    assert resp.matches == []
    assert resp.pagination.model_dump() == {"page": 1, "page_size": 1, "total": 0, "pages": 1}
    assert any(w.code == "no_vacancies" for w in resp.warnings)


@pytest.mark.parametrize(("page", "size"), [(0, None), (-1, 5), (1, 0), (1, -3)])
def test_bad_paging_arguments_are_refused(analysed, page, size) -> None:
    with pytest.raises(ValueError):
        build_response(analysed, page=page, page_size=size)


# --- warnings ----------------------------------------------------------------------


def test_warnings_have_a_code_a_sentence_a_stage_and_a_count(full) -> None:
    cv = extracted(skills=[{"name": n, "kind": "hard", "found_in": "skills_list",
                            "quote": "Python, SQL, Apache Airflow"} for n in ("Rust", "Go")],
                   experience=[])
    resp = build_response(
        run(caller=caller(reply(cv)), normalizer=FakeNormalizer(["index_unavailable"])))

    found = next(w for w in resp.warnings if w.code == "index_unavailable")
    assert (found.stage, found.count) == ("normalization", 2)
    assert found.detail == WARNING_DETAILS["index_unavailable"]


def test_an_unknown_code_keeps_its_code_with_a_generic_detail() -> None:
    out = warning_out("brand_new_code", "ranking", 3)

    assert (out.code, out.detail, out.count) == ("brand_new_code", GENERIC_DETAIL, 3)


def test_no_detail_contains_cv_text(full) -> None:
    wrong = extracted(skills=[{"name": "Python", "kind": "hard", "found_in": "skills_list",
                               "quote": "Olena Testenko loves Python"}])

    resp = build_response(run(caller=caller(reply(wrong))))

    assert any(w.code == "quote_not_found" for w in resp.warnings)
    details = " ".join(w.detail for w in resp.warnings)
    assert not any(s in details for s in SECRETS)


def test_every_code_the_stages_can_emit_has_a_sentence() -> None:
    from jobpdf import pipeline

    codes = {
        *pipeline.PARSE_WARNING_CODES.values(), *pipeline.SECTION_WARNING_CODES.values(),
        *(code for _, code in pipeline.EXPERIENCE_WARNING_FRAGMENTS),
        pipeline.NO_VACANCIES, pipeline.EMBEDDING_FAILED, pipeline.EXTRACTION_WARNING,
        "parse_warning", "sections_warning", "quote_not_found", "possible_personal_data",
        grounding.WARNING_UNGROUNDED, grounding.WARNING_FUZZY, grounding.WARNING_AMBIGUOUS,
        grounding.WARNING_LENGTH_MISMATCH,
        matcher.EMPTY_SKILL, matcher.AMBIGUOUS_ALIAS, matcher.NO_VECTOR_THRESHOLD,
        matcher.NO_CANDIDATES, matcher.LLM_UNAVAILABLE, matcher.INDEX_UNAVAILABLE,
        matcher.LLM_CHOICE_NOT_IN_CANDIDATES, matcher.LLM_ITEM_MISSING,
        matcher.LLM_INVALID_OUTPUT,
        semantic.EMPTY_EXPERIENCE_ITEM, semantic.NO_EXPERIENCE_TEXT, semantic.TRUNCATED,
        semantic.NO_RESPONSIBILITIES, semantic.NO_VACANCY_EMBEDDING,
        semantic.STALE_VACANCY_EMBEDDING, semantic.STALE_VACANCY_TEXT,
        semantic.NO_CANDIDATE_TEXT, semantic.DIMENSION_MISMATCH, semantic.UNCALIBRATED,
        overlap.EMPTY_SKILL, overlap.DUPLICATE_REQUIREMENT, overlap.NO_VACANCY_SKILLS,
        overlap.NO_CANDIDATE_SKILLS,
        score.SEMANTIC_MISSING, score.SKILL_MISSING, score.UNSCORED,
        score.BAND_WITHOUT_SEMANTIC, score.SEMANTIC_UNAVAILABLE, score.DUPLICATE_VACANCY_ID,
    }

    assert unknown_codes(codes) == set()


def test_every_sentence_is_plain_text_a_person_can_read() -> None:
    for code, detail in WARNING_DETAILS.items():
        assert detail.endswith(".") and detail[0].isupper(), code
        assert "{" not in detail and len(detail) <= 100, code
        assert code == code.lower() and " " not in code


def test_match_warnings_are_deduplicated_and_belong_to_ranking() -> None:
    resp = build_response(run(vacancies=[VACANCIES[0].model_copy(update={"required_skills": []})]))

    codes = [w.code for w in resp.matches[0].warnings]
    assert "no_vacancy_skills" in codes and len(codes) == len(set(codes))
    assert {w.stage for w in resp.matches[0].warnings} == {"ranking"}


# --- versions, scoring, insights ---------------------------------------------------


def test_versions_name_everything_that_produced_the_result(full, analysed) -> None:
    v = full.versions

    assert v.response_schema == SCHEMA_VERSION == "1"
    assert v.extraction.model == "fake-model" and v.extraction.profile_schema == "1.0"
    assert v.normalization == analysed.versions.normalization
    assert v.embedding == analysed.versions.embedding
    assert (v.ranking, v.scorer) == (analysed.versions.ranking, "linear")


def test_scoring_explains_the_fit(full) -> None:
    assert full.scoring.scorer == "linear"
    assert full.scoring.weights == {"skill": 0.7, "semantic": 0.3}
    assert {"strong_coverage", "partial_coverage", "sem_min"} <= set(full.scoring.thresholds)


def test_meta_reports_timings_and_tokens(full) -> None:
    assert (full.meta.input_tokens, full.meta.output_tokens) == (1000, 200)
    assert "total" in full.meta.timings_ms


def test_insights_are_empty_until_jm_42_fills_them(full) -> None:
    assert full.insights == Insights() and full.insights.computed is False


def test_given_insights_are_passed_through(analysed) -> None:
    java = skill_from_ref(SkillRef(raw="Rust", normalized="rust"))
    given = Insights(
        computed=True,
        within_reach=[WithinReach(vacancy_id="one", missing=[java])],
        skills_to_learn=[SkillToLearn(skill=java, unlocks=2, vacancy_ids=["one", "both"])],
    )

    assert build_response(analysed, insights=given).insights == given


def test_the_internal_result_is_not_changed(analysed) -> None:
    before = analysed.model_dump_json()

    build_response(analysed, page=2, page_size=1)

    assert analysed.model_dump_json() == before


def test_a_vacancys_codes_merge_from_every_source_once_each_in_order(analysed) -> None:
    fit = analysed.ranking.results[0]
    crafted = fit.model_copy(update={
        "warnings": ["unscored", "semantic_missing"],
        "overlap": fit.overlap.model_copy(update={"warnings": ["unscored", "empty_skill"]}),
        "semantic": fit.semantic.model_copy(update={"warnings": ["truncated", "empty_skill"]}),
    })

    codes = [w.code for w in response._match(crafted, 1).warnings]

    assert codes == ["unscored", "semantic_missing", "empty_skill", "truncated"]


def test_a_job_lists_its_own_skills_not_the_cvs_skill_list() -> None:
    cv = extracted(
        skills=[{"name": "Python", "kind": "hard", "found_in": "skills_list",
                 "quote": "Python, SQL, Apache Airflow"}],
        experience=[{**extracted()["experience"][0], "skills_used": ["SQL", "Zorblax"]}],
    )

    job = build_response(run(caller=caller(reply(cv)))).profile.experience[0]

    assert [(s.name, s.recognised) for s in job.skills] == [("SQL", True), ("Zorblax", False)]
