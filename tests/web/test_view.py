"""JM-43: the report's wording. Everything here works on real ``AnalysisResponse`` objects."""

from datetime import datetime, timezone

import pytest

from jobpdf.api.response import (
    AnalysisResponse,
    Insights,
    MatchOut,
    SkillGroupOut,
    SkillOut,
    SkillToLearn,
    WarningOut,
    WithinReach,
    build_response,
)
from jobpdf.ranking.score import Band
from jobpdf.web import view
from jobpdf.web.view import (
    BAND_LABELS,
    BEST_COUNT,
    MAX_MISSING_SHOWN,
    MAX_SEGMENTS,
    NO_JOBS,
    NO_REQUIRED,
    build_report,
    percent,
    plural,
    safe_url,
)
from tests.pipeline.fakes import run


@pytest.fixture(scope="module")
def response() -> AnalysisResponse:
    return build_response(run())


def skill(name: str, label: str | None = None, *, concept: str | None = "c", low=False):
    return SkillOut(name=name, label=label, concept_id=concept if label else None,
                    recognised=label is not None, low_confidence=low, matched_by="exact")


def group(have: int, total: int, missing: list[SkillOut] | None = None) -> SkillGroupOut:
    gaps = missing
    if gaps is None:
        gaps = [skill(f"m{i}", f"M{i}") for i in range(total - have)]
    return SkillGroupOut(matched=[skill(f"h{i}", f"H{i}") for i in range(have)], missing=gaps,
                         matched_count=have, count=total)


def one_match(response: AnalysisResponse, **changes) -> MatchOut:
    vacancy_changes = changes.pop("vacancy", {})
    match = response.matches[0]
    return match.model_copy(update={
        **changes, "vacancy": match.vacancy.model_copy(update=vacancy_changes)})


def report_of(response: AnalysisResponse, *matches: MatchOut, **changes):
    return build_report(response.model_copy(update={"matches": list(matches), **changes}))


def job_of(response: AnalysisResponse, **changes):
    return report_of(response, one_match(response, **changes)).best[0]


# --- the real pipeline's response ----------------------------------------------------------


def test_a_real_response_becomes_a_report_with_every_section(response) -> None:
    report = build_report(response)

    assert [f.label for f in report.figures] == [
        "Experience", "Skills recognised", "Strong matches", "Jobs compared"]
    assert [j.title for j in report.best] == ["Role both", "Role one", "Role none"]
    assert report.best[0].coverage == "You have 2 of 2 required skills."
    assert report.best[1].missing == "Missing: rust."
    assert [s.label for s in report.skills] == ["Python", "SQL"]
    assert report.empty is None and report.reach == [] and report.learn == []


# --- figures -----------------------------------------------------------------------------------


@pytest.mark.parametrize("months, text", [
    (0, "0 months"), (1, "1 month"), (11, "11 months"), (12, "1 year"), (13, "1 year 1 month"),
    (30, "2 years 6 months"), (24, "2 years")])
def test_experience_is_written_in_years_and_months(response, months, text) -> None:
    profile = response.profile.model_copy(update={"total_experience_months": months})
    figures = build_report(response.model_copy(update={"profile": profile})).figures

    assert (figures[0].label, figures[0].value) == ("Experience", text)


def test_experience_is_left_out_when_the_cv_does_not_say(response) -> None:
    profile = response.profile.model_copy(update={"total_experience_months": None})

    figures = build_report(response.model_copy(update={"profile": profile})).figures

    assert [f.label for f in figures][0] == "Skills recognised"


def test_figures_count_skills_once_and_strong_matches_and_jobs(response) -> None:
    profile = response.profile.model_copy(update={"skills": [
        skill("Python", "Python", concept="a"), skill("python 3", "Python", concept="a"),
        skill("???", None)]})
    counts = {**response.band_counts, Band.STRONG: 4}
    pagination = response.pagination.model_copy(update={"total": 17})

    figures = {f.label: f.value for f in build_report(response.model_copy(update={
        "profile": profile, "band_counts": counts, "pagination": pagination})).figures}

    assert figures["Skills recognised"] == "1 of 2"
    assert figures["Strong matches"] == "4" and figures["Jobs compared"] == "17"


def test_the_summary_is_trimmed_and_an_empty_one_is_none(response) -> None:
    for text, expected in (("  A developer.  ", "A developer."), ("   ", None), (None, None)):
        profile = response.profile.model_copy(update={"summary": text})
        assert build_report(response.model_copy(update={"profile": profile})).summary == expected


# --- a job row ---------------------------------------------------------------------------------


@pytest.mark.parametrize("have, total, sentence, short", [
    (7, 9, "You have 7 of 9 required skills.", "7 of 9"),
    (1, 1, "You have 1 of 1 required skill.", "1 of 1"),
    (0, 3, "You have 0 of 3 required skills.", "0 of 3"),
    (0, 0, NO_REQUIRED, "none listed")])
def test_the_coverage_sentence(response, have, total, sentence, short) -> None:
    job = job_of(response, required=group(have, total))

    assert (job.coverage, job.required_short) == (sentence, short)


def test_the_meter_has_one_segment_per_required_skill_filled_for_the_ones_you_have(
        response) -> None:
    assert job_of(response, required=group(7, 9)).segments == (True,) * 7 + (False,) * 2
    assert job_of(response, required=group(0, 3)).segments == (False,) * 3
    assert job_of(response, required=group(0, 0)).segments == ()


def test_a_long_requirement_list_is_scaled_to_the_widest_meter(response) -> None:
    assert job_of(response, required=group(10, 20)).segments == (
        (True,) * 8 + (False,) * 8)
    assert len(job_of(response, required=group(1, 200)).segments) == MAX_SEGMENTS
    assert job_of(response, required=group(16, 16)).segments == (True,) * 16


def test_missing_skills_are_named_by_standard_label_else_by_wording(response) -> None:
    missing = [skill("k8s", "Kubernetes"), skill("Terraform", None)]

    assert job_of(response, required=group(1, 3, missing)).missing == (
        "Missing: Kubernetes, Terraform.")
    assert job_of(response, required=group(3, 3, [])).missing is None


def test_only_the_first_missing_skills_are_listed_and_the_rest_counted(response) -> None:
    missing = [skill(f"s{i}", f"S{i}") for i in range(MAX_MISSING_SHOWN + 2)]

    text = job_of(response, required=group(0, 8, missing)).missing

    assert text == "Missing: S0, S1, S2, S3, S4, S5 and 2 more."
    exactly = job_of(response, required=group(0, 6, missing[:6])).missing
    assert exactly == "Missing: S0, S1, S2, S3, S4, S5."


def test_nice_to_have_is_mentioned_only_when_the_job_lists_some(response) -> None:
    assert job_of(response, preferred=group(1, 3)).nice_to_have == "Nice to have: 1 of 3."
    assert job_of(response, preferred=group(0, 0)).nice_to_have is None


def test_a_job_whose_band_was_lowered_says_so(response) -> None:
    job = job_of(response, band=Band.PARTIAL, capped_from=Band.STRONG)

    assert job.lowered == "Lowered from strong to partial after an eligibility check."
    assert job_of(response, capped_from=None).lowered is None


@pytest.mark.parametrize("fit, text", [(0.824, "82%"), (0.825, "83%"), (1.0, "100%"),
                                       (0.0, "0%"), (0.005, "1%"), (0.004, "0%"), (None, None)])
def test_fit_is_a_whole_percent(response, fit, text) -> None:
    assert job_of(response, fit=fit).fit == text
    assert percent(fit) == text


def test_every_band_has_a_label_and_the_row_carries_it(response) -> None:
    assert set(BAND_LABELS) == {band.value for band in Band}
    for band in Band:
        job = job_of(response, band=band)
        assert (job.band, job.band_label) == (band.value, BAND_LABELS[band.value])


def test_the_posting_date_company_and_location_come_through(response) -> None:
    moment = datetime(2026, 10, 9, 15, 0, tzinfo=timezone.utc)
    job = job_of(response, vacancy={"posted_at": moment, "company": "Acme", "location": "EU"})

    assert (job.posted, job.company, job.location) == ("Posted 09 Oct 2026", "Acme", "EU")
    assert job_of(response, vacancy={"posted_at": None}).posted is None


@pytest.mark.parametrize("url, ok", [
    ("https://example.com/job/1", True), ("http://example.com", True),
    ("HTTPS://EXAMPLE.COM", True),
    ("javascript:alert(1)", False), ("JaVaScRiPt:alert(1)", False), ("data:text/html,x", False),
    ("//example.com/x", False), ("example.com/x", False), ("https://", False), ("", False),
    (None, False), ("https://exa mple.com", False), ("https://example.com/\nx", False),
    ("https://example.com/\x00", False), ("ftp://example.com", False),
    ("https://example.com/" + "a" * 2100, False), ("https://[bad", False)])
def test_only_web_addresses_become_links(url, ok) -> None:
    assert (safe_url(url) is not None) is ok
    assert safe_url(url) == (url if ok else None)


def test_the_row_link_is_the_safe_address(response) -> None:
    assert job_of(response, vacancy={"url": "javascript:alert(1)"}).url is None
    assert job_of(response, vacancy={"url": "https://example.com/1"}).url == "https://example.com/1"


# --- the lists --------------------------------------------------------------------------------


def test_the_best_matches_are_the_first_few_and_the_table_has_all(response) -> None:
    matches = [one_match(response).model_copy(update={"rank": i + 1})
               for i in range(BEST_COUNT + 3)]

    report = report_of(response, *matches)

    assert [j.rank for j in report.best] == list(range(1, BEST_COUNT + 1))
    assert len(report.rest) == BEST_COUNT + 3


def test_no_matches_gives_the_empty_message_and_no_rows(response) -> None:
    report = report_of(response)

    assert report.empty == NO_JOBS and report.best == [] and report.rest == []


def test_skills_are_listed_once_and_unsure_ones_flagged(response) -> None:
    profile = response.profile.model_copy(update={"skills": [
        skill("Py", "Python", concept="a"), skill("Python", "Python", concept="a"),
        skill("JS", "JavaScript", concept="b", low=True), skill("???", None),
        skill("???", None), skill("Other", None)]})

    report = build_report(response.model_copy(update={"profile": profile}))

    assert [(s.label, s.unsure) for s in report.skills] == [("Python", False),
                                                            ("JavaScript", True)]
    assert report.unrecognised == ["???", "Other"]


def test_abbreviations_are_explained_only_when_the_wording_differs(response) -> None:
    profile = response.profile.model_copy(update={"skills": [
        skill("JS", "JavaScript", concept="a"), skill("python", "Python", concept="b"),
        skill(" k8s ", "Kubernetes", concept="c"), skill("Odd", None)]})

    report = build_report(response.model_copy(update={"profile": profile}))

    assert report.readings == ["JS was read as JavaScript.", "k8s was read as Kubernetes."]


def test_within_reach_and_skills_to_learn_appear_only_when_computed(response) -> None:
    match = one_match(response, vacancy={"id": "j1", "title": "Data engineer", "company": "Acme"})
    insights = Insights(
        computed=True,
        within_reach=[WithinReach(vacancy_id="j1", missing=[skill("dbt", "dbt"),
                                                              skill("Airflow", None)]),
                      WithinReach(vacancy_id="gone", missing=[skill("x", "X")]),
                      WithinReach(vacancy_id="j1", missing=[])],
        skills_to_learn=[SkillToLearn(skill=skill("dbt", "dbt"), unlocks=3, vacancy_ids=["j1"]),
                         SkillToLearn(skill=skill("Go", None), unlocks=1, vacancy_ids=["j1"])])

    report = report_of(response, match, insights=insights)

    assert report.reach == ["Data engineer at Acme: add dbt and Airflow to reach a strong match."]
    assert report.learn == ["dbt: would make 3 more jobs a strong match.",
                            "Go: would make 1 more job a strong match."]
    off = report_of(response, match, insights=insights.model_copy(update={"computed": False}))
    assert off.reach == [] and off.learn == []


def test_within_reach_without_a_company_has_no_dangling_at(response) -> None:
    match = one_match(response, vacancy={"id": "j1", "title": "Data engineer", "company": None})
    insights = Insights(computed=True, within_reach=[
        WithinReach(vacancy_id="j1", missing=[skill("dbt", "dbt")])])

    assert report_of(response, match, insights=insights).reach == [
        "Data engineer: add dbt to reach a strong match."]


def test_notes_use_the_fixed_sentences_and_count_what_was_left_out(response) -> None:
    warnings = [WarningOut(code="a", detail="One thing happened.", count=1),
                WarningOut(code="b", detail="Another thing happened.", count=3)]

    notes = build_report(response.model_copy(update={"warnings": warnings})).notes
    assert notes == ["One thing happened.", "Another thing happened. (affects 3 items)"]

    one = build_report(response.model_copy(update={"warnings": [], "hidden_count": 1})).notes
    many = build_report(response.model_copy(update={"warnings": [], "hidden_count": 4})).notes
    assert one == ["1 posting left out because your CV suggests you could not apply for them."]
    assert many == ["4 postings left out because your CV suggests you could not apply for them."]
    assert build_report(response.model_copy(update={"warnings": [], "hidden_count": 0})).notes == []


def test_the_scoring_text_is_built_from_the_numbers_the_response_carries(response) -> None:
    scoring = response.scoring.model_copy(update={
        "weights": {"skill": 0.7, "semantic": 0.3},
        "thresholds": {"strong_coverage": 0.8, "partial_coverage": 0.5, "sem_min": 0.3},
        "semantic_calibrated": True})

    lines = build_report(response.model_copy(update={"scoring": scoring})).scoring

    assert lines[0].startswith("Each job's fit is 70% your skills")
    assert "30% how closely your past work resembles" in lines[0]
    assert "at least 80% of the required skills" in lines[1] and "at least 50%" in lines[1]
    assert len(lines) == 2  # calibrated: no "approximate" note


def test_the_scoring_text_copes_with_a_scorer_it_does_not_know(response) -> None:
    scoring = response.scoring.model_copy(update={
        "weights": {"x": 1.0}, "thresholds": {}, "semantic_calibrated": None})

    lines = build_report(response.model_copy(update={"scoring": scoring})).scoring

    assert lines == ["Each job's fit is worked out from your skills and your past work.",
                     "The comparison of past work is approximate for now."]


def test_an_uncalibrated_comparison_is_called_approximate(response) -> None:
    scoring = response.scoring.model_copy(update={"semantic_calibrated": False})

    assert build_report(response.model_copy(update={"scoring": scoring})).scoring[-1] == (
        "The comparison of past work is approximate for now.")


def test_small_helpers() -> None:
    assert plural(1, "skill") == "1 skill" and plural(0, "skill") == "0 skills"
    assert plural(2, "posting") == "2 postings" and plural(2, "match", "matches") == "2 matches"
    assert view.MAX_SEGMENTS == 16


# --- pinned values (the constants and sentences a reader sees, not the names in the code) ---------


def test_the_numbers_that_shape_the_page_are_the_ones_documented() -> None:
    assert (BEST_COUNT, MAX_MISSING_SHOWN, MAX_SEGMENTS) == (5, 6, 16)
    assert view.MAX_URL_CHARS == 2048


def test_the_best_matches_are_the_first_five_and_the_table_has_all(response) -> None:
    matches = [one_match(response, rank=i + 1) for i in range(7)]

    report = report_of(response, *matches)

    assert [j.rank for j in report.best] == [1, 2, 3, 4, 5]
    assert [j.rank for j in report.rest] == [1, 2, 3, 4, 5, 6, 7]


def test_the_labels_are_plain_words() -> None:
    assert BAND_LABELS == {"strong": "Strong match", "partial": "Partial match",
                           "weak": "Weak match", "unscored": "Not scored"}


def test_a_lowered_band_says_from_what_to_what(response) -> None:
    job = job_of(response, band=Band.PARTIAL, capped_from=Band.STRONG)

    assert job.lowered == "Lowered from strong to partial after an eligibility check."


def test_an_address_is_accepted_up_to_exactly_the_length_limit() -> None:
    base = "https://example.com/"

    assert safe_url(base + "a" * (2048 - len(base))) is not None
    assert safe_url(base + "a" * (2049 - len(base))) is None


@pytest.mark.parametrize("url", ["HTTP://example.com/x", "HtTpS://example.com/x"])
def test_the_scheme_is_accepted_in_any_case(url) -> None:
    assert safe_url(url) == url


def test_a_negative_number_of_months_is_shown_as_none(response) -> None:
    profile = response.profile.model_copy(update={"total_experience_months": -5})

    assert build_report(response.model_copy(update={"profile": profile})).figures[0].value == (
        "0 months")


def test_a_meter_rounds_to_the_nearest_segment_not_down(response) -> None:
    # 11 of 20 is 8.8 of 16 segments: nearest is 9, rounding down would show 8
    assert job_of(response, required=group(11, 20)).segments == (True,) * 9 + (False,) * 7
    # and a whole share is exact
    assert job_of(response, required=group(16, 16)).segments == (True,) * 16
    assert job_of(response, required=group(17, 17)).segments == (True,) * 16


def test_the_same_unrecognised_wording_is_listed_once_whatever_its_case_or_spacing(
        response) -> None:
    profile = response.profile.model_copy(update={"skills": [
        skill("Foo", None), skill("foo ", None), skill(" FOO", None), skill("Bar", None)]})

    report = build_report(response.model_copy(update={"profile": profile}))

    assert report.unrecognised == ["Foo", "Bar"]
    assert [f.value for f in report.figures if f.label == "Skills recognised"] == ["0 of 2"]


def test_readings_are_only_for_recognised_skills_that_have_a_standard_name(response) -> None:
    odd = [
        SkillOut(name="myst", label="Mystery", concept_id=None, recognised=False),
        SkillOut(name="nameless", label=None, concept_id="n", recognised=True),
        skill("JS", "JavaScript", concept="js"),
    ]
    profile = response.profile.model_copy(update={"skills": odd})

    report = build_report(response.model_copy(update={"profile": profile}))

    assert report.readings == ["JS was read as JavaScript."]
    assert [s.label for s in report.skills] == ["nameless", "JavaScript"]


def test_one_weight_or_one_threshold_is_not_enough_to_quote_percentages(response) -> None:
    scoring = response.scoring.model_copy(update={
        "weights": {"skill": 0.7}, "thresholds": {"strong_coverage": 0.8},
        "semantic_calibrated": True})

    lines = build_report(response.model_copy(update={"scoring": scoring})).scoring

    assert lines == ["Each job's fit is worked out from your skills and your past work."]
    only_partial = response.scoring.model_copy(update={
        "weights": {"semantic": 0.3}, "thresholds": {"partial_coverage": 0.5},
        "semantic_calibrated": True})
    assert build_report(response.model_copy(update={"scoring": only_partial})).scoring == lines
