"""JM-49, JM-47 and JM-51: hiding or capping jobs the candidate cannot take or chose not to
see, on clear evidence only."""

import pytest
from pydantic import ValidationError

from jobpdf.api.response import build_response
from jobpdf.ranking.eligibility import (
    EXPERIENCE_SHORT,
    HIDING_REASONS,
    NOT_REMOTE_MODES,
    ON_SITE_ABROAD,
    REMOTE_OUTSIDE_REGION,
    RISKY_NICHE,
    abroad_adjustments,
    experience_adjustments,
    is_in_avoided_niche,
    is_on_site_abroad,
    location_adjustments,
    merge_adjustments,
    niche_adjustments,
    normalize_country,
)
from jobpdf.ranking.models import VacancyFeatures
from jobpdf.ranking.score import Band, VacancyAdjustment
from tests.pipeline.fakes import VACANCIES, run


def vac(vid="v", mode=None, country=None, source="s") -> VacancyFeatures:
    return VacancyFeatures(id=vid, source=source, title="T", work_mode=mode, country=country)


@pytest.mark.parametrize("mode, country, home, hidden", [
    ("on_site", "DE", "UA", True),
    ("hybrid", "DE", "UA", True),
    ("on_site", "UA", "UA", False),  # at home
    ("hybrid", "UA", "UA", False),
    ("remote", "DE", "UA", False),  # remote is never hidden here
    ("remote", None, "UA", False),
    (None, "DE", "UA", False),  # mode not known
    ("on_site", None, "UA", False),  # country not known
    (None, None, "UA", False),
])
def test_only_an_on_site_or_hybrid_vacancy_in_another_country_is_hidden(
        mode, country, home, hidden) -> None:
    assert is_on_site_abroad(vac(mode=mode, country=country), home) is hidden


def test_the_modes_that_need_you_there_are_on_site_and_hybrid() -> None:
    assert NOT_REMOTE_MODES == {"on_site", "hybrid"}


@pytest.mark.parametrize("value, code", [
    ("UA", "UA"), ("ua", "UA"), (" pl ", "PL"), ("", None), (None, None), ("ZZ", None),
    ("Ukraine", None), ("UAE", None), ("U", None), ("<script>", None),
])
def test_only_a_known_country_code_is_accepted(value, code) -> None:
    assert normalize_country(value) == code


def test_adjustments_name_the_vacancies_to_hide_and_the_reason() -> None:
    vacancies = [vac("a", "on_site", "DE"), vac("b", "remote", "DE"), vac("c", "on_site", "UA"),
                 vac("d", "hybrid", "PL"), vac("e")]

    adjustments = abroad_adjustments(vacancies, "UA")

    assert set(adjustments) == {"a", "d"}
    for adjustment in adjustments.values():
        assert adjustment.hidden and adjustment.reasons == [ON_SITE_ABROAD]
        assert adjustment.band_cap is None


@pytest.mark.parametrize("home", [None, "", "ZZ", "Ukraine", "  "])
def test_without_a_known_country_nothing_is_hidden(home) -> None:
    assert abroad_adjustments([vac("a", "on_site", "DE")], home) == {}


def test_the_country_is_read_without_regard_to_case() -> None:
    assert set(abroad_adjustments([vac("a", "on_site", "DE")], "ua")) == {"a"}


def test_an_id_two_vacancies_disagree_on_is_not_hidden_for_either() -> None:
    vacancies = [vac("1", "on_site", "DE", source="x"), vac("1", "remote", "DE", source="y"),
                 vac("2", "on_site", "DE", source="x"), vac("2", "on_site", "FR", source="y")]

    assert set(abroad_adjustments(vacancies, "UA")) == {"2"}  # both are abroad and on site


def test_no_vacancies_gives_no_adjustments() -> None:
    assert abroad_adjustments([], "UA") == {}


def test_no_adjustment_object_is_shared_with_a_caller_that_could_change_it() -> None:
    adjustments = abroad_adjustments([vac("a", "on_site", "DE")], "UA")

    with pytest.raises(ValidationError):
        adjustments["a"].hidden = False  # the model is frozen


# --- through the ranking -----------------------------------------------------------------------


def hide(*ids: str, home="UA"):
    vacancies = [v.model_copy(update={"work_mode": "on_site", "country": "DE"})
                 if v.id in ids else v for v in VACANCIES]
    return vacancies, abroad_adjustments(vacancies, home)


def test_a_hidden_vacancy_is_scored_and_listed_but_not_ranked() -> None:
    vacancies, adjustments = hide("both")

    result = run(vacancies=vacancies, adjustments=adjustments)

    assert [r.vacancy.id for r in result.ranking.results] == ["one", "none"]
    assert [r.vacancy.id for r in result.ranking.hidden] == ["both"]
    assert result.ranking.hidden_count == 1
    assert result.ranking.hidden[0].adjustment_reasons == [ON_SITE_ABROAD]
    assert result.ranking.hidden[0].fit is not None  # fully scored, never dropped


def test_the_ranking_of_the_vacancies_that_stay_does_not_change() -> None:
    vacancies, adjustments = hide("both")

    kept = run(vacancies=vacancies, adjustments=adjustments).ranking.results
    plain = [r for r in run().ranking.results if r.vacancy.id != "both"]

    assert [(r.vacancy.id, r.fit, r.band) for r in kept] == [
        (r.vacancy.id, r.fit, r.band) for r in plain]


def test_the_response_says_how_many_were_left_out_and_why() -> None:
    vacancies, adjustments = hide("both", "none")

    response = build_response(run(vacancies=vacancies, adjustments=adjustments))

    assert response.hidden_count == 2 and response.hidden_reasons == {ON_SITE_ABROAD: 2}
    assert [m.vacancy.id for m in response.matches] == ["one"]


def test_a_response_without_hidden_vacancies_has_no_reasons() -> None:
    assert build_response(run()).hidden_reasons == {}


def test_a_vacancy_with_the_same_reason_twice_counts_once() -> None:
    from jobpdf.api.response import _hidden_reasons

    vacancies, adjustments = hide("both")
    hidden = run(vacancies=vacancies, adjustments=adjustments).ranking.hidden
    doubled = [hidden[0].model_copy(update={"adjustment_reasons": ["a", "a", "b"]})]

    assert _hidden_reasons(doubled) == {"a": 1, "b": 1}


# --- JM-47: remote jobs open only elsewhere -----------------------------------------------------


def remote(vid="r", where=None, country=None) -> VacancyFeatures:
    return VacancyFeatures(id=vid, source="s", title="T", work_mode="remote",
                           location_raw=where, country=country)


@pytest.mark.parametrize("where, hidden", [
    ("USA only", True),
    ("USA, Canada", True),
    ("EU only", True),
    ("LATAM", True),
    ("Europe", False),
    ("Ukraine", False),
    ("Poland, Ukraine", False),
    ("Worldwide", False),
    ("CET +/- 3 hours", False),
    (None, False),
])
def test_a_remote_job_is_hidden_only_when_its_place_text_leaves_the_country_out(where, hidden):
    assert ("r" in location_adjustments([remote(where=where)], "UA")) is hidden


def test_a_remote_jobs_country_alone_hides_nothing() -> None:
    assert location_adjustments([remote(country="DE")], "UA") == {}  # may be the company's


def test_the_place_rules_need_a_country() -> None:
    assert location_adjustments([remote(where="USA"), vac("a", "on_site", "DE")], None) == {}


def test_both_place_rules_come_together_with_their_own_reason() -> None:
    adjustments = location_adjustments([remote(where="USA"), vac("a", "on_site", "DE")], "UA")

    assert adjustments["r"].reasons == [REMOTE_OUTSIDE_REGION] and adjustments["r"].hidden
    assert adjustments["a"].reasons == [ON_SITE_ABROAD] and adjustments["a"].hidden


# --- JM-47: years of experience ----------------------------------------------------------------


def asks(years, vid="v") -> VacancyFeatures:
    return VacancyFeatures(id=vid, source="s", title="T", min_years_experience=years)


@pytest.mark.parametrize("years, months, capped", [
    (5, 30, True),  # exactly 2.5 years short
    (5, 31, False),
    (3, 6, True),
    (3, 16, False),
    (3, 0, True),  # no experience at all
    (2, 0, False),  # junior jobs stay as they are
    (None, 0, False),  # the job asks for nothing
    (0, 0, False),
])
def test_a_job_asking_for_clearly_more_experience_is_capped_at_partial(years, months, capped):
    adjustments = experience_adjustments([asks(years)], months)

    assert ("v" in adjustments) is capped
    if capped:
        assert adjustments["v"] == VacancyAdjustment(band_cap=Band.PARTIAL,
                                                     reasons=[EXPERIENCE_SHORT])


def test_unknown_experience_caps_nothing() -> None:
    assert experience_adjustments([asks(10)], None) == {}


def test_merged_adjustments_hide_if_any_does_keep_the_lowest_cap_and_every_reason() -> None:
    merged = merge_adjustments(
        {"a": VacancyAdjustment(band_cap=Band.PARTIAL, reasons=["x"])},
        {"a": VacancyAdjustment(hidden=True, band_cap=Band.WEAK, reasons=["y", "x"]),
         "b": VacancyAdjustment(reasons=["z"])},
    )

    assert merged["a"] == VacancyAdjustment(hidden=True, band_cap=Band.WEAK, reasons=["x", "y"])
    assert merged["b"].reasons == ["z"]


# --- JM-51: industries the person chose to avoid ----------------------------------------------


def tagged(vid="v", niches=(), mode=None, country=None, source="s") -> VacancyFeatures:
    return VacancyFeatures(id=vid, source=source, title="T", niches=list(niches),
                           work_mode=mode, country=country)


def chosen(vacancies, home=None, avoid=None):
    """What the app asks for: the location rules and the industry rule, merged (api._run)."""
    return merge_adjustments(location_adjustments(vacancies, home),
                             niche_adjustments(vacancies, avoid))


def test_the_hiding_reasons_are_the_ones_the_pages_can_word() -> None:
    assert HIDING_REASONS == {ON_SITE_ABROAD, REMOTE_OUTSIDE_REGION, RISKY_NICHE}
    assert RISKY_NICHE == "risky_niche" and EXPERIENCE_SHORT not in HIDING_REASONS


@pytest.mark.parametrize("niches, avoid, hidden", [
    (["gambling"], ["gambling"], True),
    (["gambling", "crypto"], ["crypto"], True),
    (["gambling"], ["crypto"], False),
    (["gambling"], ["crypto", "gambling"], True),
    ([], ["gambling"], False),  # nothing known about it: never hidden
    (["gambling"], [], False),
    (["gambling"], ["nope"], False),
])
def test_a_vacancy_is_hidden_when_it_is_in_an_industry_that_was_chosen(
        niches, avoid, hidden) -> None:
    assert is_in_avoided_niche(tagged(niches=niches), avoid) is hidden
    assert bool(niche_adjustments([tagged("a", niches)], avoid)) is hidden


def test_niche_adjustments_hide_with_the_niche_reason_only() -> None:
    adjustments = niche_adjustments([tagged("a", ["crypto"]), tagged("b"),
                                     tagged("c", ["gambling"])], ["crypto"])

    assert set(adjustments) == {"a"}
    assert adjustments["a"].hidden is True and adjustments["a"].reasons == [RISKY_NICHE]
    assert adjustments["a"].band_cap is None


@pytest.mark.parametrize("avoid", [None, [], (), set(), ["nope"], [""]])
def test_without_a_known_choice_nothing_is_hidden(avoid) -> None:
    assert niche_adjustments([tagged("a", ["crypto"])], avoid) == {}


def test_a_stored_code_that_is_not_in_the_table_never_hides_anything() -> None:
    # e.g. an industry removed from niches.py while stored vacancies still carry its code
    assert niche_adjustments([tagged("a", ["retired"])], ["retired"]) == {}
    assert niche_adjustments([tagged("a", ["retired", "crypto"])], ["retired"]) == {}
    assert set(niche_adjustments([tagged("a", ["retired", "crypto"])],
                                 ["retired", "crypto"])) == {"a"}


def test_an_id_two_vacancies_disagree_on_is_not_hidden_for_either_niche() -> None:
    vacancies = [tagged("1", ["crypto"], source="x"), tagged("1", [], source="y"),
                 tagged("2", ["crypto"], source="x"), tagged("2", ["gambling"], source="y")]

    assert set(niche_adjustments(vacancies, ["crypto", "gambling"])) == {"2"}


def test_no_vacancies_gives_no_niche_adjustments() -> None:
    assert niche_adjustments([], ["gambling"]) == {}


def test_the_choices_join_both_reasons_on_one_adjustment_in_a_fixed_order() -> None:
    vacancies = [tagged("both", ["crypto"], "on_site", "DE"),
                 tagged("abroad", [], "on_site", "DE"),
                 tagged("niche", ["crypto"], "remote", "DE"),
                 tagged("neither", [], "remote", "DE")]

    adjustments = chosen(vacancies, "UA", ["crypto"])

    assert set(adjustments) == {"both", "abroad", "niche"}
    assert adjustments["both"].reasons == [ON_SITE_ABROAD, RISKY_NICHE]
    assert adjustments["abroad"].reasons == [ON_SITE_ABROAD]
    assert adjustments["niche"].reasons == [RISKY_NICHE]
    assert all(a.hidden for a in adjustments.values())


def test_each_choice_works_alone_and_none_hides_nothing() -> None:
    vacancies = [tagged("a", ["crypto"], "on_site", "DE")]

    assert set(chosen(vacancies, "UA")) == {"a"}
    assert set(chosen(vacancies, None, ["crypto"])) == {"a"}
    assert chosen(vacancies) == {}
    assert chosen(vacancies, None, None) == {}
    assert chosen(vacancies, "ZZ", ["nope"]) == {}


def test_joining_does_not_change_what_each_rule_says_alone() -> None:
    vacancies = [tagged("a", ["crypto"], "on_site", "DE"), tagged("b", [], "on_site", "FR"),
                 tagged("c", ["gambling"], "remote", None)]

    joined = chosen(vacancies, "UA", ["gambling", "crypto"])

    assert set(joined) == set(location_adjustments(vacancies, "UA")) | set(
        niche_adjustments(vacancies, ["gambling", "crypto"]))


# --- through the ranking and the response ------------------------------------------------------


def tag(*ids: str, niches=("crypto",), mode=None, country=None):
    return [v.model_copy(update={"niches": list(niches), "work_mode": mode, "country": country})
            if v.id in ids else v for v in VACANCIES]


def test_a_vacancy_in_an_avoided_industry_is_scored_and_listed_but_not_ranked() -> None:
    vacancies = tag("both")

    result = run(vacancies=vacancies, adjustments=chosen(vacancies, None, ["crypto"]))

    assert [r.vacancy.id for r in result.ranking.results] == ["one", "none"]
    assert [r.vacancy.id for r in result.ranking.hidden] == ["both"]
    assert result.ranking.hidden[0].adjustment_reasons == [RISKY_NICHE]
    assert result.ranking.hidden[0].fit is not None


def test_the_response_counts_each_reason_and_each_vacancy_once() -> None:
    vacancies = tag("both", mode="on_site", country="DE")
    vacancies = [v.model_copy(update={"work_mode": "on_site", "country": "DE"})
                 if v.id == "none" else v for v in vacancies]  # abroad only
    adjustments = chosen(vacancies, "UA", ["crypto"])

    response = build_response(run(vacancies=vacancies, adjustments=adjustments))

    assert response.hidden_count == 2
    assert response.hidden_reasons == {ON_SITE_ABROAD: 2, RISKY_NICHE: 1}
    assert response.hidden_unexplained == 0
    assert [m.vacancy.id for m in response.matches] == ["one"]


def test_hidden_vacancies_with_an_unknown_reason_are_counted_as_unexplained() -> None:
    from jobpdf.api.response import _hidden_unexplained

    vacancies = tag("both", mode="on_site", country="DE")
    known = chosen(vacancies, "UA", ["crypto"])
    adjustments = {**known, "one": VacancyAdjustment(hidden=True, reasons=["future_rule"]),
                   "none": VacancyAdjustment(hidden=True)}
    hidden = run(vacancies=vacancies, adjustments=adjustments).ranking.hidden

    assert len(hidden) == 3
    assert _hidden_unexplained(hidden) == 2
    assert build_response(run(vacancies=vacancies, adjustments=adjustments)
                          ).hidden_unexplained == 2


def test_a_hidden_vacancy_with_one_known_and_one_unknown_reason_is_explained() -> None:
    from jobpdf.api.response import _hidden_unexplained

    vacancies = tag("both")
    hidden = run(vacancies=vacancies,
                 adjustments=chosen(vacancies, None, ["crypto"])).ranking.hidden
    mixed = [hidden[0].model_copy(update={"adjustment_reasons": ["future_rule", RISKY_NICHE]})]

    assert _hidden_unexplained(mixed) == 0


def test_a_response_without_hidden_vacancies_has_nothing_unexplained() -> None:
    assert build_response(run()).hidden_unexplained == 0
