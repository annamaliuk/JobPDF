"""JM-49: hiding on-site vacancies abroad. A vacancy is hidden only on clear evidence."""

import pytest
from pydantic import ValidationError

from jobpdf.api.response import build_response
from jobpdf.ranking.eligibility import (
    NOT_REMOTE_MODES,
    ON_SITE_ABROAD,
    abroad_adjustments,
    is_on_site_abroad,
    normalize_country,
)
from jobpdf.ranking.models import VacancyFeatures
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
