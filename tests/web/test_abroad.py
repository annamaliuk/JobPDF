"""JM-49: the web form's "hide on-site jobs outside" choice, end to end. The real pipeline runs on
a fixture CV with a scripted LLM, so no test needs a network, a key or a database."""

import html as htmllib
import re

import pytest
from fastapi.testclient import TestClient

from jobpdf import pipeline
from jobpdf.api.app import Services, create_app
from jobpdf.api.response import AnalysisResponse, build_response
from jobpdf.places import COUNTRY_NAMES
from jobpdf.ranking.eligibility import ON_SITE_ABROAD
from jobpdf.web import routes, view
from jobpdf.web.view import EVERYTHING_HIDDEN, NO_JOBS, build_report
from tests.api.test_app import make_services
from tests.normalization.fakes import FakeEmbedder
from tests.pipeline.fakes import TODAY, VACANCIES, FakeNormalizer, caller, cv_bytes, run
from tests.web.test_progress import parse

STREAM = {"Accept": "text/event-stream"}


def analyze(content, filename, vacancies, **kwargs):
    """The real pipeline, passing on what the production wiring passes."""
    return pipeline.analyze_cv(
        content, filename, vacancies, caller=caller(), normalizer=FakeNormalizer(),
        embedder=FakeEmbedder(), today=TODAY, scorer=kwargs.get("scorer"),
        adjustments=kwargs.get("adjustments"), on_stage=kwargs.get("on_stage"))


def located(**places):
    """The fixture vacancies with a work mode and country each: ``both=("on_site", "DE")``."""
    return [v.model_copy(update=dict(zip(("work_mode", "country"), places[v.id], strict=True)))
            if v.id in places else v for v in VACANCIES]


def services_for(vacancies) -> Services:
    return make_services(analyze=analyze, vacancies=lambda: vacancies)


def page(vacancies, headers=None, **form):
    with TestClient(create_app(services_for(vacancies)), raise_server_exceptions=False) as client:
        response = client.post("/report", files={"file": ("cv.pdf", cv_bytes())}, data=form,
                               headers=headers or {})
    assert response.status_code == 200, response.text
    return response.text


ABROAD = located(both=("on_site", "DE"), one=("remote", "DE"), none=("hybrid", "PL"))


# --- the upload page ---------------------------------------------------------------------------


def test_the_upload_form_has_the_choice_with_no_hiding_first_and_every_country() -> None:
    with TestClient(create_app(make_services())) as client:
        html = client.get("/").text

    select = html.split('<select id="home_country"', 1)[1].split("</select>", 1)[0]
    assert '<option value="" selected>Don\'t hide any jobs</option>' in select
    assert select.count(" selected>") == 1
    assert len(re.findall(r'<option value="[A-Z]{2}">', select)) == len(COUNTRY_NAMES)
    assert '<option value="UA">Ukraine</option>' in select
    names = re.findall(r'<option value="[A-Z]{2}">([^<]+)</option>', select)
    assert names == sorted(names)
    assert 'for="home_country">Hide on-site jobs outside</label>' in html
    assert "Remote jobs stay." in html


def test_the_choice_is_explained_to_screen_readers_and_is_part_of_the_form() -> None:
    with TestClient(create_app(make_services())) as client:
        html = client.get("/").text

    form = html.split('<form id="upload"', 1)[1].split("</form>", 1)[0]
    assert 'aria-describedby="home_country-hint"' in form and 'id="home_country-hint"' in form
    assert 'name="home_country"' in form


def test_nothing_is_preselected_from_the_browsers_language() -> None:
    with TestClient(create_app(make_services())) as client:
        html = client.get("/", headers={"Accept-Language": "uk-UA,uk;q=0.9"}).text

    select = html.split('<select id="home_country"', 1)[1].split("</select>", 1)[0]
    assert select.count(" selected>") == 1 and 'value="" selected' in select


# --- the report --------------------------------------------------------------------------------


def test_without_the_choice_every_job_is_shown_and_nothing_is_said_about_hiding() -> None:
    html = page(ABROAD)

    for title in ("Role both", "Role one", "Role none"):
        assert title in html
    assert "left out" not in html


def test_choosing_a_country_hides_the_on_site_and_hybrid_jobs_abroad_and_says_so() -> None:
    html = page(ABROAD, home_country="UA")

    assert "Role one" in html and "Role both" not in html and "Role none" not in html
    assert "2 on-site or hybrid postings outside Ukraine left out, as you asked." in html


def test_jobs_in_the_chosen_country_stay() -> None:
    html = page(ABROAD, home_country="DE")

    assert "Role both" in html and "Role one" in html  # on site in Germany is not abroad
    assert "Role none" not in html
    assert "1 on-site or hybrid posting outside Germany left out, as you asked." in html


def test_a_job_with_nothing_known_about_it_is_never_hidden() -> None:
    html = page(located(both=(None, None), one=("on_site", None), none=(None, "DE")),
                home_country="UA")

    for title in ("Role both", "Role one", "Role none"):
        assert title in html
    assert "left out" not in html


@pytest.mark.parametrize("value", ["", "ZZ", "Ukraine", "<script>", "UAE", " "])
def test_a_choice_that_is_not_a_country_code_hides_nothing(value) -> None:
    html = page(ABROAD, home_country=value)

    assert "Role both" in html and "Role none" in html and "left out" not in html


@pytest.mark.parametrize("value", ["ua", " UA ", "Ua"])
def test_the_country_code_is_read_without_regard_to_case_or_spaces(value) -> None:
    assert "Role both" not in page(ABROAD, home_country=value)


def test_when_everything_is_left_out_the_page_says_why_and_how_to_see_the_jobs() -> None:
    html = page(located(both=("on_site", "DE"), one=("on_site", "DE"), none=("on_site", "DE")),
                home_country="UA")

    text = htmllib.unescape(html)
    assert EVERYTHING_HIDDEN in text and NO_JOBS not in text
    assert "3 on-site or hybrid postings outside Ukraine left out" in html


def test_a_report_with_no_vacancies_at_all_still_says_there_are_none() -> None:
    assert NO_JOBS in page([], home_country="UA")


def test_the_stream_gives_the_same_page_as_a_plain_request() -> None:
    plain = page(ABROAD, home_country="UA")
    with TestClient(create_app(services_for(ABROAD))) as client:
        response = client.post("/report", files={"file": ("cv.pdf", cv_bytes())},
                               data={"home_country": "UA"}, headers=STREAM)

    events = parse(response.text)
    assert events[-1][0] == "done" and events[-1][1]["html"] == plain


def test_the_language_and_the_country_choices_work_together() -> None:
    from tests.pipeline.fakes import PYTHON

    services = make_services(analyze=analyze, vacancies=lambda: ABROAD,
                             skill_names=lambda: {PYTHON: "Пайтон"})
    with TestClient(create_app(services)) as client:
        html = client.post("/report", files={"file": ("cv.pdf", cv_bytes())},
                           data={"lang": "uk", "home_country": "UA"}).text

    assert "Пайтон" in html and "Role both" not in html and "outside Ukraine" in html


def test_choosing_a_country_is_counted_in_the_metrics_like_any_analysis() -> None:
    services = services_for(ABROAD)
    with TestClient(create_app(services)) as client:
        client.post("/report", files={"file": ("cv.pdf", cv_bytes())}, data={"home_country": "UA"})

    assert services.recorder.snapshot().analyses.total == 1


def test_the_slot_is_given_back_after_a_report_with_the_choice() -> None:
    services = make_services(analyze=analyze, vacancies=lambda: ABROAD, max_concurrent=1)
    with TestClient(create_app(services)) as client:
        for _ in range(2):
            response = client.post("/report", files={"file": ("cv.pdf", cv_bytes())},
                                   data={"home_country": "UA"})
            assert response.status_code == 200


def test_an_analyze_function_that_does_not_know_adjustments_is_not_given_them() -> None:
    seen = []

    def strict(content, filename, vacancies, *, caller, normalizer, embedder, scorer):
        seen.append(1)  # a TypeError if ``adjustments`` were passed without a choice
        return analyze(content, filename, vacancies)

    services = make_services(analyze=strict, vacancies=lambda: ABROAD)
    with TestClient(create_app(services)) as client:
        assert client.post("/report", files={"file": ("cv.pdf", cv_bytes())}).status_code == 200

    assert seen == [1]


@pytest.mark.parametrize("value", ["", "ZZ", "Ukraine", " "])
def test_a_choice_that_is_not_a_country_is_not_passed_on_at_all(value) -> None:
    seen = []

    def strict(content, filename, vacancies, *, caller, normalizer, embedder, scorer):
        seen.append(1)  # a TypeError if ``adjustments`` were passed for a junk choice
        return analyze(content, filename, vacancies)

    services = make_services(analyze=strict, vacancies=lambda: ABROAD)
    with TestClient(create_app(services)) as client:
        response = client.post("/report", files={"file": ("cv.pdf", cv_bytes())},
                               data={"home_country": value})

    assert response.status_code == 200 and seen == [1]


def test_a_vacancy_list_that_fails_is_counted_as_a_failed_analysis() -> None:
    def broken():
        raise RuntimeError("database down")

    services = make_services(analyze=analyze, vacancies=broken)
    with TestClient(create_app(services), raise_server_exceptions=False) as client:
        response = client.post("/report", files={"file": ("cv.pdf", cv_bytes())},
                               data={"home_country": "UA"})

    assert response.status_code == 500
    assert services.recorder.snapshot().analyses.total == 1


def test_the_json_api_is_unaffected_and_says_nothing_about_countries() -> None:
    from tests.api.test_app import HEADERS

    services = services_for(ABROAD)
    with TestClient(create_app(services)) as client:
        body = client.post("/v1/analyze", files={"file": ("cv.pdf", cv_bytes())},
                           data={"home_country": "UA"}, headers=HEADERS).json()

    assert body["hidden_count"] == 0 and body["hidden_reasons"] == {}
    assert len(body["matches"]) == 3


# --- the wording -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def response() -> AnalysisResponse:
    return build_response(run())


def with_hidden(response, reasons: dict[str, int], count: int, matches=None) -> AnalysisResponse:
    update = {"hidden_reasons": reasons, "hidden_count": count}
    if matches is not None:
        update["matches"] = matches
    return response.model_copy(update=update)


def test_the_note_names_the_country_and_counts_singular_and_plural(response) -> None:
    one = build_report(with_hidden(response, {ON_SITE_ABROAD: 1}, 1), home_country="PL")
    many = build_report(with_hidden(response, {ON_SITE_ABROAD: 4}, 4), home_country="PL")

    assert "1 on-site or hybrid posting outside Poland left out, as you asked." in one.notes
    assert "4 on-site or hybrid postings outside Poland left out, as you asked." in many.notes


def test_the_note_copes_with_a_country_it_does_not_know(response) -> None:
    for home in (None, "ZZ", ""):
        notes = build_report(with_hidden(response, {ON_SITE_ABROAD: 2}, 2), home_country=home).notes

        assert "2 on-site or hybrid postings in another country left out, as you asked." in notes


def test_a_reason_the_page_does_not_know_keeps_the_old_wording_for_its_share(response) -> None:
    report = build_report(with_hidden(response, {ON_SITE_ABROAD: 2}, 5), home_country="UA")

    assert "2 on-site or hybrid postings outside Ukraine left out, as you asked." in report.notes
    assert ("3 postings left out because your CV suggests you could not apply for them."
            in report.notes)


def test_hidden_postings_without_any_reason_use_the_old_sentence_for_all(response) -> None:
    report = build_report(with_hidden(response, {}, 2))

    assert report.notes[-1] == ("2 postings left out because your CV suggests you could not "
                                "apply for them.")


def test_no_hidden_postings_add_no_note(response) -> None:
    assert not any("left out" in note for note in build_report(response).notes)


def test_the_empty_message_depends_on_why_the_list_is_empty(response) -> None:
    hidden_all = with_hidden(response, {ON_SITE_ABROAD: 3}, 3, matches=[])
    mixed = with_hidden(response, {ON_SITE_ABROAD: 1}, 3, matches=[])
    none_at_all = with_hidden(response, {}, 0, matches=[])
    other_reason = with_hidden(response, {}, 2, matches=[])

    assert build_report(hidden_all).empty == EVERYTHING_HIDDEN
    assert build_report(mixed).empty == NO_JOBS
    assert build_report(none_at_all).empty == NO_JOBS
    assert build_report(other_reason).empty == NO_JOBS
    assert build_report(response).empty is None


def test_the_form_field_and_the_view_agree_on_the_reason_code() -> None:
    assert view.ON_SITE_ABROAD == ON_SITE_ABROAD == "on_site_abroad"
    assert routes.HOME_COUNTRY_FIELD == "home_country"
