"""JM-51: the upload form's "hide jobs in these industries" choice, end to end. The real pipeline
runs on a fixture CV with a scripted model, so no test needs a network, a key or a database."""

import html as htmllib
import re

import pytest
from fastapi.testclient import TestClient

from jobpdf import niches
from jobpdf.api.app import create_app
from jobpdf.api.response import AnalysisResponse, build_response
from jobpdf.ranking.eligibility import ON_SITE_ABROAD, RISKY_NICHE
from jobpdf.web import routes
from jobpdf.web.view import (
    EVERYTHING_HIDDEN,
    EVERYTHING_HIDDEN_BOTH,
    EVERYTHING_IN_NICHES,
    NO_JOBS,
    build_report,
)
from tests.api.test_app import HEADERS, make_services
from tests.pipeline.fakes import VACANCIES, cv_bytes, run
from tests.web.test_abroad import analyze
from tests.web.test_progress import parse

STREAM = {"Accept": "text/event-stream"}


def tagged(**per_id):
    """The fixture vacancies with industries and, optionally, a place: ``one=(["crypto"], )``."""
    out = []
    for v in VACANCIES:
        if v.id in per_id:
            codes, *place = per_id[v.id]
            update = {"niches": list(codes)}
            if place:
                update.update(work_mode=place[0][0], country=place[0][1])
            v = v.model_copy(update=update)
        out.append(v)
    return out


def page(vacancies, headers=None, **form):
    services = make_services(analyze=analyze, vacancies=lambda: vacancies)
    with TestClient(create_app(services), raise_server_exceptions=False) as client:
        response = client.post("/report", files={"file": ("cv.pdf", cv_bytes())}, data=form,
                               headers=headers or {})
    assert response.status_code == 200, response.text
    return response.text


def text_of(html: str) -> str:
    return htmllib.unescape(html)


BETS = tagged(both=(["gambling"],), one=(["crypto"],), none=([],))
TICKED = {"avoid_gambling": "1"}


# --- the upload page ---------------------------------------------------------------------------


def test_the_form_has_one_unticked_checkbox_per_industry_in_the_tables_order() -> None:
    with TestClient(create_app(make_services())) as client:
        html = client.get("/").text

    form = html.split('<form id="upload"', 1)[1].split("</form>", 1)[0]
    group = form.split('<fieldset class="field"', 1)[1].split("</fieldset>", 1)[0]
    boxes = re.findall(r'<input type="checkbox" name="([a-z_]+)" value="1">', group)
    assert boxes == [niches.field_name(code) for code in niches.CODES]
    assert "checked" not in group
    for niche in niches.NICHES:
        assert f'name="avoid_{niche.code}" value="1"> {niche.label}</label>' in group
    assert "<legend>Hide jobs in these industries</legend>" in group


def test_the_group_is_explained_to_screen_readers_and_nothing_is_preticked_by_language() -> None:
    with TestClient(create_app(make_services())) as client:
        html = client.get("/", headers={"Accept-Language": "uk-UA,uk;q=0.9"}).text

    assert 'aria-describedby="niches-hint"' in html and 'id="niches-hint"' in html
    assert "if it is unclear, the job stays" in " ".join(html.split())
    assert " checked" not in html


def test_the_form_fields_are_the_ones_the_route_reads() -> None:
    assert routes.NICHE_FIELDS == tuple(f"avoid_{code}" for code in niches.CODES)
    assert set(routes.NICHE_FIELDS) <= set(routes.FORM_FIELDS)
    assert {"lang", "home_country"} <= set(routes.FORM_FIELDS)


# --- the report --------------------------------------------------------------------------------


def test_without_a_choice_every_job_is_shown_and_nothing_is_said_about_hiding() -> None:
    html = page(BETS)

    for title in ("Role both", "Role one", "Role none"):
        assert title in html
    assert "left out" not in html


def test_ticking_an_industry_hides_the_jobs_in_it_and_says_so() -> None:
    html = page(BETS, **TICKED)

    assert "Role both" not in html and "Role one" in html and "Role none" in html
    assert "1 posting in gambling left out, as you asked." in html


def test_several_industries_are_named_in_the_note() -> None:
    two = text_of(page(BETS, avoid_gambling="1", avoid_crypto="1"))
    three = text_of(page(tagged(both=(["gambling"],), one=(["crypto"],), none=(["forex"],)),
                         avoid_gambling="1", avoid_crypto="1", avoid_forex="1"))

    assert "2 postings in gambling or crypto left out, as you asked." in two
    assert "Role none" in two
    assert "3 postings in gambling, crypto or forex left out, as you asked." in three


def test_a_job_with_no_industry_known_is_never_hidden() -> None:
    html = page(tagged(both=([],), one=([],), none=([],)), avoid_gambling="1", avoid_crypto="1")

    for title in ("Role both", "Role one", "Role none"):
        assert title in html
    assert "left out" not in html


def test_an_industry_that_was_not_ticked_hides_nothing() -> None:
    assert "Role one" in page(BETS, **TICKED)
    assert "Role both" in page(BETS, avoid_crypto="1")


@pytest.mark.parametrize("value", ["", "on", "true", "0", "yes", "11", " 1", "<script>"])
def test_only_the_value_the_form_sends_ticks_an_industry(value) -> None:
    html = page(BETS, avoid_gambling=value)

    assert "Role both" in html and "left out" not in html


def test_a_field_for_an_industry_that_does_not_exist_is_ignored() -> None:
    html = page(BETS, avoid_nope="1", avoid_="1", AVOID_GAMBLING="1")

    assert "Role both" in html and "left out" not in html


def test_both_choices_work_together_and_a_job_hidden_twice_counts_in_each_note() -> None:
    vacancies = tagged(both=(["gambling"], ("on_site", "DE")),
                       one=([], ("on_site", "DE")),
                       none=(["gambling"], ("remote", None)))

    html = text_of(page(vacancies, home_country="UA", **TICKED))

    assert "Role both" not in html and "Role one" not in html and "Role none" not in html
    assert "2 on-site or hybrid postings outside Ukraine left out, as you asked." in html
    assert "2 postings in gambling left out, as you asked." in html
    assert "could not apply" not in html  # every hidden job has a reason the page knows


def test_the_two_choices_leave_in_what_neither_asks_to_hide() -> None:
    vacancies = tagged(both=(["gambling"], ("remote", None)), one=([], ("on_site", "DE")),
                       none=([], ("remote", "DE")))

    html = page(vacancies, home_country="UA", **TICKED)

    assert "Role both" not in html and "Role one" not in html and "Role none" in html


def test_when_everything_is_left_out_for_an_industry_the_page_says_so() -> None:
    html = text_of(page(tagged(both=(["crypto"],), one=(["crypto"],), none=(["gambling"],)),
                        avoid_crypto="1", avoid_gambling="1"))

    assert EVERYTHING_IN_NICHES in html and NO_JOBS not in html
    assert "3 postings in gambling or crypto left out" in html


def test_when_everything_is_left_out_for_both_reasons_the_page_says_so() -> None:
    vacancies = tagged(both=(["gambling"], ("remote", None)), one=([], ("on_site", "DE")),
                       none=([], ("on_site", "DE")))

    html = text_of(page(vacancies, home_country="UA", **TICKED))

    assert EVERYTHING_HIDDEN_BOTH in html and NO_JOBS not in html


def test_when_everything_is_left_out_for_the_country_alone_the_old_sentence_stays() -> None:
    vacancies = tagged(both=([], ("on_site", "DE")), one=([], ("on_site", "DE")),
                       none=([], ("on_site", "DE")))

    assert EVERYTHING_HIDDEN in text_of(page(vacancies, home_country="UA"))


def test_the_stream_gives_the_same_page_as_a_plain_request() -> None:
    plain = page(BETS, avoid_gambling="1", avoid_crypto="1")
    services = make_services(analyze=analyze, vacancies=lambda: BETS)
    with TestClient(create_app(services)) as client:
        response = client.post("/report", files={"file": ("cv.pdf", cv_bytes())},
                               data={"avoid_gambling": "1", "avoid_crypto": "1"}, headers=STREAM)

    events = parse(response.text)
    assert events[-1][0] == "done" and events[-1][1]["html"] == plain


def test_the_language_and_the_industry_choices_work_together() -> None:
    from tests.pipeline.fakes import PYTHON

    services = make_services(analyze=analyze, vacancies=lambda: BETS,
                             skill_names=lambda: {PYTHON: "Пайтон"})
    with TestClient(create_app(services)) as client:
        html = client.post("/report", files={"file": ("cv.pdf", cv_bytes())},
                           data={"lang": "uk", **TICKED}).text

    assert "Пайтон" in html and "Role both" not in html and "in gambling" in html


def test_choosing_industries_is_counted_in_the_metrics_like_any_analysis() -> None:
    services = make_services(analyze=analyze, vacancies=lambda: BETS)
    with TestClient(create_app(services)) as client:
        client.post("/report", files={"file": ("cv.pdf", cv_bytes())}, data=TICKED)

    assert services.recorder.snapshot().analyses.total == 1


def test_the_slot_is_given_back_after_a_report_with_the_choice() -> None:
    services = make_services(analyze=analyze, vacancies=lambda: BETS, max_concurrent=1)
    with TestClient(create_app(services)) as client:
        for _ in range(2):
            response = client.post("/report", files={"file": ("cv.pdf", cv_bytes())}, data=TICKED)
            assert response.status_code == 200


@pytest.mark.parametrize("form", [{}, {"avoid_gambling": ""}, {"avoid_gambling": "on"},
                                  {"avoid_nope": "1"}, {"home_country": "ZZ"}])
def test_an_analyze_function_that_does_not_know_adjustments_is_not_given_them(form) -> None:
    seen = []

    def strict(content, filename, vacancies, *, caller, normalizer, embedder, scorer):
        seen.append(1)  # a TypeError if ``adjustments`` were passed without a real choice
        return analyze(content, filename, vacancies)

    services = make_services(analyze=strict, vacancies=lambda: BETS)
    with TestClient(create_app(services)) as client:
        response = client.post("/report", files={"file": ("cv.pdf", cv_bytes())}, data=form)

    assert response.status_code == 200 and seen == [1]


def test_a_failing_vacancy_list_is_a_failed_analysis_with_the_choice_too() -> None:
    def broken():
        raise RuntimeError("database down")

    services = make_services(analyze=analyze, vacancies=broken)
    with TestClient(create_app(services), raise_server_exceptions=False) as client:
        response = client.post("/report", files={"file": ("cv.pdf", cv_bytes())}, data=TICKED)

    assert response.status_code == 500
    assert services.recorder.snapshot().analyses.total == 1


def test_the_json_api_ignores_the_form_fields_and_hides_nothing() -> None:
    services = make_services(analyze=analyze, vacancies=lambda: BETS)
    with TestClient(create_app(services)) as client:
        body = client.post("/v1/analyze", files={"file": ("cv.pdf", cv_bytes())},
                           data=TICKED, headers=HEADERS).json()

    assert body["hidden_count"] == 0 and body["hidden_reasons"] == {}
    assert body["hidden_unexplained"] == 0 and len(body["matches"]) == 3


# --- the wording -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def response() -> AnalysisResponse:
    return build_response(run())


def hidden(response, reasons: dict[str, int], count: int, unexplained: int | None = None,
           matches=None) -> AnalysisResponse:
    update = {"hidden_reasons": reasons, "hidden_count": count,
              "hidden_unexplained": count - sum(reasons.values()) if unexplained is None
              else unexplained}
    if matches is not None:
        update["matches"] = matches
    return response.model_copy(update=update)


def test_the_note_counts_singular_and_plural_and_names_what_was_chosen(response) -> None:
    one = build_report(hidden(response, {RISKY_NICHE: 1}, 1), avoid_niches=["crypto"])
    many = build_report(hidden(response, {RISKY_NICHE: 4}, 4), avoid_niches=["crypto"])

    assert "1 posting in crypto left out, as you asked." in one.notes
    assert "4 postings in crypto left out, as you asked." in many.notes


@pytest.mark.parametrize("avoid, names", [
    (["gambling"], "gambling"),
    (["gambling", "crypto"], "gambling or crypto"),
    (["crypto", "gambling"], "gambling or crypto"),  # the table's order, not the caller's
    (["gambling", "crypto", "forex"], "gambling, crypto or forex"),
    (["mlm", "adult"], "adult content or network marketing"),
    (["forex"], "forex"),
    (["gambling", "nope"], "gambling"),
])
def test_the_note_lists_the_industries_in_the_tables_order(response, avoid, names) -> None:
    report = build_report(hidden(response, {RISKY_NICHE: 2}, 2), avoid_niches=avoid)

    assert f"2 postings in {names} left out, as you asked." in report.notes


@pytest.mark.parametrize("avoid", [(), ["nope"], None])
def test_the_note_copes_with_a_page_that_was_not_told_the_industries(response, avoid) -> None:
    report = build_report(hidden(response, {RISKY_NICHE: 2}, 2), avoid_niches=avoid or ())

    assert "2 postings in an industry you chose to avoid left out, as you asked." in report.notes


def test_each_known_reason_has_its_own_note_and_unexplained_ones_keep_the_old_wording(
        response) -> None:
    mixed = hidden(response, {ON_SITE_ABROAD: 2, RISKY_NICHE: 3}, 6, unexplained=1)

    notes = build_report(mixed, home_country="UA", avoid_niches=["gambling"]).notes

    assert notes[-3:] == [
        "2 on-site or hybrid postings outside Ukraine left out, as you asked.",
        "3 postings in gambling left out, as you asked.",
        "1 posting left out because your CV suggests you could not apply for them."]


def test_no_hidden_postings_add_no_note(response) -> None:
    assert not any("left out" in n for n in build_report(response, avoid_niches=["crypto"]).notes)


def test_the_empty_message_depends_on_why_the_list_is_empty(response) -> None:
    niche_only = hidden(response, {RISKY_NICHE: 3}, 3, matches=[])
    abroad_only = hidden(response, {ON_SITE_ABROAD: 3}, 3, matches=[])
    # three jobs, one of them hidden for both reasons, so the counts add up to four
    both = hidden(response, {ON_SITE_ABROAD: 2, RISKY_NICHE: 2}, 3, unexplained=0, matches=[])
    partly_explained = hidden(response, {RISKY_NICHE: 1}, 3, matches=[])
    other_reason = hidden(response, {}, 2, matches=[])
    nothing = hidden(response, {}, 0, matches=[])

    assert build_report(niche_only).empty == EVERYTHING_IN_NICHES
    assert build_report(abroad_only).empty == EVERYTHING_HIDDEN
    assert build_report(both).empty == EVERYTHING_HIDDEN_BOTH
    assert build_report(partly_explained).empty == NO_JOBS
    assert build_report(other_reason).empty == NO_JOBS
    assert build_report(nothing).empty == NO_JOBS
    assert build_report(response).empty is None


def test_the_industry_message_says_how_to_get_the_jobs_back() -> None:
    assert "untick" in EVERYTHING_IN_NICHES and "upload page" in EVERYTHING_IN_NICHES
    assert "change these choices" in EVERYTHING_HIDDEN_BOTH
