"""JM-53: the report's "Your skills" form. Skills added or removed are re-ranked from the state the
page carries, without the CV and without a model call. The real pipeline makes the first page on
a fixture CV with a scripted LLM, so nothing here needs a network, a key or a database."""

import base64
import html as htmllib
import json
import logging
import re

import pytest
from fastapi.testclient import TestClient

from jobpdf.api.access import BODY_OVERHEAD_BYTES
from jobpdf.api.app import create_app
from jobpdf.api.response import build_response
from jobpdf.ranking.semantic import embed_vacancy
from jobpdf.web import routes, update
from jobpdf.web.update import (
    MAX_ADDED_SKILLS,
    MAX_SKILL_CHARS,
    MAX_STATE_CHARS,
    REPORT_EXPIRED,
    Typed,
    UpdateError,
    decode_state,
    encode_state,
    state_from_result,
    typed_skills,
    update_report,
)
from tests.api.test_app import make_services
from tests.normalization.fakes import FakeEmbedder
from tests.pipeline.fakes import SECRETS, VACANCIES, FakeNormalizer, cv_bytes, run
from tests.web.test_abroad import analyze, located

# The fixture vacancies, prepared for the past-work comparison so the vectors in the state matter.
EMBEDDED = [embed_vacancy(v.model_copy(update={"responsibilities": [f"Build {v.id} pipelines"]}),
                          FakeEmbedder()) for v in VACANCIES]


def ranked(response) -> list[tuple]:
    return [(m.vacancy.id, m.band, m.fit, m.components.semantic) for m in response.matches]


def first_state(vacancies=VACANCIES, home=None, language="en"):
    result = run(vacancies=vacancies)
    return result, state_from_result(result, home, language)


# --- the state ---------------------------------------------------------------------------------


def test_the_state_survives_the_page_and_holds_no_cv_text() -> None:
    _, state = first_state(EMBEDDED)
    text = encode_state(state)

    assert decode_state(text) == state
    assert re.fullmatch(r"[A-Za-z0-9_=-]+", text)  # safe inside an HTML attribute as it is
    decoded = base64.urlsafe_b64decode(text).decode()
    assert not any(secret in decoded for secret in SECRETS)  # no employer, name or file name
    assert all(not m.candidates for m in state.skills.values())  # trimmed, ranking never reads them
    assert len(text) < MAX_STATE_CHARS


def test_the_state_keeps_the_numbers_ranking_needs() -> None:
    result, state = first_state(EMBEDDED, home="UA", language="uk")

    assert state.vectors == result.vectors and state.vectors is not None
    assert state.experience_months == result.experience_months == 48
    assert (state.home_country, state.language) == ("UA", "uk")
    assert all(w.stage != "ranking" for w in state.warnings)  # ranking adds its own again


def encoded(state_dict: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(state_dict).encode()).decode()


@pytest.mark.parametrize("make", [
    lambda s: "",
    lambda s: "A" * (MAX_STATE_CHARS + 1),
    lambda s: "not base64 at all!",
    lambda s: "ПРИВІТ",
    lambda s: base64.urlsafe_b64encode(b"{not json").decode(),
    lambda s: encoded({**s, "extra": 1}),
    lambda s: encoded({**s, "version": 99}),
    lambda s: encoded({**s, "versions": {**s["versions"], "ranking": "1"}}),
    lambda s: encoded({k: v for k, v in s.items() if k != "skills"}),
], ids=["empty", "too_big", "not_base64", "not_ascii", "not_json", "unknown_field",
        "other_version", "older_ranking", "missing_field"])
def test_a_state_this_page_did_not_make_is_refused(make) -> None:
    _, state = first_state()

    with pytest.raises(UpdateError) as caught:
        decode_state(make(state.model_dump(mode="json")))

    assert caught.value.code == REPORT_EXPIRED


# --- what the person typed ---------------------------------------------------------------------


def test_typed_skills_are_split_trimmed_and_listed_once() -> None:
    assert typed_skills(" Airflow, dbt ;Spark\n\n, dbt ", ["Docker", " Spark "]) == Typed(
        ["Airflow", "dbt", "Spark", "Docker"])
    assert typed_skills("", []) == Typed([]) and typed_skills(" , ;") == Typed([])
    assert typed_skills("CI/CD, Apache Airflow").names == ["CI/CD", "Apache Airflow"]


def test_skills_over_the_limits_are_left_out_and_counted() -> None:
    many = ", ".join(f"s{i}" for i in range(MAX_ADDED_SKILLS + 3))

    assert typed_skills(many) == Typed([f"s{i}" for i in range(MAX_ADDED_SKILLS)], 0, 3)
    assert typed_skills("dbt, " + "x" * (MAX_SKILL_CHARS + 1)) == Typed(["dbt"], 1, 0)
    assert typed_skills("x" * MAX_SKILL_CHARS).names == ["x" * MAX_SKILL_CHARS]


# --- re-ranking --------------------------------------------------------------------------------


@pytest.mark.parametrize("vacancies", [VACANCIES, EMBEDDED], ids=["skills_only", "embedded"])
def test_an_update_with_no_changes_gives_exactly_the_first_ranking(vacancies) -> None:
    result, state = first_state(vacancies)

    new_state, response = update_report(state, [], [], normalizer=FakeNormalizer(),
                                        vacancies=vacancies)

    assert ranked(response) == ranked(build_response(result))
    assert response.insights == build_response(result).insights
    assert new_state == state


def test_the_past_work_vectors_in_the_state_are_what_the_update_compares() -> None:
    result, state = first_state(EMBEDDED)

    _, response = update_report(state, [], [], normalizer=FakeNormalizer(), vacancies=EMBEDDED)

    first = [m.components.semantic_raw for m in build_response(result).matches]
    assert [m.components.semantic_raw for m in response.matches] == first
    assert None not in first


def test_adding_a_missing_skill_moves_the_job_up_and_it_is_no_longer_missing() -> None:
    result, state = first_state()
    before = {m.vacancy.id: m for m in build_response(result).matches}

    new_state, response = update_report(state, ["Rust"], [], normalizer=FakeNormalizer(),
                                        vacancies=VACANCIES)
    after = {m.vacancy.id: m for m in response.matches}

    assert [s.name for s in before["one"].required.missing] == ["rust"]
    assert after["one"].required.missing == [] and after["one"].fit > before["one"].fit
    assert (before["one"].band.value, after["one"].band.value) == ("partial", "strong")
    assert [(s.name, s.added) for s in response.profile.skills if s.added] == [("Rust", True)]
    assert list(new_state.skills)[-1] == "added[0]"


def test_typed_skills_are_matched_without_the_language_model() -> None:
    _, state = first_state()
    normalizer = FakeNormalizer()

    update_report(state, ["Rust", "Go"], [], normalizer=normalizer, vacancies=VACANCIES)

    assert normalizer.typed == [(["Rust", "Go"], False)]


def test_a_skill_already_in_the_cv_or_typed_again_is_not_added_twice() -> None:
    _, state = first_state()

    once, _ = update_report(state, ["Rust", "python"], [], normalizer=FakeNormalizer(),
                            vacancies=VACANCIES)
    twice, _ = update_report(once, ["rust", "Go"], [], normalizer=FakeNormalizer(),
                             vacancies=VACANCIES)

    assert [m.raw for p, m in twice.skills.items() if p.startswith("added")] == ["Rust", "Go"]
    assert list(twice.skills)[-2:] == ["added[0]", "added[1]"]


def test_removing_a_skill_takes_it_out_of_the_ranking_and_can_take_back_an_added_one() -> None:
    result, state = first_state()
    added, _ = update_report(state, ["Rust"], [], normalizer=FakeNormalizer(), vacancies=VACANCIES)

    back, response = update_report(added, [], ["raw:rust"], normalizer=FakeNormalizer(),
                                   vacancies=VACANCIES)
    assert ranked(response) == ranked(build_response(result)) and back == state

    sql = next(m.concept_id for m in state.skills.values() if m.raw == "SQL")
    _, without_sql = update_report(state, [], [sql], normalizer=FakeNormalizer(),
                                   vacancies=VACANCIES)
    both = next(m for m in without_sql.matches if m.vacancy.id == "both")
    assert [s.name for s in both.required.missing] == ["sql"]


def test_the_country_choice_and_the_experience_cap_still_apply() -> None:
    abroad = located(both=("on_site", "DE"))
    _, state = first_state(vacancies=abroad, home="UA")
    asks_more = [v.model_copy(update={"min_years_experience": 7}) if v.id == "one" else v
                 for v in abroad]  # 84 months asked, 48 in the CV: 3 years short

    _, response = update_report(state, ["Rust"], [], normalizer=FakeNormalizer(),
                                vacancies=asks_more)

    assert "both" not in {m.vacancy.id for m in response.matches} and response.hidden_count == 1
    one = next(m for m in response.matches if m.vacancy.id == "one")
    assert (one.band.value, one.capped_from.value) == ("partial", "strong")


def test_the_update_logs_counts_and_never_a_skill(caplog) -> None:
    _, state = first_state()

    with caplog.at_level(logging.DEBUG):
        update_report(state, ["Zeppelinish"], ["raw:whatever"], normalizer=FakeNormalizer(),
                      vacancies=VACANCIES)

    assert "report updated" in caplog.text
    assert "Zeppelinish" not in caplog.text and "whatever" not in caplog.text


# --- the page ----------------------------------------------------------------------------------


def services(**changes):
    base = {"analyze": analyze, "vacancies": lambda: VACANCIES, "normalizer": FakeNormalizer()}
    return make_services(**{**base, **changes})


@pytest.fixture
def client():
    with TestClient(create_app(services()), raise_server_exceptions=False) as c:
        yield c


def report_page(client, **form) -> str:
    response = client.post("/report", files={"file": ("cv.pdf", cv_bytes())}, data=form)
    assert response.status_code == 200, response.text
    return response.text


def state_of(html: str) -> str:
    return re.search(r'name="state" value="([^"]*)"', html).group(1)


def post_update(client, html: str, **form):
    return client.post(routes.UPDATE_PATH, data={"state": state_of(html), **form})


def table(html: str) -> list[str]:
    """The "All matches" rows as plain text."""
    rows = html.split('id="all-heading"')[1].split("</table>")[0].split("<tr>")[2:]
    return [re.sub(r"\s+", " ", htmllib.unescape(re.sub(r"<[^>]+>", " ", r))).strip() for r in rows]


def test_the_skills_and_the_form_come_before_the_best_matches(client) -> None:
    html = report_page(client)

    assert html.index('id="skills-heading"') < html.index('id="best-heading"')
    assert f'<form id="update" method="post" action="{routes.UPDATE_PATH}">' in html
    assert '<label for="add-skills">Add skills we missed</label>' in html
    assert 'name="skills" type="text"' in html and html.count('id="skills-heading"') == 1
    assert "Matches updated" not in html


def test_the_form_posts_back_without_any_script_and_gives_the_same_matches(client) -> None:
    first = report_page(client)

    again = post_update(client, first)

    assert again.status_code == 200
    assert table(again.text) == table(first)
    assert "Matches updated with your changes" in again.text
    for name, value in routes.PAGE_HEADERS.items():
        assert again.headers[name] == value  # the same locked-down page, CSP unchanged


def test_an_added_skill_shows_as_added_by_you_and_changes_the_matches(client) -> None:
    first = report_page(client)

    page = post_update(client, first, skills="Rust").text

    assert "<h3>Added by you</h3>" in page
    assert '<input type="checkbox" name="remove" value="raw:rust">' in page
    assert "Rust</span> <span class=\"quiet\">(not in our skill list)</span>" in page
    assert "Missing: rust." not in page and "Missing: rust." in first
    assert table(page) != table(first)
    # and it can be taken back on the next update
    assert table(post_update(client, page, remove="raw:rust").text) == table(first)


def test_within_reach_has_an_i_have_this_tick_that_adds_the_skill(client) -> None:
    first = report_page(client)

    assert 'form="update" name="have" value="rust"> I have <span>rust</span>' in first
    ticked = post_update(client, first, have="rust").text
    assert "<h3>Added by you</h3>" in ticked and "Missing: rust." not in ticked


def test_the_language_and_country_of_the_first_page_are_kept(client) -> None:
    abroad = located(both=("on_site", "DE"))
    with TestClient(create_app(services(vacancies=lambda: abroad))) as c:
        first = report_page(c, home_country="UA", lang="uk")
        page = post_update(c, first).text

    assert "Role both" not in page and "outside Ukraine left out" in page
    assert decode_state(state_of(page)).language == "uk"


@pytest.mark.parametrize("state", ["broken", ""])
def test_a_bad_state_is_an_error_page_that_repeats_nothing(client, state) -> None:
    response = client.post(routes.UPDATE_PATH, data={"state": state})

    assert response.status_code == 400
    assert htmllib.escape(routes.WEB_DETAILS[REPORT_EXPIRED], quote=False) in response.text
    assert "broken" not in response.text


def test_skills_over_the_limits_keep_the_report_and_say_what_was_left_out(client) -> None:
    first = report_page(client)
    typed = ", ".join(["Rust", "y" * 61, "z" * 70] + [f"s{i}" for i in range(MAX_ADDED_SKILLS)])

    page = post_update(client, first, skills=typed).text

    assert "Only the first 20 skills were added; 1 more was left out. " \
           "2 skills longer than 60 characters were skipped." in page
    assert "<h3>Added by you</h3>" in page and "Missing: rust." not in page
    assert "y" * 61 not in page and "s19" not in page  # Rust and s0 to s18 were the first 20
    assert "left out" not in post_update(client, first, skills="Rust").text


def test_an_update_too_big_to_be_a_report_is_refused_before_it_is_read(client) -> None:
    size = routes.MAX_UPDATE_BYTES + BODY_OVERHEAD_BYTES + 1
    response = client.post(routes.UPDATE_PATH, data={"state": "A" * size})

    assert response.status_code == 413


def test_when_the_server_is_busy_an_update_waits_like_an_upload() -> None:
    svc = services(max_concurrent=1)
    with TestClient(create_app(svc)) as c:
        first = report_page(c)
        svc.slots.acquire()
        try:
            busy = post_update(c, first)
        finally:
            svc.slots.release()
        assert busy.status_code == 503 and busy.headers["Retry-After"]
        assert post_update(c, first).status_code == 200  # the slot was given back


def test_an_update_that_fails_inside_is_a_generic_page(monkeypatch, client, caplog) -> None:
    first = report_page(client)

    def broken(*args, **kwargs):
        raise RuntimeError("Rust " + SECRETS[0])

    monkeypatch.setattr(update, "update_report", broken)
    with caplog.at_level(logging.DEBUG):
        response = post_update(client, first, skills="Rust")

    assert response.status_code == 500
    assert SECRETS[0] not in response.text and SECRETS[0] not in caplog.text
