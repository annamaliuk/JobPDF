"""JM-46: skill names in Ukrainian on the web report. The real pipeline runs on a fixture CV with
a scripted LLM; the names come from a plain dictionary, so no test needs a database or a network."""

import json
import logging

import pytest
from fastapi.testclient import TestClient
from jinja2 import UndefinedError

from jobpdf.api.app import Services, create_app
from jobpdf.api.response import (
    Insights,
    SkillOut,
    SkillToLearn,
    WithinReach,
    build_response,
)
from jobpdf.web import progress, routes
from jobpdf.web.view import Line, build_report
from tests.api.test_app import HEADERS, make_services
from tests.pipeline.fakes import PYTHON, SECRETS, SQL, cv_bytes, run
from tests.web.test_progress import analyze_with_stages, parse
from tests.web.test_routes import XSS, post
from tests.web.test_view import group, one_match, report_of, skill

PYTHON_UK = "Пайтон (мова програмування)"
NAMES = {PYTHON: PYTHON_UK, SQL: "SQL", "c:k8s": "Kubernetes (оркестрація)", "c:js": "ДжейЕс"}


def uk_skill(name: str, label: str, concept: str) -> SkillOut:
    return skill(name, label, concept=concept)


@pytest.fixture(scope="module")
def response():
    return build_response(run())


def services_with(names=NAMES, **changes) -> Services:
    calls: list[int] = []

    def provide():
        calls.append(1)
        return names

    services = make_services(skill_names=provide, **changes)
    services.calls = calls  # type: ignore[attr-defined]
    return services


def page(services: Services, **form) -> str:
    with TestClient(create_app(services), raise_server_exceptions=False) as client:
        response = client.post("/report", files={"file": ("cv.pdf", cv_bytes(), "application/pdf")},
                               data=form)
    assert response.status_code == 200, response.text
    return response.text


# --- the report's wording ----------------------------------------------------------------------


def test_without_names_the_report_is_exactly_what_it_was(response) -> None:
    assert build_report(response, None) == build_report(response)
    assert build_report(response, {}) == build_report(response)


def test_a_skill_with_a_name_is_printed_in_ukrainian_and_the_sentence_stays_english(
        response) -> None:
    required = group(0, 2, [uk_skill("k8s", "Kubernetes", "c:k8s"), skill("Terraform", None)])

    job = report_of(response, one_match(response, required=required), ).best[0]
    localized = build_report(
        response.model_copy(update={"matches": [one_match(response, required=required)]}),
        NAMES).best[0]

    assert job.missing == "Missing: Kubernetes, Terraform."
    assert localized.missing == "Missing: Kubernetes (оркестрація), Terraform."
    assert isinstance(localized.missing, Line)
    assert localized.missing.uk == ("Kubernetes (оркестрація)",)


def test_a_skill_the_names_do_not_know_keeps_its_english_name(response) -> None:
    required = group(0, 1, [uk_skill("go", "Go language", "c:go")])
    match = one_match(response, required=required)

    job = build_report(response.model_copy(update={"matches": [match]}), NAMES).best[0]

    assert job.missing == "Missing: Go language." and job.missing.uk == ()


def test_a_name_that_is_the_same_word_as_the_english_one_is_not_marked_as_ukrainian(
        response) -> None:
    required = group(0, 1, [uk_skill("sql", "SQL", SQL)])
    match = one_match(response, required=required)

    job = build_report(response.model_copy(update={"matches": [match]}), NAMES).best[0]

    assert job.missing == "Missing: SQL." and job.missing.uk == ()


def test_unrecognised_skills_are_never_renamed(response) -> None:
    odd = [SkillOut(name="Foo", label=None, concept_id=None, recognised=False)]
    profile = response.profile.model_copy(update={"skills": odd})

    report = build_report(response.model_copy(update={"profile": profile}), {"": "X", "Foo": "Y"})

    assert report.unrecognised == ["Foo"] and report.skills == []


def test_your_skills_are_listed_in_ukrainian_and_marked(response) -> None:
    report = build_report(response, NAMES)

    by_label = {s.label: s for s in report.skills}
    assert by_label[PYTHON_UK].uk is True
    assert by_label["SQL"].uk is False  # same word in both languages


def test_a_reading_names_the_standard_skill_in_ukrainian(response) -> None:
    profile = response.profile.model_copy(update={"skills": [uk_skill("JS", "JavaScript", "c:js")]})

    report = build_report(response.model_copy(update={"profile": profile}), NAMES)

    assert report.readings == ["JS was read as ДжейЕс."]
    assert report.readings[0].uk == ("ДжейЕс",)


def test_no_reading_is_shown_when_the_person_wrote_the_name_that_is_shown(response) -> None:
    profile = response.profile.model_copy(update={"skills": [
        uk_skill("  джейес ", "JavaScript", "c:js")]})

    report = build_report(response.model_copy(update={"profile": profile}), NAMES)

    assert report.readings == []


def test_within_reach_and_skills_to_learn_use_the_names_too(response) -> None:
    match = one_match(response, vacancy={"id": "j1", "title": "Data engineer", "company": "Acme"})
    insights = Insights(
        computed=True,
        within_reach=[WithinReach(vacancy_id="j1", missing=[
            uk_skill("k8s", "Kubernetes", "c:k8s"), uk_skill("JS", "JavaScript", "c:js")])],
        skills_to_learn=[SkillToLearn(skill=uk_skill("k8s", "Kubernetes", "c:k8s"), unlocks=2,
                                      vacancy_ids=["j1"])])

    report = report_of(response, match, insights=insights)
    localized = build_report(
        response.model_copy(update={"matches": [match], "insights": insights}), NAMES)

    assert report.reach == [
        "Data engineer at Acme: add Kubernetes and JavaScript to reach a strong match."]
    assert localized.reach == [
        "Data engineer at Acme: add Kubernetes (оркестрація) and ДжейЕс to reach a strong match."]
    assert localized.reach[0].uk == ("Kubernetes (оркестрація)", "ДжейЕс")
    assert localized.learn == [
        "Kubernetes (оркестрація): would make 2 more jobs a strong match."]
    assert localized.learn[0].uk == ("Kubernetes (оркестрація)",)


def test_only_the_names_that_are_shown_are_marked_when_the_list_is_cut(response) -> None:
    gaps = [uk_skill(f"s{i}", f"S{i}", f"c:{i}") for i in range(8)]
    names = {f"c:{i}": f"Навичка {i}" for i in range(8)}
    match = one_match(response, required=group(0, 8, gaps))

    job = build_report(response.model_copy(update={"matches": [match]}), names).best[0]

    assert job.missing.endswith("and 2 more.") and len(job.missing.uk) == 6
    assert "Навичка 5" in job.missing and "Навичка 6" not in job.missing


def test_the_names_change_no_number_and_no_order(response) -> None:
    plain, localized = build_report(response), build_report(response, NAMES)

    assert localized.figures == plain.figures
    assert [(j.rank, j.title, j.band, j.fit, j.coverage, j.segments) for j in localized.rest] == [
        (j.rank, j.title, j.band, j.fit, j.coverage, j.segments) for j in plain.rest]
    assert [s.unsure for s in localized.skills] == [s.unsure for s in plain.skills]


def test_a_line_is_a_plain_string_that_also_lists_its_names() -> None:
    line = Line("Missing: Б.", ("Б",))

    assert line == "Missing: Б." and isinstance(line, str) and line.uk == ("Б",)
    assert Line("x").uk == ()
    assert json.dumps({"a": line}) == '{"a": "Missing: \\u0411."}'


# --- marking the names for screen readers ------------------------------------------------------


def render(text, *names) -> str:
    return routes._templates.from_string("{{ value | marked }}").render(value=Line(text, names))


def test_a_sentence_with_no_ukrainian_name_is_printed_exactly_as_before() -> None:
    assert render("Missing: A & B.") == "Missing: A &amp; B."
    assert routes._templates.from_string("{{ v | marked }}").render(v="plain <b>") == (
        "plain &lt;b&gt;")


def test_each_ukrainian_name_is_wrapped_and_the_rest_is_escaped() -> None:
    html = render("Missing: Докер, Kubernetes & Докер.", "Докер")

    assert html == ('Missing: <span lang="uk">Докер</span>, Kubernetes &amp; '
                    '<span lang="uk">Докер</span>.')


def test_the_longer_name_wins_when_one_name_is_inside_another() -> None:
    html = render("add Дані та Бази даних.", "Дані", "Бази даних", "Бази")

    assert html == 'add <span lang="uk">Дані</span> та <span lang="uk">Бази даних</span>.'


def test_names_with_pattern_characters_are_matched_literally() -> None:
    html = render("add C++ (мова) and .NET*.", "C++ (мова)", ".NET*")

    assert html == 'add <span lang="uk">C++ (мова)</span> and <span lang="uk">.NET*</span>.'


def test_html_in_a_name_or_around_it_is_escaped() -> None:
    html = render(f"{XSS} and <i>Докер</i> and {XSS}", "<i>Докер</i>")

    assert "<script>" not in html and "<i>" not in html
    assert '<span lang="uk">&lt;i&gt;Докер&lt;/i&gt;</span>' in html
    assert "&lt;script&gt;" in html


def test_an_empty_name_is_ignored() -> None:
    assert render("text", "", "") == "text"


def test_every_sentence_that_can_hold_a_name_goes_through_the_filter() -> None:
    report_template = (routes.TEMPLATE_DIR / "report.html").read_text(encoding="utf-8")

    assert report_template.count("| marked") == 4  # missing, reach, learn, readings
    assert '<li{% if skill.uk %} lang="uk"{% endif %}>' in report_template
    assert "|safe" not in report_template.replace(" ", "")  # no way around escaping


# --- the upload page ---------------------------------------------------------------------------


def test_the_upload_form_offers_the_two_languages_with_english_first_by_default() -> None:
    with TestClient(create_app(make_services())) as client:
        html = client.get("/").text

    assert '<select id="lang" name="lang">' in html
    assert '<option value="en" selected>English</option>' in html
    assert '<option value="uk" lang="uk">Українська</option>' in html
    assert 'for="lang">Skill names in the report</label>' in html


@pytest.mark.parametrize("header, picked", [
    ("uk-UA,uk;q=0.9,en;q=0.8", "uk"), ("en-US,en;q=0.9", "en"), ("", "en"),
])
def test_the_form_preselects_the_browsers_language(header, picked) -> None:
    with TestClient(create_app(make_services())) as client:
        html = client.get("/", headers={"Accept-Language": header}).text

    selected = {"en": '<option value="en" selected>English</option>',
                "uk": '<option value="uk" lang="uk" selected>Українська</option>'}[picked]
    assert selected in html
    language_select = html.split('<select id="lang"', 1)[1].split("</select>", 1)[0]
    assert language_select.count(" selected>") == 1


def test_the_language_choice_is_part_of_the_form_and_needs_no_script() -> None:
    with TestClient(create_app(make_services())) as client:
        html = client.get("/").text

    form = html.split('<form id="upload"', 1)[1].split("</form>", 1)[0]
    assert 'name="lang"' in form and 'name="file"' in form


# --- the report page ---------------------------------------------------------------------------


def test_asking_for_ukrainian_prints_the_names_in_ukrainian_and_marks_them() -> None:
    html = page(services_with(), lang="uk")

    assert f'<li lang="uk">{PYTHON_UK}</li>' in html
    assert "<li>SQL</li>" in html  # the same word in both languages is not marked
    assert "Role both" in html and "You have 2 of 2 required skills." in html
    assert f'<li>Python was read as <span lang="uk">{PYTHON_UK}</span>.</li>' in html


def test_asking_for_english_or_nothing_never_reads_the_names() -> None:
    for form in ({"lang": "en"}, {}, {"lang": ""}, {"lang": "ru"}, {"lang": XSS}):
        services = services_with()

        html = page(services, **form)

        assert PYTHON_UK not in html and 'lang="uk"' not in html, form
        assert services.calls == [], form  # type: ignore[attr-defined]
        assert "<li>Python</li>" in html


def test_the_names_are_read_once_per_ukrainian_report() -> None:
    services = services_with()

    page(services, lang="uk")

    assert services.calls == [1]  # type: ignore[attr-defined]


@pytest.mark.parametrize("names", [{}, {"other": "Інше"}])
def test_a_report_without_names_is_the_english_one(names) -> None:
    english = page(services_with(), lang="en")

    assert page(services_with(names), lang="uk") == english


def test_names_that_cannot_be_read_give_the_english_report_not_an_error(caplog) -> None:
    def broken():
        raise RuntimeError(f"cannot connect {SECRETS[0]}")

    with caplog.at_level(logging.DEBUG):
        html = page(make_services(skill_names=broken), lang="uk")

    assert "<li>Python</li>" in html and 'lang="uk"' not in html
    assert "RuntimeError" in caplog.text and SECRETS[0] not in caplog.text


def test_the_ukrainian_report_is_as_private_and_locked_down_as_the_english_one() -> None:
    with TestClient(create_app(services_with())) as client:
        response = client.post("/report", files={"file": ("cv.pdf", cv_bytes())},
                               data={"lang": "uk"})

    for name, value in routes.PAGE_HEADERS.items():
        assert response.headers[name] == value, name


def test_text_from_the_names_is_escaped_in_the_page() -> None:
    html = page(services_with({PYTHON: f"{XSS}<b>"}), lang="uk")

    assert XSS not in html and "<b>" not in html
    assert "&lt;script&gt;" in html


def test_the_choice_is_read_by_the_stream_too_and_gives_the_same_page() -> None:
    blocking = page(services_with(analyze=analyze_with_stages), lang="uk")
    with TestClient(create_app(services_with(analyze=analyze_with_stages))) as client:
        response = client.post("/report", files={"file": ("cv.pdf", cv_bytes())},
                               data={"lang": "uk"}, headers={"Accept": "text/event-stream"})

    events = parse(response.text)
    assert events[-1][0] == "done" and events[-1][1]["html"] == blocking
    assert PYTHON_UK in events[-1][1]["html"]
    assert [d["step"] for n, d in events if n == "step"] == list(progress.STEP_IDS)


def test_the_language_field_changes_nothing_else_about_a_request() -> None:
    services = services_with()
    with TestClient(create_app(services)) as client:
        no_file = client.post("/report", data={"lang": "uk"})
        wrong_type = post(client, b"x", "cv.txt", data={"lang": "uk"})

    assert no_file.status_code == 400 and wrong_type.status_code == 415
    assert services.calls == []  # type: ignore[attr-defined]


def test_a_file_sent_in_the_language_field_is_ignored() -> None:
    services = services_with()
    with TestClient(create_app(services)) as client:
        response = client.post("/report", files={"file": ("cv.pdf", cv_bytes()),
                                                 "lang": ("lang.txt", b"uk")})

    assert response.status_code == 200 and PYTHON_UK not in response.text


def test_the_json_api_still_answers_in_english() -> None:
    services = services_with()
    with TestClient(create_app(services)) as client:
        response = client.post("/v1/analyze", files={"file": ("cv.pdf", cv_bytes())},
                               data={"lang": "uk"}, headers=HEADERS)

    labels = {s["label"] for s in response.json()["profile"]["skills"]}
    assert response.status_code == 200 and PYTHON_UK not in labels and "Python" in labels
    assert services.calls == []  # type: ignore[attr-defined]


def test_services_without_names_have_none_to_offer() -> None:
    services = make_services()

    assert services.skill_names() == {}
    with pytest.raises(UndefinedError):  # the page needs both values from its route
        routes._templates.get_template("index.html").render(
            steps=progress.STEPS, max_bytes=1, field="file", path="/report")


@pytest.mark.parametrize("asked", ["uk", "UK", " uk ", "Uk"])
def test_the_language_code_is_read_without_regard_to_case_or_spaces(asked) -> None:
    assert PYTHON_UK in page(services_with(), lang=asked)


def test_the_report_page_marks_the_names_in_every_section(response) -> None:
    match = one_match(response, vacancy={"id": "j1", "title": "Data engineer", "company": "Acme"},
                      required=group(0, 1, [uk_skill("k8s", "Kubernetes", "c:k8s")]))
    insights = Insights(
        computed=True,
        within_reach=[WithinReach(vacancy_id="j1",
                                  missing=[uk_skill("k8s", "Kubernetes", "c:k8s")])],
        skills_to_learn=[SkillToLearn(skill=uk_skill("k8s", "Kubernetes", "c:k8s"), unlocks=1,
                                      vacancy_ids=["j1"])])
    profile = response.profile.model_copy(update={"skills": [uk_skill("JS", "JavaScript", "c:js")]})
    report = build_report(response.model_copy(update={
        "matches": [match], "insights": insights, "profile": profile}), NAMES)

    html = routes._render("report.html", report=report)

    span = '<span lang="uk">Kubernetes (оркестрація)</span>'
    assert f"Missing: {span}." in html
    assert f"add {span} to reach a strong match." in html
    assert f"<li>{span}: would make 1 more job a strong match.</li>" in html
    assert '<li>JS was read as <span lang="uk">ДжейЕс</span>.</li>' in html
    assert '<li lang="uk">ДжейЕс</li>' in html
