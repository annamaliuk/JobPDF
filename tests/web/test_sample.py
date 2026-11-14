"""JM-52: "Try a sample CV" on the upload page, end to end. The real pipeline runs on the
fixture CV with a scripted model, so no test needs a network, a key or a database."""

import html as htmllib
import re

import pytest
from fastapi.testclient import TestClient

from jobpdf import pipeline
from jobpdf.api.access import AccessControl, ApiKeys, InMemoryCounterStore, RateLimit, RateLimiter
from jobpdf.api.app import create_app
from jobpdf.api.sample import MemoCaller, SampleCV
from jobpdf.extraction.llm import LLMResponseError
from jobpdf.web import routes
from jobpdf.web.view import build_report
from tests.api.test_app import IP_SECRET, make_services
from tests.normalization.fakes import FakeEmbedder
from tests.pipeline.fakes import (
    PYTHON,
    SECRETS,
    TODAY,
    FakeNormalizer,
    caller,
    cv_bytes,
    extracted,
    reply,
)
from tests.web.test_abroad import ABROAD
from tests.web.test_progress import parse

STREAM = {"Accept": "text/event-stream"}


def analyze(content, filename, vacancies, **kwargs):
    """The real pipeline with the caller the app passes: the sample's memo, or svc.caller."""
    return pipeline.analyze_cv(
        content, filename, vacancies, caller=kwargs["caller"], normalizer=FakeNormalizer(),
        embedder=FakeEmbedder(), today=TODAY, scorer=kwargs.get("scorer"),
        adjustments=kwargs.get("adjustments"), on_stage=kwargs.get("on_stage"))


class Upload:
    """svc.caller for uploads: a fresh scripted answer for every upload, counted."""

    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def call(self, **request):
        self.calls += 1
        return caller().call(**request)


def sample_of(answers=None) -> SampleCV:
    inner = caller(*(answers or [reply(extracted())]))
    return SampleCV(content=cv_bytes(), filename="sample_cv.pdf",
                    caller=MemoCaller(lambda: inner))


def services(sample="default", **changes):
    sample = sample_of() if sample == "default" else sample
    base = {"analyze": analyze, "caller": Upload(), "sample": sample}
    return make_services(**{**base, **changes})


def client_for(svc) -> TestClient:
    return TestClient(create_app(svc), raise_server_exceptions=False)


def try_sample(client, headers=None, **form):
    return client.post(routes.SAMPLE_REPORT_PATH, data=form, headers=headers or {})


# --- the upload page ---------------------------------------------------------------------------


def test_the_upload_page_offers_the_sample_inside_the_form() -> None:
    with client_for(services()) as client:
        html = client.get("/").text

    form = html.split('<form id="upload"', 1)[1].split("</form>", 1)[0]
    button = re.search(r"<button[^>]*>Try a sample CV</button>", form, re.S).group(0)
    assert 'formaction="/report/sample"' in button and "formnovalidate" in button
    assert 'type="submit"' in button and 'class="secondary"' in button
    assert 'aria-describedby="sample-hint"' in button and 'id="sample-hint"' in form
    assert '<a href="/sample-cv" target="_blank" rel="noopener">See the sample CV</a>' in form
    assert form.index("Check my CV") < form.index("Try a sample CV")


def test_without_a_sample_the_page_offers_none() -> None:
    with client_for(services(sample=None)) as client:
        html = client.get("/").text

    assert "Try a sample CV" not in html and "/sample-cv" not in html
    assert "/report/sample" not in html and "Check my CV" in html


def test_the_paths_in_the_pages_are_the_routes() -> None:
    assert routes.SAMPLE_REPORT_PATH == "/report/sample" and routes.SAMPLE_CV_PATH == "/sample-cv"
    report = (routes.TEMPLATE_DIR / "report.html").read_text()
    assert f'href="{routes.SAMPLE_CV_PATH}"' in report


# --- the sample report -------------------------------------------------------------------------


def test_the_sample_report_says_whose_cv_it_is() -> None:
    with client_for(services()) as client:
        response = try_sample(client)

    assert response.status_code == 200
    html = response.text
    assert "<title>Sample CV report" in html and "<h1>Report for a sample CV</h1>" in html
    assert "Your CV report" not in html
    note = html.split('<p class="sample-note">', 1)[1].split("</p>", 1)[0]
    assert "not yours" in note and 'href="/sample-cv"' in note and 'href="/"' in note
    assert '<a class="button" href="/">Check your own CV</a>' in html
    assert "Check another CV" not in html
    assert response.headers["Cache-Control"] == "no-store"
    assert "default-src 'none'" in response.headers["Content-Security-Policy"]


def test_an_upload_report_has_no_sample_note() -> None:
    with client_for(services()) as client:
        html = client.post("/report", files={"file": ("cv.pdf", cv_bytes())}).text

    assert "<h1>Your CV report</h1>" in html and "sample-note" not in html
    assert "Check another CV" in html and "Check your own CV" not in html


def test_the_model_is_asked_once_for_any_number_of_sample_reports() -> None:
    svc = services()  # the sample's scripted model has one answer: a second ask would fail
    with client_for(svc) as client:
        pages = [try_sample(client).text for _ in range(3)]

    assert pages[0] == pages[1] == pages[2]
    assert svc.sample.caller.calls == 1 and svc.caller.calls == 0


def test_uploads_never_use_the_samples_memo() -> None:
    svc = services()
    with client_for(svc) as client:
        try_sample(client)
        for _ in range(2):
            assert client.post("/report", files={"file": ("cv.pdf", cv_bytes())}).status_code == 200

    assert svc.caller.calls == 2 and svc.sample.caller.calls == 1


def test_the_analysis_gets_the_memo_for_the_sample_and_the_normal_caller_for_uploads() -> None:
    seen = []

    def spy(content, filename, vacancies, **kwargs):
        seen.append((filename, kwargs["caller"]))
        return analyze(content, filename, vacancies, **kwargs)

    svc = services(analyze=spy)
    with client_for(svc) as client:
        try_sample(client)
        client.post("/report", files={"file": ("mine.pdf", cv_bytes())})

    assert seen == [("sample_cv.pdf", svc.sample.caller), ("mine.pdf", svc.caller)]


def test_the_forms_choices_apply_to_the_sample() -> None:
    svc = services(vacancies=lambda: ABROAD,
                   skill_names=lambda: {PYTHON: "Пайтон"})
    with client_for(svc) as client:
        html = htmllib.unescape(try_sample(client, lang="uk", home_country="UA").text)

    assert "Пайтон" in html
    assert "Role both" not in html and "outside Ukraine" in html


def test_a_file_sent_along_with_the_sample_is_ignored() -> None:
    with client_for(services()) as client:
        response = client.post(routes.SAMPLE_REPORT_PATH,
                               files={"file": ("other.pdf", b"not a pdf at all")},
                               data={"lang": "en"})

    assert response.status_code == 200 and "Report for a sample CV" in response.text


def test_junk_choices_are_handled_like_on_the_upload_form() -> None:
    with client_for(services(vacancies=lambda: ABROAD)) as client:
        html = try_sample(client, lang="xx", home_country="Atlantis").text

    assert "Report for a sample CV" in html and "Role both" in html and "left out" not in html


def test_the_stream_gives_the_same_page_as_a_plain_request() -> None:
    svc = services()
    with client_for(svc) as client:
        plain = try_sample(client).text
        events = parse(try_sample(client, headers=STREAM).text)

    assert [e[0] for e in events][-1] == "done" and events[-1][1]["html"] == plain
    assert svc.sample.caller.calls == 1


def test_a_failing_model_gives_the_usual_error_and_is_asked_again_next_time() -> None:
    svc = services(sample=sample_of([LLMResponseError("no tool call"), reply(extracted())]))
    with client_for(svc) as client:
        failed = try_sample(client)
        worked = try_sample(client)

    assert failed.status_code == 502 and "Report for a sample CV" not in failed.text
    assert worked.status_code == 200 and "Report for a sample CV" in worked.text
    assert svc.sample.caller.calls == 1


def test_a_sample_report_is_counted_like_any_analysis() -> None:
    svc = services()
    with client_for(svc) as client:
        try_sample(client)
        try_sample(client)

    assert svc.recorder.snapshot().analyses.total == 2


def test_the_sample_shares_the_per_address_limit_with_uploads() -> None:
    limiter = RateLimiter(InMemoryCounterStore(), ip_secret=IP_SECRET,
                          ip_limits=(RateLimit("hour", 3600, 2),))
    svc = services(access=AccessControl(ApiKeys({}), limiter))
    with client_for(svc) as client:
        assert client.post("/report", files={"file": ("cv.pdf", cv_bytes())}).status_code == 200
        assert try_sample(client).status_code == 200
        refused = try_sample(client)

    assert refused.status_code == 429 and "Retry-After" in refused.headers
    assert svc.recorder.snapshot().access_denied == {"rate_limited": 1}


def test_the_sample_waits_for_a_free_analysis_slot_like_an_upload() -> None:
    svc = services(max_concurrent=1)
    svc.slots.acquire()
    try:
        with client_for(svc) as client:
            busy = try_sample(client)
    finally:
        svc.slots.release()

    assert busy.status_code == 503 and "Retry-After" in busy.headers


def test_the_slot_is_given_back_after_a_sample_report() -> None:
    svc = services(max_concurrent=1)
    with client_for(svc) as client:
        for _ in range(3):
            assert try_sample(client).status_code == 200
        assert try_sample(client, headers=STREAM).status_code == 200
        assert try_sample(client).status_code == 200


def test_a_body_over_the_size_limit_is_refused_before_reading() -> None:
    with client_for(services()) as client:
        response = client.post(routes.SAMPLE_REPORT_PATH, content=b"x",
                               headers={"Content-Length": str(50 * 1024 * 1024),
                                        "Content-Type": "application/x-www-form-urlencoded"})

    assert response.status_code == 413


def test_a_malformed_form_is_a_bad_request() -> None:
    with client_for(services()) as client:
        response = client.post(routes.SAMPLE_REPORT_PATH, content=b"\xd3\xff garbage",
                               headers={"Content-Type": "multipart/form-data; boundary=b"})

    assert response.status_code == 400 and "Report for a sample CV" not in response.text


def test_without_a_sample_the_routes_say_so() -> None:
    with client_for(services(sample=None)) as client:
        report = try_sample(client)
        file = client.get(routes.SAMPLE_CV_PATH)

    for response in (report, file):
        assert response.status_code == 404
        assert "There is no sample CV on this server" in response.text


def test_the_sample_routes_answer_only_their_methods() -> None:
    with client_for(services()) as client:
        assert client.get(routes.SAMPLE_REPORT_PATH).status_code == 405
        assert client.post(routes.SAMPLE_CV_PATH).status_code == 405


# --- the file ----------------------------------------------------------------------------------


def test_the_sample_cv_is_served_for_the_browser_to_show() -> None:
    with client_for(services()) as client:
        response = client.get(routes.SAMPLE_CV_PATH)

    assert response.status_code == 200 and response.content == cv_bytes()
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"] == 'inline; filename="sample-cv.pdf"'
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-cache"


def test_a_docx_sample_is_served_as_a_docx() -> None:
    docx = SampleCV(content=b"PK docx", filename="sample_cv.docx", caller=MemoCaller(caller))
    with client_for(services(sample=docx)) as client:
        response = client.get(routes.SAMPLE_CV_PATH)

    assert response.headers["content-type"].endswith("wordprocessingml.document")
    assert response.headers["content-disposition"] == 'inline; filename="sample-cv.docx"'


def test_serving_the_file_does_not_count_as_an_analysis() -> None:
    svc = services()
    with client_for(svc) as client:
        client.get(routes.SAMPLE_CV_PATH)

    assert svc.recorder.snapshot().analyses.total == 0 and svc.sample.caller.calls == 0


# --- the page model ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def response():
    from jobpdf.api.response import build_response
    from tests.pipeline.fakes import run

    return build_response(run())


def test_a_report_is_for_the_visitors_cv_unless_told_otherwise(response) -> None:
    assert build_report(response).sample is False
    assert build_report(response, sample=True).sample is True


def test_the_sample_note_carries_no_cv_text_and_no_logs_carry_any(caplog) -> None:
    with client_for(services()) as client:
        try_sample(client)

    assert not any(secret in caplog.text for secret in SECRETS)
