"""JM-43: the web page. The real pipeline runs on a fixture CV with a scripted LLM, as in the
API tests, so nothing here needs a network, a key or a model."""

import ipaddress
import logging
import re

import pytest
from fastapi.testclient import TestClient

from jobpdf.api import app as api
from jobpdf.api.access import (
    AccessControl,
    ApiKeys,
    InMemoryCounterStore,
    RateLimit,
    RateLimiter,
)
from jobpdf.api.app import Services, create_app
from jobpdf.pipeline import MAX_UPLOAD_BYTES, PipelineError
from jobpdf.web import routes
from jobpdf.web.routes import CONTENT_SECURITY_POLICY, PAGE_HEADERS, WEB_DETAILS
from tests.api.test_app import HEADERS, IP_SECRET, analyze_offline, make_services
from tests.pipeline.fakes import SECRETS, VACANCIES, cv_bytes

XSS = "<script>alert('x')</script>"


def post(client, content=None, name="cv.pdf", field="file", **kwargs):
    data = cv_bytes() if content is None else content
    return client.post("/report", files={field: (name, data, "application/pdf")}, **kwargs)


@pytest.fixture
def services() -> Services:
    return make_services()


@pytest.fixture
def client(services) -> TestClient:
    with TestClient(create_app(services), raise_server_exceptions=False) as c:
        yield c


def with_ip_limit(per_hour: int) -> Services:
    limiter = RateLimiter(InMemoryCounterStore(), ip_secret=IP_SECRET,
                          ip_limits=(RateLimit("hour", 3600, per_hour),))
    return make_services(access=AccessControl(ApiKeys({}), limiter))


# --- the pages ---------------------------------------------------------------------------------


def test_the_upload_page_has_a_form_that_posts_a_file_to_the_report(client) -> None:
    response = client.get("/")

    assert response.status_code == 200 and "text/html" in response.headers["content-type"]
    html = response.text
    assert '<form id="upload" method="post" action="/report" enctype="multipart/form-data"' in html
    assert 'name="file" type="file" required' in html
    assert f'data-max-bytes="{MAX_UPLOAD_BYTES}"' in html and "up to 5 MB" in html
    assert 'rel="stylesheet" href="/static/style.css"' in html
    assert '<script src="/static/app.js" defer></script>' in html


def test_the_upload_page_tells_people_where_their_cv_goes(client) -> None:
    html = client.get("/").text

    assert "sent to a language model service (Google Gemini)" in html
    assert "Nothing" in html and "not stored" in html


def test_every_page_is_private_and_locked_down(client) -> None:
    for response in (client.get("/"), post(client)):
        for name, value in PAGE_HEADERS.items():
            assert response.headers[name] == value, name
    assert PAGE_HEADERS["Cache-Control"] == "no-store"
    assert PAGE_HEADERS["Referrer-Policy"] == "no-referrer"
    assert PAGE_HEADERS["X-Frame-Options"] == "DENY"


def test_the_security_headers_are_exactly_these() -> None:
    assert PAGE_HEADERS == {
        "Cache-Control": "no-store",
        "Content-Security-Policy": CONTENT_SECURITY_POLICY,
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
    }
    assert CONTENT_SECURITY_POLICY == (
        "default-src 'none'; style-src 'self'; script-src 'self'; img-src 'self'; "
        "connect-src 'none'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")


def test_the_sentences_a_person_reads_for_a_bad_request_and_a_limit() -> None:
    assert WEB_DETAILS == {
        "invalid_request": "Choose a PDF or Word file and try again.",
        "rate_limited": "You have uploaded a lot of CVs for now. Try again in {wait}.",
    }


def test_a_template_that_names_something_the_page_did_not_pass_fails_loudly() -> None:
    from jinja2 import UndefinedError

    with pytest.raises(UndefinedError):
        routes._templates.from_string("{{ nothing_passed }}").render()
    with pytest.raises(UndefinedError):
        routes._templates.get_template("error.html").render(detail="x")  # no code


def test_the_content_security_policy_allows_nothing_inline_or_remote() -> None:
    policy = dict(part.strip().split(" ", 1) for part in CONTENT_SECURITY_POLICY.split(";"))

    assert policy["default-src"] == "'none'" and policy["connect-src"] == "'none'"
    assert policy["script-src"] == "'self'" and policy["style-src"] == "'self'"
    assert policy["form-action"] == "'self'" and policy["frame-ancestors"] == "'none'"
    assert "unsafe" not in CONTENT_SECURITY_POLICY and "*" not in CONTENT_SECURITY_POLICY


def test_the_json_api_is_unchanged_and_the_pages_are_not_in_its_description(client) -> None:
    assert client.post("/v1/analyze", files={"file": ("cv.pdf", cv_bytes())}).status_code == 401
    paths = client.get("/openapi.json").json()["paths"]

    assert "/v1/analyze" in paths and "/" not in paths and "/report" not in paths


# --- the report --------------------------------------------------------------------------------


def test_a_cv_without_any_key_gets_the_report_page(client) -> None:
    response = post(client)

    assert response.status_code == 200 and "text/html" in response.headers["content-type"]
    html = response.text
    assert "<h1>Your CV report</h1>" in html
    assert "Role both" in html and "You have 2 of 2 required skills." in html
    assert "Missing: rust." in html and "Strong match" in html
    assert "How the fit is scored" in html and "Check another CV" in html
    assert 'name="robots" content="noindex"' in html


def test_the_report_counts_in_the_metrics_like_an_api_call(client, services) -> None:
    post(client)

    snapshot = services.recorder.snapshot()
    assert snapshot.analyses.total == 1 and snapshot.analyses.ok == 1


def test_the_report_shows_all_matches_not_one_page(client) -> None:
    html = post(client).text

    assert html.count('<li class="job">') == 3  # best matches (all three fit in the top five)
    assert html.count("<tr>") == 4  # header row + three matches


def test_no_file_of_the_cv_is_left_behind_after_a_report_or_a_failure(
        client, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)  # the conftest points the pipeline's temp folder under tmp_path

    post(client)
    post(client, b"not a pdf at all")

    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


# --- nothing from a CV or a posting can run as code ----------------------------------------------


def hostile_services(**vacancy_changes) -> Services:
    hostile = [v.model_copy(update=vacancy_changes) for v in VACANCIES]
    return make_services(vacancies=lambda: hostile)


def report_for(services: Services) -> str:
    with TestClient(create_app(services)) as client:
        return post(client).text


def test_text_from_a_vacancy_is_escaped() -> None:
    html = report_for(hostile_services(title=XSS, company=XSS, location_raw=XSS, source=XSS))

    assert "<script>alert" not in html
    assert "&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt;" in html


def test_text_from_the_cv_is_escaped(services) -> None:
    def analyze(*args, **kwargs):
        result = analyze_offline(*args, **kwargs)
        profile = result.profile.model_copy(update={"summary": XSS})
        return result.model_copy(update={"profile": profile})

    html = report_for(make_services(analyze=analyze))

    assert "<script>alert" not in html and "&lt;script&gt;" in html


@pytest.mark.parametrize("url", ["javascript:alert(1)", "JAVASCRIPT:alert(1)",
                                 "data:text/html,<b>x</b>", "vbscript:x", "//evil.example/x"])
def test_a_vacancy_address_that_is_not_http_never_becomes_a_link(url) -> None:
    html = report_for(hostile_services(url=url))

    assert not re.search(r"<a [^>]*href=\"(?!/)(?!#)", html.replace('href="/static', ""))
    assert url.lower() not in html.lower()


def test_a_vacancy_address_that_is_http_becomes_a_safe_link() -> None:
    html = report_for(hostile_services(url="https://example.com/jobs/1?a=1&b=2"))

    link = re.search(r'<a href="([^"]+)"([^>]*)>Role both</a>', html)
    assert link and link.group(1) == "https://example.com/jobs/1?a=1&amp;b=2"
    for part in ('rel="noopener noreferrer nofollow"', 'target="_blank"'):
        assert part in link.group(2)


def test_a_quote_in_an_address_cannot_break_out_of_the_attribute() -> None:
    html = report_for(hostile_services(url='https://example.com/"onmouseover="alert(1)'))

    assert 'onmouseover="alert' not in html


# --- failures ----------------------------------------------------------------------------------


@pytest.mark.parametrize("name, content, status, sentence", [
    ("cv.txt", b"plain text", 415, api.ERROR_DETAILS["unsupported_type"]),
    ("cv.pdf", b"not a pdf at all", 422, api.ERROR_DETAILS["corrupt"]),
])
def test_a_file_that_cannot_be_read_gets_a_page_with_the_reason(
        client, name, content, status, sentence) -> None:
    response = post(client, content, name)

    assert response.status_code == status and "text/html" in response.headers["content-type"]
    assert sentence in response.text and "That did not work" in response.text
    assert 'role="alert"' in response.text and "Try another file" in response.text
    assert response.headers["cache-control"] == "no-store"


def test_a_request_without_the_file_field_asks_for_a_file(client) -> None:
    response = post(client, field="document")

    assert response.status_code == 400
    assert WEB_DETAILS["invalid_request"] in response.text


def test_a_file_over_the_size_limit_is_refused_with_the_limit_in_the_sentence(client) -> None:
    response = post(client, b"x" * (MAX_UPLOAD_BYTES + 1000))

    assert response.status_code == 413 and "5 MB" in response.text


def test_a_failed_pipeline_stage_uses_the_same_status_as_the_api(services) -> None:
    def analyze(*args, **kwargs):
        raise PipelineError("extraction", "llm_error")

    with TestClient(create_app(make_services(analyze=analyze))) as client:
        response = post(client)

    assert response.status_code == api.PIPELINE_STATUS["llm_error"]
    assert api.ERROR_DETAILS["llm_error"] in response.text


def test_a_failure_code_the_page_has_never_heard_of_is_a_server_error_with_a_generic_sentence(
        ) -> None:
    def analyze(*args, **kwargs):
        raise PipelineError("extraction", "brand_new_code")

    with TestClient(create_app(make_services(analyze=analyze))) as client:
        response = post(client)

    assert response.status_code == 500 and api.GENERIC_DETAIL in response.text


def test_a_failure_is_counted_in_the_metrics_like_an_api_failure() -> None:
    services = make_services()
    with TestClient(create_app(services), raise_server_exceptions=False) as client:
        post(client, b"not a pdf at all")

    snapshot = services.recorder.snapshot()
    assert snapshot.analyses.total == 1 and snapshot.analyses.ok == 0
    assert sum(snapshot.errors.values()) == 1


def test_an_unexpected_error_is_a_generic_page_that_repeats_nothing(caplog) -> None:
    def analyze(*args, **kwargs):
        raise RuntimeError(f"boom {SECRETS[0]} {XSS}")

    with TestClient(create_app(make_services(analyze=analyze)),
                    raise_server_exceptions=False) as client, caplog.at_level(logging.DEBUG):
        response = post(client)

    assert response.status_code == 500 and api.ERROR_DETAILS["internal_error"] in response.text
    assert SECRETS[0] not in response.text and "boom" not in response.text
    assert SECRETS[0] not in caplog.text and "RuntimeError" in caplog.text


def test_a_page_that_cannot_be_drawn_is_the_same_generic_page(monkeypatch, caplog, client) -> None:
    def broken(response):
        raise KeyError(SECRETS[0])

    monkeypatch.setattr(routes, "build_report", broken)

    with caplog.at_level(logging.DEBUG):
        response = post(client)

    assert response.status_code == 500 and api.ERROR_DETAILS["internal_error"] in response.text
    assert SECRETS[0] not in response.text + caplog.text and "KeyError" in caplog.text


def test_no_error_page_contains_cv_text(client) -> None:
    for response in (post(client, b"%PDF-1.4 broken " + SECRETS[0].encode(), "cv.pdf"),
                     post(client, SECRETS[1].encode(), "cv.txt")):
        assert not any(secret in response.text for secret in SECRETS)


# --- limits ------------------------------------------------------------------------------------


def test_uploads_are_limited_per_address_and_the_page_says_when_to_come_back() -> None:
    with TestClient(create_app(with_ip_limit(2))) as client:
        assert [post(client).status_code for _ in range(2)] == [200, 200]
        refused = post(client)

    assert refused.status_code == 429
    assert "Try again in" in refused.text and "{wait}" not in refused.text
    assert int(refused.headers["retry-after"]) > 0
    assert re.search(r"Try again in \d+ minutes?\.", refused.text)


def test_a_refused_upload_does_not_start_an_analysis_and_is_counted() -> None:
    services = with_ip_limit(1)
    with TestClient(create_app(services)) as client:
        post(client)
        post(client)

    snapshot = services.recorder.snapshot()
    assert snapshot.analyses.total == 1 and snapshot.access_denied == {"rate_limited": 1}


def test_the_limit_is_not_shared_with_api_keys() -> None:
    with TestClient(create_app(with_ip_limit(1))) as client:
        assert post(client).status_code == 200
        assert post(client).status_code == 429
        assert client.get("/health").status_code in (200, 503)


def test_when_the_server_is_busy_the_page_says_so_with_a_retry_time() -> None:
    services = make_services(max_concurrent=1)
    assert services.slots.acquire(blocking=False)  # someone else's analysis holds the only slot
    with TestClient(create_app(services)) as client:
        response = post(client)

    assert response.status_code == 503 and api.ERROR_DETAILS["busy"] in response.text
    assert response.headers["retry-after"] == str(api.BUSY_RETRY_AFTER_S)
    assert services.recorder.snapshot().access_denied == {"busy": 1}


def test_the_slot_is_given_back_after_a_failure() -> None:
    services = make_services(max_concurrent=1)

    def analyze(*args, **kwargs):
        raise PipelineError("parse", "corrupt")

    services.analyze = analyze
    with TestClient(create_app(services)) as client:
        assert post(client).status_code == 422
        assert post(client).status_code == 422  # a second try is not "busy"


def test_the_wait_is_written_in_whole_minutes() -> None:
    assert routes._wait(1) == "1 minute" and routes._wait(60) == "1 minute"
    assert routes._wait(61) == "2 minutes" and routes._wait(3600) == "60 minutes"
    assert routes._wait(0) == "1 minute"


@pytest.mark.parametrize("host, counted", [
    ("203.0.113.9", "203.0.113.9"), ("2001:db8::1", "2001:db8::1"),
    (" 198.51.100.7 ", "198.51.100.7"),
    ("testclient", routes.UNKNOWN_CLIENT), ("", routes.UNKNOWN_CLIENT),
    ("/run/app.sock", routes.UNKNOWN_CLIENT), ("999.1.1.1", routes.UNKNOWN_CLIENT)])
def test_a_client_without_a_usable_address_is_limited_not_failed(host, counted) -> None:
    class Request:
        client = type("Peer", (), {"host": host})()

    assert routes._client_ip(Request()) == counted


def test_a_request_with_no_peer_at_all_is_counted_with_the_unknown_bucket() -> None:
    class Request:
        client = None

    assert routes._client_ip(Request()) == routes.UNKNOWN_CLIENT


def test_unknown_clients_share_one_limit(monkeypatch) -> None:
    with TestClient(create_app(with_ip_limit(1))) as client:  # host "testclient" is not an IP
        assert post(client).status_code == 200
        assert post(client).status_code == 429


def test_the_address_and_the_declared_size_are_what_the_access_check_gets(monkeypatch) -> None:
    seen = []
    services = make_services()
    original = services.access.authorize_web

    def spy(client_ip, length):
        seen.append((client_ip, length))
        return original(client_ip, length)

    monkeypatch.setattr(services.access, "authorize_web", spy)
    with TestClient(create_app(services), client=("203.0.113.9", 5555)) as client:
        sent = client.post("/report", files={"file": ("cv.pdf", cv_bytes())})

    assert sent.status_code == 200
    assert len(seen) == 1 and seen[0][0] == "203.0.113.9"
    assert isinstance(seen[0][1], int) and seen[0][1] > len(cv_bytes())


def test_unknown_clients_are_kept_apart_from_the_local_machine() -> None:
    address = ipaddress.ip_address(routes.UNKNOWN_CLIENT)

    assert address.is_unspecified and not address.is_loopback  # health checks come from 127.0.0.1


# --- the files the pages load --------------------------------------------------------------------


@pytest.mark.parametrize("path, kind", [
    ("/static/style.css", "text/css"), ("/static/app.js", "javascript"),
    ("/static/favicon.svg", "image/svg+xml")])
def test_the_static_files_are_served(client, path, kind) -> None:
    response = client.get(path)

    assert response.status_code == 200 and kind in response.headers["content-type"]
    assert len(response.content) > 100


@pytest.mark.parametrize("path", ["/static/missing.css", "/static/../routes.py",
                                  "/static/%2e%2e/routes.py", "/static/", "/static"])
def test_other_static_paths_are_not_found(client, path) -> None:
    assert client.get(path).status_code in (404, 307)
    assert "RuntimeError" not in client.get(path).text


def test_an_uploaded_key_header_is_not_needed_and_is_ignored(client) -> None:
    assert post(client, headers=HEADERS).status_code == 200
