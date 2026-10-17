"""JM-36: the FastAPI app. The real pipeline runs on a fixture CV with a scripted LLM, a fake
normalizer and a fake embedder, so no test needs a network, a key or a model."""

import asyncio
import logging
from functools import partial
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from jobpdf.api import app as app_module
from jobpdf.api.access import (
    AccessControl,
    ApiKeys,
    ConfigError,
    InMemoryCounterStore,
    RateLimit,
    RateLimiter,
    hash_key,
    new_api_key,
)
from jobpdf.api.app import (
    BUSY_RETRY_AFTER_S,
    ERROR_DETAILS,
    MAX_BODY_BYTES,
    MAX_PAGE_SIZE,
    PIPELINE_STATUS,
    BodyLimit,
    BodyTooLarge,
    Services,
    create_app,
)
from jobpdf.extraction.llm import FakeToolCaller
from jobpdf.monitoring.health import build_health
from jobpdf.monitoring.metrics import MetricsRecorder
from jobpdf.pipeline import analyze_cv
from tests.monitoring.test_health import NOW, FakeConn, stats_row, write_cache
from tests.normalization.fakes import FakeEmbedder
from tests.pipeline.fakes import (
    SECRETS,
    TODAY,
    VACANCIES,
    FakeNormalizer,
    caller,
    cv_bytes,
)

KEY = new_api_key()
OTHER_KEY = new_api_key()
HEADERS = {"X-API-Key": KEY}
SECRET_DB = "postgresql://jobpdf:hunter2@db.internal/jobpdf"
IP_SECRET = "x" * 32


def analyze_offline(content, filename, vacancies, **kwargs):
    """The real pipeline with a fresh scripted LLM for every request."""
    return analyze_cv(content, filename, vacancies, caller=caller(), normalizer=FakeNormalizer(),
                      embedder=FakeEmbedder(), today=TODAY, scorer=kwargs.get("scorer"))


def access(*, limits=None, max_bytes=None) -> AccessControl:
    keys = ApiKeys({"alice": hash_key(KEY), "bob": hash_key(OTHER_KEY)})
    limiter = RateLimiter(InMemoryCounterStore(), ip_secret=IP_SECRET,
                          key_limits=limits or (RateLimit("minute", 60, 100),))
    return AccessControl(keys, limiter, **({"max_bytes": max_bytes} if max_bytes else {}))


def make_services(**changes) -> Services:
    base = {"access": access(), "vacancies": lambda: VACANCIES, "analyze": analyze_offline}
    return Services(**{**base, **changes})


@pytest.fixture
def services() -> Services:
    return make_services()


@pytest.fixture
def client(services) -> TestClient:
    with TestClient(create_app(services), raise_server_exceptions=False) as c:
        yield c


def post(client, content=None, *, name="cv.pdf", headers=HEADERS, field="file", **params):
    data = cv_bytes() if content is None else content
    return client.post("/v1/analyze", files={field: (name, data, "application/pdf")},
                       headers=headers, params=params)


def error(response) -> dict:
    return response.json()["error"]


# --- analyze: the happy path ----------------------------------------------------------------


def test_a_cv_with_a_valid_key_gets_ranked_matches(client) -> None:
    response = post(client)
    body = response.json()

    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    assert [m["vacancy"]["id"] for m in body["matches"]][:2] == ["both", "one"]
    assert body["pagination"]["total"] == 3 and body["versions"]["response_schema"]
    assert body["meta"]["timings_ms"]["total"] >= 0


def test_the_bearer_scheme_works_too(client) -> None:
    assert post(client, headers={"Authorization": f"Bearer {KEY}"}).status_code == 200


def test_pages_slice_the_ordered_matches(client) -> None:
    first = post(client, page=1, page_size=2).json()
    second = post(client, page=2, page_size=2).json()

    assert [m["vacancy"]["id"] for m in first["matches"]] == ["both", "one"]
    assert [m["vacancy"]["id"] for m in second["matches"]] == ["none"]
    assert first["pagination"]["pages"] == 2


@pytest.mark.parametrize(
    "params", [{"page": 0}, {"page_size": 0}, {"page_size": MAX_PAGE_SIZE + 1}])
def test_bad_paging_is_refused(client, params) -> None:
    assert post(client, **params).status_code == 422


def test_the_largest_page_size_is_200(client) -> None:
    assert post(client, page_size=200).status_code == 200
    assert post(client, page_size=201).status_code == 422


def test_the_openapi_description_builds(client) -> None:
    schema = client.get("/openapi.json").json()

    assert "/v1/analyze" in schema["paths"] and "/health" in schema["paths"]


# --- analyze: access ------------------------------------------------------------------------


def test_no_key_is_401_with_a_stable_code(client) -> None:
    response = post(client, headers={})

    assert response.status_code == 401
    assert error(response) == {"code": "missing_key", "stage": None,
                               "detail": ERROR_DETAILS["missing_key"]}


def test_a_wrong_key_is_401(client) -> None:
    response = post(client, headers={"X-API-Key": "nope"})

    assert response.status_code == 401 and error(response)["code"] == "invalid_key"
    assert response.headers["cache-control"] == "no-store"


def test_the_rate_limit_answers_429_with_retry_after(services) -> None:
    services.access = access(limits=(RateLimit("minute", 60, 2),))
    with TestClient(create_app(services)) as client:
        assert [post(client).status_code for _ in range(2)] == [200, 200]
        limited = post(client)

    assert limited.status_code == 429 and error(limited)["code"] == "rate_limited"
    assert 1 <= int(limited.headers["retry-after"]) <= 60


def test_each_key_has_its_own_budget(services) -> None:
    services.access = access(limits=(RateLimit("minute", 60, 1),))
    with TestClient(create_app(services)) as client:
        assert post(client).status_code == 200
        assert post(client).status_code == 429
        assert post(client, headers={"X-API-Key": OTHER_KEY}).status_code == 200


def test_a_refused_request_never_reaches_the_pipeline(services) -> None:
    calls = []
    services.analyze = lambda *a, **k: calls.append(1)
    with TestClient(create_app(services)) as client:
        post(client, headers={})
        post(client, headers={"X-API-Key": "nope"})

    assert calls == []


def test_a_declared_size_over_the_limit_is_413_before_the_body_is_read(services) -> None:
    services.access = access(max_bytes=1000)
    calls = []
    services.analyze = lambda *a, **k: calls.append(1)
    with TestClient(create_app(services)) as client:
        response = post(client, b"x" * 200_000)

    assert response.status_code == 413 and error(response)["code"] == "too_large"
    assert calls == []


def test_a_body_that_declares_nothing_is_cut_off_at_the_cap(client) -> None:
    def chunks():  # a well-formed start, then a file part that never ends
        yield (b'--b\r\nContent-Disposition: form-data; name="file"; filename="cv.pdf"\r\n'
               b"Content-Type: application/pdf\r\n\r\n")
        for _ in range(MAX_BODY_BYTES // (1024 * 1024) + 2):
            yield b"x" * (1024 * 1024)

    response = client.post("/v1/analyze", content=chunks(), headers={
        **HEADERS, "Content-Type": "multipart/form-data; boundary=b"})

    assert response.status_code == 413 and error(response)["code"] == "too_large"


def test_a_file_over_the_limit_but_within_the_form_overhead_is_refused_when_read(services) -> None:
    services.access = access(max_bytes=1000)
    calls = []
    services.analyze = lambda *a, **k: calls.append(1)
    with TestClient(create_app(services)) as client:
        response = post(client, b"x" * 2000)  # the whole request is under max + overhead

    assert response.status_code == 413 and error(response)["code"] == "too_large"
    assert calls == []


def test_a_declared_size_is_judged_before_any_of_the_body_is_parsed(services) -> None:
    services.access = access(max_bytes=1000)
    with TestClient(create_app(services)) as client:
        response = client.post("/v1/analyze", content=b"not multipart " * 20_000, headers={
            **HEADERS, "Content-Type": "multipart/form-data; boundary=b"})

    assert response.status_code == 413  # parsing it would have been a 400


def test_the_body_limit_counts_bytes_across_chunks() -> None:
    async def run(sizes, cap):
        sent = iter([{"type": "http.request", "body": b"x" * n, "more_body": True}
                     for n in sizes])

        async def receive():
            return next(sent)

        async def inner(scope, receive, send):
            for _ in sizes:
                await receive()

        await BodyLimit(inner, cap)({"type": "http"}, receive, None)

    asyncio.run(run([5, 5], 10))  # exactly the cap passes
    with pytest.raises(BodyTooLarge):
        asyncio.run(run([5, 6], 10))


def test_the_body_limit_leaves_other_scopes_alone() -> None:
    seen = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    asyncio.run(BodyLimit(inner, 1)({"type": "lifespan"}, None, None))

    assert seen == ["lifespan"]


def test_the_pipeline_gets_the_upload_the_vacancies_and_every_service(services) -> None:
    seen = {}

    def spy(content, filename, vacancies, **kwargs):
        seen.update(content=content, filename=filename, vacancies=vacancies, **kwargs)
        return analyze_offline(content, filename, vacancies)

    marker = object()
    services.analyze, services.scorer = spy, marker
    services.caller, services.embedder = "caller", "embedder"
    services.normalizer = "normalizer"
    with TestClient(create_app(services)) as client:
        assert post(client, name="my.cv.PDF").status_code == 200

    assert seen["content"] == cv_bytes() and seen["filename"] == "my.cv.PDF"
    assert list(seen["vacancies"]) == VACANCIES
    assert (seen["scorer"], seen["caller"], seen["embedder"], seen["normalizer"]) == (
        marker, "caller", "embedder", "normalizer")


def test_the_uploaded_form_and_its_temporary_file_are_closed_on_every_path(
        services, monkeypatch) -> None:
    from starlette.datastructures import FormData

    closed = []
    original = FormData.close

    async def spy(self):
        closed.append(1)
        await original(self)

    monkeypatch.setattr(FormData, "close", spy)
    with TestClient(create_app(services)) as client:
        post(client)  # a success
        post(client, field="upload")  # no file field
        services.access = access(max_bytes=1000)
        post(client, b"x" * 2000)  # refused while reading

    assert closed == [1, 1, 1]


def test_the_vacancies_are_asked_for_on_every_request(services) -> None:
    asked = []
    services.vacancies = lambda: asked.append(1) or VACANCIES
    with TestClient(create_app(services)) as client:
        post(client)
        post(client)

    assert asked == [1, 1]


# --- analyze: bad requests and pipeline failures --------------------------------------------


@pytest.mark.parametrize("kwargs", [{"field": "upload"}, {"field": "files"}])
def test_a_form_without_the_file_field_is_400(client, kwargs) -> None:
    response = post(client, **kwargs)

    assert response.status_code == 400 and error(response)["code"] == "invalid_request"


def test_a_text_field_named_file_is_not_a_file(client) -> None:
    response = client.post("/v1/analyze", data={"file": "just text"}, headers=HEADERS)

    assert response.status_code == 400 and error(response)["code"] == "invalid_request"


def test_a_body_that_is_not_a_form_is_400(client) -> None:
    response = client.post("/v1/analyze", json={"file": "x"}, headers=HEADERS)

    assert response.status_code == 400 and error(response)["code"] == "invalid_request"


def test_a_malformed_multipart_body_is_400(client) -> None:
    response = client.post("/v1/analyze", content=b"\xd3\xff garbage", headers={
        **HEADERS, "Content-Type": "multipart/form-data; boundary=b"})

    assert response.status_code == 400 and error(response)["code"] == "invalid_request"


@pytest.mark.parametrize(("content", "name", "status", "code", "stage"), [
    (b"hello", "cv.txt", 415, "unsupported_type", "parse"),
    (b"%PDF-1.7 not really", "cv.pdf", 422, "corrupt", "parse"),
])
def test_unusable_files_are_client_errors(client, content, name, status, code, stage) -> None:
    response = post(client, content, name=name)

    assert response.status_code == status
    assert error(response) == {"code": code, "stage": stage, "detail": ERROR_DETAILS[code]}


def test_a_server_without_an_llm_key_says_503(services) -> None:
    services.analyze = lambda content, filename, vacancies, **k: analyze_cv(
        content, filename, vacancies, normalizer=FakeNormalizer(), embedder=FakeEmbedder())
    with TestClient(create_app(services)) as client:
        response = post(client)

    assert response.status_code == 503
    assert (error(response)["code"], error(response)["stage"]) == ("llm_not_configured",
                                                                   "extraction")


def test_an_llm_failure_is_502_and_its_text_is_not_echoed(services) -> None:
    def failing(content, filename, vacancies, **k):
        boom = FakeToolCaller([RuntimeError("401 for key sk-ant-SECRET and Olena Testenko")])
        return analyze_cv(content, filename, vacancies, caller=boom, normalizer=FakeNormalizer(),
                          embedder=FakeEmbedder())

    services.analyze = failing
    with TestClient(create_app(services)) as client:
        response = post(client)

    assert response.status_code == 502 and error(response)["code"] == "llm_error"
    assert "sk-ant" not in response.text and "Olena" not in response.text


def test_an_unexpected_exception_is_500_without_its_text(services, caplog) -> None:
    def broken(*args, **kwargs):
        raise ValueError(f"leaked {SECRET_DB} and {SECRETS[0]}")

    services.analyze = broken
    with TestClient(create_app(services)) as client, caplog.at_level(logging.DEBUG):
        response = post(client)
        snapshot = client.get("/metrics", headers=HEADERS).json()

    assert response.status_code == 500 and error(response)["code"] == "internal_error"
    assert "hunter2" not in response.text + caplog.text and SECRETS[0] not in response.text
    assert "ValueError" in caplog.text
    assert snapshot["errors"] == {"unexpected": 1}


@pytest.mark.parametrize(("stage", "code", "status"), [
    ("parse", "unsupported_type", 415), ("parse", "too_large", 413), ("parse", "encrypted", 422),
    ("parse", "corrupt", 422), ("parse", "too_many_pages", 422), ("parse", "no_text", 422),
    ("extraction", "llm_not_configured", 503), ("extraction", "llm_error", 502),
    ("extraction", "extraction_invalid", 502),
])
def test_each_pipeline_failure_has_its_own_status(services, stage, code, status) -> None:
    from jobpdf.pipeline import PipelineError

    def fail(*args, **kwargs):
        raise PipelineError(stage, code)

    services.analyze = fail
    with TestClient(create_app(services)) as client:
        response = post(client)

    assert response.status_code == status
    assert error(response) == {"code": code, "stage": stage, "detail": ERROR_DETAILS[code]}
    assert response.headers["cache-control"] == "no-store"


def test_every_detail_is_a_whole_sentence() -> None:
    assert all(len(text) >= 20 and text.endswith(".") for text in ERROR_DETAILS.values())


def test_every_pipeline_status_has_a_fixed_detail() -> None:
    assert set(PIPELINE_STATUS) <= set(ERROR_DETAILS)
    assert all(code in ERROR_DETAILS for code in
               ("missing_key", "invalid_key", "rate_limited", "too_large", "limiter_unavailable",
                "invalid_request", "busy", "internal_error"))


def test_an_unlisted_pipeline_code_is_a_500_with_a_generic_detail(services) -> None:
    from jobpdf.pipeline import PipelineError

    def odd(*args, **kwargs):
        raise PipelineError("ranking", "something_new")

    services.analyze = odd
    with TestClient(create_app(services)) as client:
        response = post(client)

    assert response.status_code == 500
    assert error(response) == {"code": "something_new", "stage": "ranking",
                               "detail": "The request failed."}


def test_the_limiter_being_down_refuses_uploads_with_503(services) -> None:
    class Down(InMemoryCounterStore):
        def hit(self, *args, **kwargs):
            from jobpdf.api.access import LimiterUnavailable

            raise LimiterUnavailable(7)

    services.access = AccessControl(
        ApiKeys({"alice": hash_key(KEY)}), RateLimiter(Down(), ip_secret=IP_SECRET))
    with TestClient(create_app(services)) as client:
        response = post(client)

    assert response.status_code == 503 and error(response)["code"] == "limiter_unavailable"


# --- load shedding ---------------------------------------------------------------------------


def test_when_every_slot_is_taken_the_next_upload_gets_busy_and_a_slot_frees_up(services) -> None:
    services.max_concurrent = 1
    services.__post_init__()
    with TestClient(create_app(services)) as client:
        assert services.slots.acquire(blocking=False)
        busy = post(client)
        services.slots.release()
        after = post(client)
        denied = client.get("/metrics", headers=HEADERS).json()["access_denied"]

    assert busy.status_code == 503 and error(busy)["code"] == "busy"
    assert busy.headers["retry-after"] == str(BUSY_RETRY_AFTER_S)
    assert after.status_code == 200 and denied == {"busy": 1}


def test_a_slot_is_released_after_a_failed_analysis(services) -> None:
    services.max_concurrent = 1
    services.__post_init__()
    services.analyze = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    with TestClient(create_app(services)) as client:
        assert post(client).status_code == 500
        assert post(client).status_code == 500  # not 503 busy


# --- health and metrics ---------------------------------------------------------------------


def health_for(tmp_path: Path, conn) -> partial:
    write_cache(tmp_path, "remotive", 3600)
    return partial(build_health, connect=lambda: conn, environ={"ANTHROPIC_API_KEY": "x"},
                   cache_base=tmp_path, clock=lambda: NOW.timestamp())


def test_health_is_public_and_200_when_everything_is_fine(services, tmp_path) -> None:
    services.health = health_for(tmp_path, FakeConn([stats_row()]))
    with TestClient(create_app(services)) as client:
        response = client.get("/health")

    assert response.status_code == 200 and response.json()["status"] == "ok"
    assert response.json()["checks"]["api_keys"]["data"] == {"keys": "set"}
    assert response.headers["cache-control"] == "no-store"
    assert KEY not in response.text


def test_health_is_503_when_a_dependency_is_down(services, tmp_path) -> None:
    def unreachable():
        raise psycopg.OperationalError(SECRET_DB)

    services.health = health_for(tmp_path, None)
    services.health.keywords["connect"] = unreachable
    with TestClient(create_app(services)) as client:
        response = client.get("/health")

    assert response.status_code == 503 and response.json()["status"] == "down"
    assert "hunter2" not in response.text


def test_health_stays_200_when_degraded(services, tmp_path) -> None:
    services.health = health_for(tmp_path, FakeConn([]))  # no vacancies yet
    with TestClient(create_app(services)) as client:
        response = client.get("/health")

    assert response.status_code == 200 and response.json()["status"] == "degraded"


def test_the_normalization_service_is_part_of_the_report(services, tmp_path) -> None:
    services.health = health_for(tmp_path, FakeConn([stats_row()]))
    services.normalizer = type("N", (), {"status": lambda self: {"index": "unavailable"}})()
    with TestClient(create_app(services)) as client:
        report = client.get("/health").json()

    assert report["checks"]["normalization"]["code"] == "index_unavailable"


def test_metrics_need_a_key(client) -> None:
    missing = client.get("/metrics")
    wrong = client.get("/metrics", headers={"X-API-Key": "nope"})

    assert (missing.status_code, wrong.status_code) == (401, 401)
    ok = client.get("/metrics", headers=HEADERS)
    assert ok.status_code == 200 and ok.headers["cache-control"] == "no-store"


def test_metrics_count_what_happened_without_any_secret(client) -> None:
    post(client)
    post(client, headers={})
    post(client, b"hello", name="cv.txt")

    snapshot = client.get("/metrics", headers=HEADERS)
    body = snapshot.json()

    assert body["analyses"]["total"] == 2 and body["analyses"]["ok"] == 1
    assert body["errors"] == {"parse:unsupported_type": 1}
    assert body["access_denied"] == {"missing_key": 1}
    assert body["stages"]["total"]["count"] == 1
    assert KEY not in snapshot.text and hash_key(KEY) not in snapshot.text


def test_metrics_say_invalid_key_for_a_wrong_key(client) -> None:
    client.get("/metrics", headers={"X-API-Key": "nope"})

    body = client.get("/metrics", headers=HEADERS).json()

    assert body["access_denied"] == {"invalid_key": 1}


# --- startup ---------------------------------------------------------------------------------


def test_without_services_they_are_built_at_startup(monkeypatch, services) -> None:
    built = []
    monkeypatch.setattr(app_module, "build_services", lambda: built.append(1) or services)

    with TestClient(create_app()) as client:
        assert client.get("/metrics", headers=HEADERS).status_code == 200

    assert built == [1]


def test_a_bad_configuration_stops_the_app_from_starting(monkeypatch) -> None:
    def refuse():
        raise ConfigError("JOBPDF_RATE_LIMIT_SECRET is missing")

    monkeypatch.setattr(app_module, "build_services", refuse)

    with pytest.raises(ConfigError), TestClient(create_app()):
        pass


def test_the_service_is_closed_at_shutdown(services) -> None:
    closed = []
    services.normalizer = type("N", (), {"close": lambda self: closed.append(1)})()
    with TestClient(create_app(services)):
        assert closed == []

    assert closed == [1]


def test_build_services_wires_the_real_pieces_from_the_environment(monkeypatch) -> None:
    fake_access = access()
    fake_normalizer = FakeNormalizer()
    monkeypatch.setattr(app_module, "build_access_control", lambda: fake_access)
    monkeypatch.setattr(app_module, "get_normalization_service", lambda: fake_normalizer)
    monkeypatch.setattr(app_module, "database_url", lambda: None)

    built = app_module.build_services()

    assert built.access is fake_access and built.normalizer is fake_normalizer
    assert list(built.vacancies()) == [] and type(built.scorer).__name__ == "LinearScorer"
    assert isinstance(built.recorder, MetricsRecorder)


def test_build_services_reads_vacancies_from_the_database_url(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "build_access_control", access)
    monkeypatch.setattr(app_module, "get_normalization_service", FakeNormalizer)
    monkeypatch.setattr(app_module, "database_url", lambda: "postgresql://x")
    monkeypatch.setattr(app_module, "postgres_loader", lambda url: lambda: VACANCIES[:1])

    assert list(app_module.build_services().vacancies()) == VACANCIES[:1]


# --- privacy ---------------------------------------------------------------------------------


def test_no_cv_text_key_or_address_reaches_the_logs_or_the_errors(client, caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        responses = [post(client), post(client, headers={}), post(client, b"hello", name="a.txt"),
                     post(client, b"%PDF-1.7 broken")]
    everything = caplog.text + "".join(r.text for r in responses if r.status_code != 200)

    assert not any(secret in everything for secret in SECRETS)
    assert KEY not in everything and "testclient" not in caplog.text.lower()
    assert responses[0].status_code == 200
