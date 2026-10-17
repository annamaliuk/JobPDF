"""The FastAPI app (JM-36): the JSON API in front of the CV pipeline.

    POST /v1/analyze   multipart ``file`` (PDF or DOCX) + ``X-API-Key`` -> ranked matches
    GET  /health       the health report (JM-34); 200 for ok/degraded, 503 for down
    GET  /metrics      the metrics snapshot (JM-34); needs a valid API key

Run it with ``uvicorn jobpdf.api.app:create_app --factory`` (see the README). Everything
else is wiring: the pipeline (JM-28), the response contract (JM-29), access control (JM-30),
the vacancy store (JM-32) and monitoring (JM-34) already do the work, and this module only
chains them and maps their failures to HTTP.

Order of checks on ``/v1/analyze``: API key, declared size, rate limit (all before a byte of
the body is read), then the body is read under a hard byte cap, then the pipeline runs.
Every refusal is counted in the metrics by code.

Decisions to review (all constants or arguments):

- Errors are ``{"error": {"code", "stage", "detail"}}``. ``code`` is stable; ``detail`` is a
  fixed English sentence and never holds CV text, a key or an address.
- At most ``MAX_CONCURRENT_ANALYSES`` analyses run at once; the next gets 503 ``busy`` with
  ``Retry-After``. Each analysis holds the CPU for seconds and costs LLM money.
- ``/health`` is public (a load balancer has no key). It reports counts, ages, versions and
  "set"/"missing", nothing secret. ``/metrics`` needs a key.
- A page of results re-runs the pipeline (nothing is stored between requests, JM-29), so
  asking for page 2 costs another upload and another rate-limit hit.
- A startup that cannot be configured (no keys secret, no database) fails at once instead of
  answering 500 on the first upload.
- Run uvicorn with ``--no-access-log``: its access log prints client addresses, which the
  rate limiter deliberately never stores.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException

from jobpdf.api.access import (
    BODY_OVERHEAD_BYTES,
    TOO_LARGE,
    AccessControl,
    AccessError,
    api_key_from_headers,
    build_access_control,
)
from jobpdf.api.response import AnalysisResponse, build_response
from jobpdf.api.vacancies import VacancyProvider, postgres_loader
from jobpdf.extraction.llm import ToolCaller
from jobpdf.monitoring.health import HealthReport, build_health, http_status
from jobpdf.monitoring.metrics import MetricsRecorder, MetricsSnapshot
from jobpdf.normalization.db import database_url
from jobpdf.normalization.embedder import TextEmbedder
from jobpdf.normalization.service import NormalizationService, get_normalization_service
from jobpdf.pipeline import MAX_UPLOAD_BYTES, PipelineError, PipelineResult, analyze_cv
from jobpdf.ranking.learned import select_scorer
from jobpdf.ranking.models import VacancyFeatures
from jobpdf.ranking.score import Scorer

log = logging.getLogger(__name__)

ANALYZE_PATH = "/v1/analyze"
FILE_FIELD = "file"
MAX_PAGE_SIZE = 200
MAX_CONCURRENT_ANALYSES = 4
BUSY_RETRY_AFTER_S = 5
MAX_BODY_BYTES = MAX_UPLOAD_BYTES + BODY_OVERHEAD_BYTES
NO_STORE = {"Cache-Control": "no-store"}

# Error codes of the app itself (the access and pipeline codes come from their modules).
INVALID_REQUEST = "invalid_request"
BUSY = "busy"
INTERNAL_ERROR = "internal_error"

# Pipeline failures (JM-28) -> status. Unlisted codes are the app's fault: 500.
PIPELINE_STATUS: dict[str, int] = {
    "unsupported_type": 415,
    "too_large": 413,
    "encrypted": 422,
    "corrupt": 422,
    "too_many_pages": 422,
    "no_text": 422,
    "llm_not_configured": 503,
    "llm_error": 502,
    "extraction_invalid": 502,
}

ERROR_DETAILS: dict[str, str] = {
    "missing_key": "Send your API key in the X-API-Key header.",
    "invalid_key": "The API key is not valid.",
    "rate_limited": "Too many uploads for now; try again after the time in Retry-After.",
    "too_large": "The file is larger than the 5 MB limit.",
    "limiter_unavailable": "Uploads are paused because the limits cannot be checked; retry soon.",
    "unsupported_type": "Only PDF and DOCX files are accepted.",
    "encrypted": "The file is password-protected.",
    "corrupt": "The file could not be opened.",
    "too_many_pages": "The file has too many pages.",
    "no_text": "No text could be found in the file.",
    "llm_not_configured": "Reading CVs is switched off on this server.",
    "llm_error": "The language model could not read this CV; try again later.",
    "extraction_invalid": "The language model's answer could not be used; try again later.",
    INVALID_REQUEST: "Send a multipart form with the CV in the 'file' field.",
    BUSY: "The server is busy with other CVs; retry soon.",
    INTERNAL_ERROR: "Something went wrong on the server.",
}
GENERIC_DETAIL = "The request failed."


class BodyTooLarge(Exception):
    """The request body went past ``MAX_BODY_BYTES`` while it was being read."""


class BodyLimit:
    """Counts the bytes of every request body as it is read and stops at the cap.

    The declared ``Content-Length`` is checked by access control; this covers the bodies that
    declare nothing (chunked) or lie, so an upload can never fill memory or disk.
    """

    def __init__(self, app: Any, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        seen = 0

        async def limited() -> Any:
            nonlocal seen
            message = await receive()
            seen += len(message.get("body", b""))  # only request messages carry a body
            if seen > self.max_bytes:
                raise BodyTooLarge
            return message

        await self.app(scope, limited, send)


@dataclass
class Services:
    """Everything the endpoints use. ``build_services`` makes the real ones; tests pass fakes."""

    access: AccessControl
    vacancies: Callable[[], Sequence[VacancyFeatures]]
    recorder: MetricsRecorder = field(default_factory=MetricsRecorder)
    normalizer: NormalizationService | None = None
    scorer: Scorer | None = None
    caller: ToolCaller | None = None
    embedder: TextEmbedder | None = None
    analyze: Callable[..., PipelineResult] = analyze_cv
    health: Callable[..., HealthReport] = build_health
    max_concurrent: int = MAX_CONCURRENT_ANALYSES
    slots: threading.BoundedSemaphore = field(init=False)

    def __post_init__(self) -> None:
        self.slots = threading.BoundedSemaphore(self.max_concurrent)


def build_services() -> Services:
    """The real services, from the environment. Slow (it loads the embedding model) and
    strict: a missing key secret or an unreachable database raises here, at startup."""
    access = build_access_control()
    normalizer = get_normalization_service()
    url = database_url()
    provider = VacancyProvider(postgres_loader(url) if url else (lambda: ()))
    return Services(access=access, vacancies=provider.get, normalizer=normalizer,
                    scorer=select_scorer())


def create_app(services: Services | None = None) -> FastAPI:
    """The app. Without ``services`` they are built at startup (``uvicorn --factory``)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.services is None:
            app.state.services = await run_in_threadpool(build_services)
        yield
        close = getattr(app.state.services.normalizer, "close", None)
        if callable(close):
            close()

    app = FastAPI(title="JobPDF", version="1", lifespan=lifespan)
    app.state.services = services
    app.add_middleware(BodyLimit)

    @app.get("/health")
    def health(request: Request) -> JSONResponse:
        svc: Services = request.app.state.services
        report = svc.health(normalizer=svc.normalizer, api_keys=svc.access.keys)
        return JSONResponse(report.model_dump(mode="json"), status_code=http_status(report),
                            headers=NO_STORE)

    @app.get("/metrics", response_model=MetricsSnapshot)
    def metrics(request: Request) -> JSONResponse:
        svc: Services = request.app.state.services
        try:
            svc.access.keys.authenticate(api_key_from_headers(request.headers))
        except AccessError as exc:
            svc.recorder.record_access_denied(exc.code)
            return _access_error(exc)
        return JSONResponse(svc.recorder.snapshot().model_dump(mode="json"), headers=NO_STORE)

    @app.post(ANALYZE_PATH, response_model=AnalysisResponse)
    async def analyze(
        request: Request,
        page: int = Query(1, ge=1),
        page_size: int | None = Query(None, ge=1, le=MAX_PAGE_SIZE),
    ) -> JSONResponse:
        svc: Services = request.app.state.services
        try:
            svc.access.authorize_api(request.headers, _declared_length(request))
            content, filename = await _read_form(request, svc)
        except AccessError as exc:
            svc.recorder.record_access_denied(exc.code)
            return _access_error(exc)
        except BadRequest:
            return _error(INVALID_REQUEST, 400)

        if not svc.slots.acquire(blocking=False):
            svc.recorder.record_access_denied(BUSY)
            return _error(BUSY, 503, retry_after=BUSY_RETRY_AFTER_S)
        try:
            result = await run_in_threadpool(_run, svc, content, filename)
        except PipelineError as exc:
            return _error(exc.code, PIPELINE_STATUS.get(exc.code, 500), stage=exc.stage)
        except Exception as exc:  # already counted by the recorder; never echo the text
            log.error("analysis failed unexpectedly (%s)", type(exc).__name__)
            return _error(INTERNAL_ERROR, 500)
        finally:
            svc.slots.release()
        body = build_response(result, page=page, page_size=page_size)
        return JSONResponse(body.model_dump(mode="json"), headers=NO_STORE)

    from jobpdf.web import register_web  # here, not at the top: the web module imports this one

    register_web(app)
    return app


class BadRequest(Exception):
    """The request is not a multipart form with a file."""


def _declared_length(request: Request) -> int | None:
    raw = request.headers.get("content-length")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None  # the byte cap still applies while the body is read


async def _read_form(request: Request, svc: Services) -> tuple[bytes, str]:
    """The uploaded file's bytes and name; closes the form (and its temporary file) at once."""
    try:
        form = await request.form()
    except BodyTooLarge:
        raise AccessError(TOO_LARGE) from None
    except HTTPException:  # malformed multipart
        raise BadRequest from None
    try:
        upload = form.get(FILE_FIELD)
        if not isinstance(upload, UploadFile):
            raise BadRequest
        return svc.access.read_upload(upload.file), upload.filename or ""
    finally:
        await form.close()


def _run(svc: Services, content: bytes, filename: str) -> PipelineResult:
    return svc.recorder.observe_analysis(lambda: svc.analyze(
        content, filename, svc.vacancies(), caller=svc.caller, normalizer=svc.normalizer,
        embedder=svc.embedder, scorer=svc.scorer))


def _error(code: str, status: int, *, stage: str | None = None,
           retry_after: int | None = None) -> JSONResponse:
    headers = dict(NO_STORE)
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    body = {"error": {"code": code, "stage": stage,
                      "detail": ERROR_DETAILS.get(code, GENERIC_DETAIL)}}
    return JSONResponse(body, status_code=status, headers=headers)


def _access_error(exc: AccessError) -> JSONResponse:
    return _error(exc.code, exc.status, retry_after=exc.retry_after)
