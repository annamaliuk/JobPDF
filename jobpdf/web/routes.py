"""The web page in front of the same pipeline as the JSON API (JM-43).

    GET  /         the upload page
    POST /report   multipart ``file`` -> the report as an HTML page (one request, nothing stored);
                   with ``Accept: text/event-stream`` the same request streams progress steps
                   and ends with the page (JM-44, see ``progress.py``)
    GET  /static/  one stylesheet and one small script

There is no login and no key in the browser. A key in a page's script is not a secret, so this
path is the one the access rules (JM-30) made for a web form: the 5 MB size check and a limit
per client address (the address is hashed, never stored). It shares everything else with
``/v1/analyze``: the byte cap on the body, the limit on analyses running at once, the
pipeline, and the metrics, so the web page cannot be a way around any of them.

Behind the Caddy proxy the client address comes from ``X-Forwarded-For``, which uvicorn
trusts only from the addresses in ``FORWARDED_ALLOW_IPS`` (docker-compose.yml sets it, because
the app is reachable only through the proxy). Without that every visitor would share one
address, the proxy's, and one visitor could use up everyone's limit.

Pages are never cached and cannot load anything from another site (the Content-Security-Policy
below), and they do not send a referrer. Pages and errors show fixed sentences only; a CV's
text appears in the report for the person who uploaded it and nowhere else (not in a log, not in
an error).
"""

from __future__ import annotations

import ipaddress
import logging
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from markupsafe import Markup, escape
from starlette.staticfiles import StaticFiles

from jobpdf.api import app as api  # shares its form reading, limits and pipeline run
from jobpdf.api.access import AccessError
from jobpdf.api.response import build_response
from jobpdf.normalization import skill_names
from jobpdf.pipeline import MAX_UPLOAD_BYTES, PipelineError, PipelineResult
from jobpdf.web import progress
from jobpdf.web.view import build_report

log = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"
REPORT_PATH = "/report"
UNKNOWN_CLIENT = "0.0.0.0"  # one shared bucket for a request with no usable address

CONTENT_SECURITY_POLICY = (
    "default-src 'none'; style-src 'self'; script-src 'self'; img-src 'self'; "
    "connect-src 'none'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
)
# The upload page alone may make requests, and only to itself: its script posts the CV and reads
# the progress stream (JM-44). Every other page keeps ``connect-src 'none'``.
INDEX_CONTENT_SECURITY_POLICY = CONTENT_SECURITY_POLICY.replace(
    "connect-src 'none'", "connect-src 'self'")
PAGE_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}

# What the page says when a request fails. The API's sentences (ERROR_DETAILS) are for
# programs; these few are reworded for a person.
WEB_DETAILS: dict[str, str] = {
    api.INVALID_REQUEST: "Choose a PDF or Word file and try again.",
    "rate_limited": "You have uploaded a lot of CVs for now. Try again in {wait}.",
}

_templates = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(["html"]),
    undefined=StrictUndefined,  # a typo in a template fails a test instead of printing nothing
    trim_blocks=True,
    lstrip_blocks=True,
)


def marked(value: str) -> str | Markup:
    """A report sentence for the page. When it holds Ukrainian skill names (a ``view.Line``), each
    one is wrapped in ``<span lang="uk">`` so a screen reader pronounces it right; everything
    else is escaped as usual. English output has no names to mark and passes through unchanged."""
    unique = sorted({n for n in getattr(value, "uk", ()) if n and n.strip()},
                    key=len, reverse=True)
    if not unique:
        return value
    pieces = re.split("(" + "|".join(re.escape(n) for n in unique) + ")", str(value))
    return Markup("").join(  # re.split puts each match at an odd position
        Markup('<span lang="uk">{}</span>').format(piece) if i % 2 else escape(piece)
        for i, piece in enumerate(pieces))


_templates.filters["marked"] = marked


def register_web(app: FastAPI) -> None:
    """Add the pages to ``app``."""
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index(request: Request) -> HTMLResponse:
        return _page("index.html",
                     headers={"Content-Security-Policy": INDEX_CONTENT_SECURITY_POLICY},
                     max_bytes=MAX_UPLOAD_BYTES, field=api.FILE_FIELD, path=REPORT_PATH,
                     steps=progress.STEPS, language_field=skill_names.LANGUAGE_FIELD,
                     language=skill_names.preferred_language(request.headers.get("accept-language")))

    @app.post(REPORT_PATH, response_class=HTMLResponse, include_in_schema=False)
    async def report(request: Request) -> HTMLResponse:
        svc: api.Services = request.app.state.services
        try:
            svc.access.authorize_web(_client_ip(request), api._declared_length(request))
            content, filename, fields = await api.read_form(
                request, svc, (skill_names.LANGUAGE_FIELD,))
        except AccessError as exc:
            svc.recorder.record_access_denied(exc.code)
            return _error_page(exc.code, exc.status, retry_after=exc.retry_after)
        except api.BadRequest:
            return _error_page(api.INVALID_REQUEST, 400)

        language = skill_names.choose_language(fields[skill_names.LANGUAGE_FIELD])
        if not svc.slots.acquire(blocking=False):
            svc.recorder.record_access_denied(api.BUSY)
            return _error_page(api.BUSY, 503, retry_after=api.BUSY_RETRY_AFTER_S)
        if progress.wants_stream(request.headers.get("accept")):
            return _stream_report(svc, content, filename, language)  # now owns the slot
        try:
            result = await run_in_threadpool(api._run, svc, content, filename)
        except PipelineError as exc:
            return _error_page(exc.code, api.PIPELINE_STATUS.get(exc.code, 500))
        except Exception as exc:  # already counted by the recorder; never echo the text
            log.error("analysis failed unexpectedly (%s)", type(exc).__name__)
            return _error_page(api.INTERNAL_ERROR, 500)
        finally:
            svc.slots.release()
        try:
            # In a thread: the first Ukrainian report reads the names from the database.
            html = await run_in_threadpool(_report_html, svc, result, language)
        except Exception as exc:  # a page that cannot be drawn must not echo the CV either
            log.error("report page failed (%s)", type(exc).__name__)
            return _error_page(api.INTERNAL_ERROR, 500)
        return HTMLResponse(html, headers=PAGE_HEADERS)


def _report_html(svc: api.Services, result: PipelineResult, language: str) -> str:
    """The report page. Skill names are Ukrainian only when asked for; if the names cannot be
    loaded the page is made in English (``skill_names`` logs why)."""
    names: Mapping[str, str] | None = None
    if language == skill_names.UKRAINIAN:
        try:
            names = svc.skill_names()
        except Exception as exc:  # never a reason to lose the report
            log.warning("ukrainian skill names unavailable (%s)", type(exc).__name__)
    return _render("report.html", report=build_report(build_response(result), names))


def _stream_report(svc: api.Services, content: bytes, filename: str,
                   language: str) -> StreamingResponse:
    """The report as a stream of progress steps. The analysis slot taken by the caller goes to
    the worker thread, which gives it back when the analysis ends, however the client behaves."""

    def work(on_stage) -> progress.Outcome:
        try:
            result = api._run(svc, content, filename, on_stage=on_stage)
        except PipelineError as exc:
            return _failed(exc.code, api.PIPELINE_STATUS.get(exc.code, 500))
        except Exception as exc:  # already counted by the recorder; never echo the text
            log.error("analysis failed unexpectedly (%s)", type(exc).__name__)
            return _failed(api.INTERNAL_ERROR, 500)
        on_stage(progress.RENDER_STAGE)
        try:
            html = _report_html(svc, result, language)
        except Exception as exc:  # a page that cannot be drawn must not echo the CV either
            log.error("report page failed (%s)", type(exc).__name__)
            return _failed(api.INTERNAL_ERROR, 500)
        return progress.Outcome("done", {"status": 200, "html": html})

    try:
        stream = progress.AnalysisStream(work, svc.slots.release)
    except Exception:
        svc.slots.release()  # the worker never started, so nobody else will
        raise
    return StreamingResponse(stream.events(), media_type=progress.STREAM_MEDIA_TYPE,
                             headers={**PAGE_HEADERS, "X-Accel-Buffering": "no"})


def _failed(code: str, status: int) -> progress.Outcome:
    detail = _error_detail(code, None)
    html = _render("error.html", detail=detail, code=code)
    return progress.Outcome("failed", {"status": status, "html": html})


def _client_ip(request: Request) -> str:
    """The address the limiter counts. A request without a usable address (no peer, a unix
    socket, a name that is not an IP) is counted together with all others like it, so it is
    limited too instead of failing the page."""
    host = request.client.host if request.client else ""
    try:
        return str(ipaddress.ip_address(host.strip()))
    except ValueError:
        return UNKNOWN_CLIENT


def _render(template: str, **context: Any) -> str:
    return _templates.get_template(template).render(**context)


def _page(template: str, status: int = 200, headers: dict[str, str] | None = None,
          **context: Any) -> HTMLResponse:
    html = _render(template, **context)
    return HTMLResponse(html, status_code=status, headers={**PAGE_HEADERS, **(headers or {})})


def _error_detail(code: str, retry_after: int | None) -> str:
    detail = WEB_DETAILS.get(code) or api.ERROR_DETAILS.get(code, api.GENERIC_DETAIL)
    return detail.replace("{wait}", _wait(retry_after) if retry_after else "a while")


def _error_page(code: str, status: int, *, retry_after: int | None = None) -> HTMLResponse:
    headers = {}
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return _page("error.html", status=status, headers=headers,
                 detail=_error_detail(code, retry_after), code=code)


def _wait(seconds: int) -> str:
    minutes = math.ceil(max(seconds, 1) / 60)
    return "1 minute" if minutes == 1 else f"{minutes} minutes"
