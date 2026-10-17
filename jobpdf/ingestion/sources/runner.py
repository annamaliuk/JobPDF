"""Fetch one job source politely: cached, rationed, no retries, stale fallback.

The same safeguards as ``jobpdf.ingestion.remotive`` (JM-31), generalised to sources whose
results span several pages:

- A run younger than ``policy.max_cache_age_s`` is served from the cache; no request.
- A live run is allowed only if the daily budget and the minimum interval allow it. Each
  request is written to a ledger *before* it is sent. There are no retries.
- A run asks for at most ``policy.max_pages`` pages, pausing ``policy.pause_s`` between them.
  If a later page fails, the pages already fetched are kept and the result carries the
  ``partial`` warning; if the first page fails, the newest cache is served as ``stale``,
  or ``FetchError(code)`` is raised when there is none.
- A run is cached as received (``data/cache/<source>/``, git-ignored) only if it holds at
  least one usable posting, so a bad response never replaces a good cache.

API keys appear in requests (a query string or the URL path). A ``Request`` never shows its
URL, and nothing here logs a URL, a body, an exception message, a title or a description:
logs carry the source name, counts and codes.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobpdf.ingestion.sources.models import (
    HTTP_ERROR,
    INVALID_RESPONSE,
    NETWORK_ERROR,
    NO_POSTINGS,
    NOT_CONFIGURED,
    PARTIAL,
    QUOTA_EXHAUSTED,
    RESPONSE_TOO_LARGE,
    SKIPPED_POSTINGS,
    STALE_CACHE,
    TOO_SOON,
    FeedError,
    FetchError,
    ParsedPage,
    Request,
    SourceAdapter,
    SourcePosting,
    SourceResult,
)

log = logging.getLogger(__name__)

USER_AGENT = "JobPDF/0.1 (student project; polite, cached, rate-limited)"
CACHE_ENV = "JOBPDF_CACHE_DIR"
LEDGER_NAME = "requests.json"
CACHE_SUFFIX = ".json"
CACHE_STAMP = "%Y%m%dT%H%M%SZ"
CACHE_FORMAT = 1
DAY_S = 24 * 3600
REPO_ROOT = Path(__file__).resolve().parents[3]
# The scheduled job (dags/job_sources_fetch.py) runs every 8 hours; staying under that keeps
# a run that starts a little early from being served the previous run's cache.
SCHEDULED_MAX_AGE_S = 7 * 3600

Transport = Callable[[Request, float, int], tuple[int, bytes]]


def cache_dir(source: str, base: Path | None = None) -> Path:
    if base is None:
        base = Path(os.environ.get(CACHE_ENV) or REPO_ROOT / "data" / "cache")
    return base / source


def urllib_transport(request: Request, timeout: float, max_bytes: int) -> tuple[int, bytes]:
    """Plain stdlib HTTP. Returns up to ``max_bytes + 1`` bytes so callers can detect an
    oversized body; HTTP errors become ``(status, b"")``."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json", **request.headers}
    prepared = urllib.request.Request(  # noqa: S310
        request.url, data=request.body, headers=headers, method=request.method)
    try:
        with urllib.request.urlopen(prepared, timeout=timeout) as response:  # noqa: S310
            return response.status, response.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:
        return exc.code, b""


# --- cache and ledger --------------------------------------------------------------


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _cache_files(folder: Path) -> list[tuple[datetime, Path]]:
    found = []
    for path in folder.glob(f"*{CACHE_SUFFIX}"):
        try:
            moment = datetime.strptime(path.stem, CACHE_STAMP).replace(tzinfo=timezone.utc)
        except ValueError:  # the ledger and stray files
            continue
        found.append((moment, path))
    return sorted(found, reverse=True)


def _read_ledger(folder: Path, now: float) -> list[float]:
    """Times of requests in the last 24 hours. An unreadable ledger counts as one request
    just made, so the guard errs towards waiting."""
    try:
        times = json.loads((folder / LEDGER_NAME).read_text(encoding="utf-8"))
        return [float(t) for t in times if now - float(t) < DAY_S]
    except FileNotFoundError:
        return []
    except (OSError, ValueError, TypeError):
        log.warning("request ledger unreadable; assuming a request was just made")
        return [now]


def _write_ledger(folder: Path, times: list[float]) -> None:
    _write_atomic(folder / LEDGER_NAME, json.dumps(times).encode("utf-8"))


class _Run:
    """The pages of one run, as decoded JSON, and what they parse to."""

    def __init__(self, pages: list[Any], complete: bool) -> None:
        self.pages = pages
        self.complete = complete


_Parsed = tuple[list[SourcePosting], int, int]  # postings, skipped, filtered


def _parse_run(adapter: SourceAdapter, run: _Run) -> _Parsed:
    """Postings of a run, deduplicated by id (the first wins), with skipped and filtered
    counts. Raises FeedError when the first page is not a feed or nothing is usable."""
    postings: list[SourcePosting] = []
    seen: set[str] = set()
    skipped = filtered = raw = 0
    for index, page in enumerate(run.pages):
        try:
            parsed: ParsedPage = adapter.parse_page(page)
        except FeedError:
            if index == 0:
                raise
            break
        raw += parsed.raw_count
        skipped += parsed.skipped
        filtered += parsed.filtered
        for posting in parsed.postings:
            if posting.source_id in seen:
                skipped += 1
                continue
            seen.add(posting.source_id)
            postings.append(posting)
    if not postings:
        raise FeedError(NO_POSTINGS if raw == 0 or filtered else INVALID_RESPONSE)
    return postings, skipped, filtered


def _load_cache(adapter: SourceAdapter, folder: Path
                ) -> tuple[datetime, _Run, _Parsed] | None:
    for moment, path in _cache_files(folder):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["format"] != CACHE_FORMAT:
                raise ValueError
            run = _Run(data["pages"], bool(data["complete"]))
            return moment, run, _parse_run(adapter, run)
        except (OSError, ValueError, KeyError, TypeError):  # FeedError is a ValueError
            log.warning("%s: ignoring an unusable cache file", adapter.name)
    return None


# --- the live run ------------------------------------------------------------------


def _live_run(adapter: SourceAdapter, folder: Path, clock: Callable[[], float],
              transport: Transport, sleep: Callable[[float], None]
              ) -> tuple[_Run, list[str]] | str:
    """The pages of a live run (with warning codes), or the code of why there are none."""
    policy = adapter.policy
    now = clock()
    times = _read_ledger(folder, now)
    if len(times) >= policy.max_requests_per_day:
        return QUOTA_EXHAUSTED
    if times and now - max(times) < policy.min_interval_s:
        return TOO_SOON

    pages: list[Any] = []
    problem = ""
    for _ in range(policy.max_pages):
        request = adapter.next_request(pages)
        if request is None:
            return _Run(pages, True), []
        if pages:
            sleep(policy.pause_s)
        now = clock()
        if len(times) >= policy.max_requests_per_day:
            problem = QUOTA_EXHAUSTED
            break
        times = [*times, now]
        _write_ledger(folder, times)  # before sending: a failed request still counts
        outcome = _fetch_page(adapter, request, transport)
        if isinstance(outcome, str):
            problem = outcome
            break
        pages.append(outcome)
    else:  # max_pages reached: complete only if the source had nothing more to ask for
        if adapter.next_request(pages) is None:
            return _Run(pages, True), []
        problem = ""
    if not pages:
        return problem
    return _Run(pages, False), [PARTIAL, *([problem] if problem else [])]


def _fetch_page(adapter: SourceAdapter, request: Request, transport: Transport
                ) -> Any | str:
    policy = adapter.policy
    try:
        status, body = transport(request, policy.timeout_s, policy.max_response_bytes)
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        log.warning("%s: request failed (%s)", adapter.name, type(exc).__name__)
        return NETWORK_ERROR
    if status != 200:
        log.warning("%s: HTTP status %d", adapter.name, status)
        return HTTP_ERROR
    if len(body) > policy.max_response_bytes:
        return RESPONSE_TOO_LARGE
    try:
        page = json.loads(body)
    except (ValueError, UnicodeError):
        return INVALID_RESPONSE
    try:
        adapter.parse_page(page)  # a page that is not this source's feed is not kept
    except FeedError as exc:
        return exc.code
    return page


def fetch_source(
    adapter: SourceAdapter,
    *,
    base: Path | None = None,
    max_age_s: float | None = None,
    transport: Transport | None = None,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> SourceResult:
    """A source's postings: from the cache when fresh, else one guarded live run.

    Raises FetchError when there is nothing to serve (code ``not_configured`` when a key is
    missing, otherwise the reason the live run failed and that there is no cache).
    """
    if not adapter.is_configured():
        raise FetchError(NOT_CONFIGURED)
    folder = cache_dir(adapter.name, base)
    limit = adapter.policy.max_cache_age_s if max_age_s is None else max_age_s
    now = (clock or time.time)()
    cached = _load_cache(adapter, folder)

    if cached is not None and now - cached[0].timestamp() <= limit:
        return _result(adapter, cached, from_cache=True, stale=False,
                       warnings=_cached_warnings(cached))

    outcome = _live_run(adapter, folder, clock or time.time, transport or urllib_transport,
                        sleep or time.sleep)
    if not isinstance(outcome, str):
        run, warnings = outcome
        try:
            parsed = _parse_run(adapter, run)
        except FeedError as exc:
            outcome = exc.code
        else:
            stamp = datetime.fromtimestamp(now, tz=timezone.utc)
            payload = {"format": CACHE_FORMAT, "complete": run.complete, "pages": run.pages}
            _write_atomic(folder / f"{stamp.strftime(CACHE_STAMP)}{CACHE_SUFFIX}",
                          json.dumps(payload).encode("utf-8"))
            for _, old in _cache_files(folder)[adapter.policy.keep:]:
                old.unlink(missing_ok=True)
            log.info("%s: fetched %d postings in %d pages (%d skipped, %d filtered)",
                     adapter.name, len(parsed[0]), len(run.pages), parsed[1], parsed[2])
            return _result(adapter, (stamp, run, parsed), from_cache=False, stale=False,
                           warnings=warnings)

    if cached is None:
        log.warning("%s: live fetch failed (%s) and there is no cache", adapter.name, outcome)
        raise FetchError(outcome)
    log.warning("%s: serving older data: live fetch failed (%s)", adapter.name, outcome)
    return _result(adapter, cached, from_cache=True, stale=True,
                   warnings=[STALE_CACHE, outcome, *_cached_warnings(cached)])


def _cached_warnings(cached: tuple[datetime, _Run, Any]) -> list[str]:
    return [] if cached[1].complete else [PARTIAL]


def _result(adapter: SourceAdapter, found: tuple[datetime, _Run, _Parsed],
            *, from_cache: bool, stale: bool, warnings: list[str]) -> SourceResult:
    fetched_at, run, (postings, skipped, filtered) = found
    if skipped:
        warnings = [*warnings, SKIPPED_POSTINGS]
    return SourceResult(
        source=adapter.name, postings=postings, fetched_at=fetched_at, from_cache=from_cache,
        stale=stale, pages=len(run.pages), skipped=skipped, filtered=filtered,
        warnings=list(dict.fromkeys(warnings)),
    )


def run_source(adapter: SourceAdapter, **kwargs: Any) -> dict[str, Any]:
    """What a scheduled task runs for one source: a small summary, never the postings.
    A source that is not configured is reported as skipped rather than failing the task."""
    kwargs.setdefault("max_age_s", SCHEDULED_MAX_AGE_S)
    try:
        result = fetch_source(adapter, **kwargs)
    except FetchError as exc:
        if exc.code == NOT_CONFIGURED:
            return {"source": adapter.name, "status": "skipped", "reason": NOT_CONFIGURED}
        raise
    return {
        "source": adapter.name, "status": "ok", "count": len(result.postings),
        "pages": result.pages, "skipped": result.skipped, "filtered": result.filtered,
        "from_cache": result.from_cache, "stale": result.stale,
        "fetched_at": result.fetched_at.isoformat(), "warnings": result.warnings,
    }
