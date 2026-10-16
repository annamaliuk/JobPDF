"""Fetch Remotive postings (JM-31): one polite request, cached, never in a loop.

Remotive's API is free but rationed: at most 4 requests a day are advised, more than 2 a
minute gets you blocked, and its jobs are delayed by 24 hours anyway. So this module
treats a request as expensive:

- ``fetch_postings()`` serves the cache when it is younger than ``MAX_CACHE_AGE_S`` and
  makes no request at all.
- A live request is made only if the guard allows it (``MAX_REQUESTS_PER_DAY`` in any 24
  hours, ``MIN_INTERVAL_S`` between two). It is written to a ledger *before* it is sent,
  because a failed request still counts against Remotive's limit. There are no retries.
- Whatever goes wrong (guard, network, HTTP status, unusable body), the newest cache is
  served and marked ``stale``; with no cache at all ``FetchError(code)`` is raised.
- Every new response is stored as received under ``data/cache/remotive/`` (git-ignored).
  An unusable or empty response is never stored, so it cannot replace a good cache.

What it does not do: filter "new" postings (there is no since-parameter, so JM-32/33
compare ids), clean descriptions or find skills (JM-48), or schedule itself
(``dags/remotive_fetch.py``).

Attribution (Remotive's condition for API access): every posting keeps its ``url``, and
``SOURCE`` names the provider. Postings must not be passed on to third-party job sites.

Logs carry counts and codes only: no URL, title or description.
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

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

log = logging.getLogger(__name__)

SOURCE = "remotive"
API_URL = "https://remotive.com/api/remote-jobs"
USER_AGENT = "JobPDF/0.1 (student project; respects Remotive's rate limit)"
REQUEST_TIMEOUT_S = 30.0
MAX_RESPONSE_BYTES = 64 * 1024 * 1024

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ENV = "JOBPDF_CACHE_DIR"  # replaces data/cache (e.g. a mounted volume in Docker)
LEDGER_NAME = "requests.json"
CACHE_SUFFIX = ".json"
CACHE_STAMP = "%Y%m%dT%H%M%SZ"
CACHE_KEEP = 8  # newest responses kept (two days at 4 a day)

MAX_CACHE_AGE_S = 6 * 3600  # how long ad-hoc callers reuse a response
# The scheduled job (dags/remotive_fetch.py) runs every 8 hours. Its cache limit is under
# that interval, so a run starting a little early still refreshes; otherwise refreshes would
# quietly drop to every other run.
SCHEDULED_MAX_AGE_S = 7 * 3600
MAX_REQUESTS_PER_DAY = 4
DAY_S = 24 * 3600
MIN_INTERVAL_S = 30.0  # Remotive blocks more than 2 requests a minute

# Why a live fetch did not (or could not) happen. Used as FetchError.code and as a warning.
QUOTA_EXHAUSTED = "quota_exhausted"
TOO_SOON = "too_soon"
NETWORK_ERROR = "network_error"
HTTP_ERROR = "http_error"
RESPONSE_TOO_LARGE = "response_too_large"
INVALID_RESPONSE = "invalid_response"
NO_POSTINGS = "no_postings"
NO_CACHE = "no_cache"
STALE_CACHE = "stale_cache"  # warning only: an older cache was served instead of fresh data
SKIPPED_POSTINGS = "skipped_postings"  # warning only: some postings were unusable


class FetchError(RuntimeError):
    """No postings could be obtained and there is no cache to fall back on."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class FeedError(ValueError):
    """The response body is not a usable Remotive feed (``code`` says why)."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class RemotivePosting(BaseModel):
    """One job as Remotive sends it. Unknown fields are ignored, so an API addition does
    not break fetching; ``id``, ``url`` and ``title`` are the only required ones."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: int
    url: str
    title: str
    company_name: str | None = None
    category: str | None = None
    tags: list[str] = []
    job_type: str | None = None
    publication_date: datetime | None = None  # timezone-aware; Remotive's naive times are UTC
    candidate_required_location: str | None = None
    salary: str | None = None
    description: str | None = None  # HTML as received

    @field_validator("url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("url must be http(s)")
        return value

    @field_validator("title")
    @classmethod
    def _title_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("title is empty")
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def _tags(cls, value: Any) -> Any:
        return [] if value is None else value

    @field_validator("publication_date")
    @classmethod
    def _utc(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class ParsedFeed(BaseModel):
    model_config = ConfigDict(frozen=True)

    postings: list[RemotivePosting]
    skipped: int  # postings dropped because they were malformed
    reported_count: int | None  # the feed's own "job-count"
    legal_notice: str | None


class FetchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    postings: list[RemotivePosting]
    fetched_at: datetime  # when the data was fetched from Remotive (UTC)
    from_cache: bool  # no request was made this time
    stale: bool  # a live fetch was needed but failed; this is older data
    skipped: int
    reported_count: int | None
    legal_notice: str | None
    warnings: list[str]  # codes only


Transport = Callable[[str, float], tuple[int, bytes]]


# --- parsing -----------------------------------------------------------------------


def parse_feed(body: bytes) -> ParsedFeed:
    """Postings from a raw response. One malformed posting is skipped and counted; a body
    that is not JSON, has no ``jobs`` list or has no usable posting raises FeedError."""
    try:
        data = json.loads(body)
    except (ValueError, UnicodeError):
        raise FeedError(INVALID_RESPONSE) from None
    jobs = data.get("jobs") if isinstance(data, dict) else None
    if not isinstance(jobs, list):
        raise FeedError(INVALID_RESPONSE)
    postings: list[RemotivePosting] = []
    seen: set[int] = set()
    skipped = 0
    for job in jobs:
        try:
            posting = RemotivePosting.model_validate(job)
        except ValidationError:
            skipped += 1
            continue
        if posting.id in seen:  # the same id twice in one feed: the first one wins
            skipped += 1
            continue
        seen.add(posting.id)
        postings.append(posting)
    if not postings:
        raise FeedError(NO_POSTINGS if not jobs else INVALID_RESPONSE)
    count = data.get("job-count")
    notice = data.get("0-legal-notice")
    return ParsedFeed(
        postings=postings, skipped=skipped,
        reported_count=count if isinstance(count, int) and not isinstance(count, bool) else None,
        legal_notice=notice if isinstance(notice, str) else None,
    )


# --- cache and ledger --------------------------------------------------------------


def cache_dir(base: Path | None = None) -> Path:
    """``<data/cache or $JOBPDF_CACHE_DIR>/remotive``."""
    if base is None:
        base = Path(os.environ.get(CACHE_ENV) or REPO_ROOT / "data" / "cache")
    return base / SOURCE


def _stamp(moment: datetime) -> str:
    return moment.strftime(CACHE_STAMP)


def _cache_files(folder: Path) -> list[tuple[datetime, Path]]:
    """Cached responses, newest first, by the time in their file name."""
    found = []
    for path in folder.glob(f"*{CACHE_SUFFIX}"):
        try:  # the ledger and stray files have no timestamp in their name, so they never match
            moment = datetime.strptime(path.stem, CACHE_STAMP).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        found.append((moment, path))
    return sorted(found, reverse=True)


def _latest_cache(folder: Path) -> tuple[datetime, ParsedFeed] | None:
    for moment, path in _cache_files(folder):
        try:
            return moment, parse_feed(path.read_bytes())
        except (OSError, FeedError):
            log.warning("ignoring an unusable cache file")
    return None


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _read_ledger(folder: Path, now: float) -> list[float]:
    """Times of recent live requests. An unreadable ledger counts as one request just made,
    so the guard errs on the side of waiting rather than of overspending."""
    path = folder / LEDGER_NAME
    try:
        times = json.loads(path.read_text(encoding="utf-8"))
        return [float(t) for t in times if now - float(t) < DAY_S]
    except FileNotFoundError:
        return []
    except (OSError, ValueError, TypeError):
        log.warning("request ledger unreadable; assuming a request was just made")
        return [now]


def _guard(times: list[float], now: float) -> str | None:
    """Why a live request is not allowed right now, or None."""
    if len(times) >= MAX_REQUESTS_PER_DAY:
        return QUOTA_EXHAUSTED
    if times and now - max(times) < MIN_INTERVAL_S:
        return TOO_SOON
    return None


# --- fetching ----------------------------------------------------------------------


def _urllib_transport(url: str, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status, response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        return exc.code, b""


def fetch_postings(
    *,
    base: Path | None = None,
    max_age_s: float = MAX_CACHE_AGE_S,
    transport: Transport | None = None,
    clock: Callable[[], float] | None = None,
) -> FetchResult:
    """Remotive's postings: from the cache when fresh, else one guarded live request.

    Raises FetchError only when there is nothing to serve (no cache and no live data).
    """
    folder = cache_dir(base)
    now = (clock or time.time)()  # looked up per call, so tests can replace time.time
    cached = _latest_cache(folder)

    if cached is not None and now - cached[0].timestamp() <= max_age_s:
        return _result(cached, from_cache=True, stale=False, warnings=[])

    reason = _live_fetch(folder, now, transport or _urllib_transport)
    if isinstance(reason, ParsedFeed):
        fetched_at = datetime.fromtimestamp(now, tz=timezone.utc)
        return _result((fetched_at, reason), from_cache=False, stale=False, warnings=[])

    if cached is None:
        log.warning("no Remotive data: live fetch failed (%s) and there is no cache", reason)
        raise FetchError(reason)
    log.warning("serving older Remotive data: live fetch failed (%s)", reason)
    return _result(cached, from_cache=True, stale=True, warnings=[STALE_CACHE, reason])


def _live_fetch(folder: Path, now: float, transport: Transport) -> ParsedFeed | str:
    """A parsed feed (already cached on disk), or the code of why there is none."""
    times = _read_ledger(folder, now)
    blocked = _guard(times, now)
    if blocked is not None:
        return blocked
    _write_atomic(folder / LEDGER_NAME, json.dumps([*times, now]).encode("utf-8"))

    try:
        status, body = transport(API_URL, REQUEST_TIMEOUT_S)
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        log.warning("Remotive request failed (%s)", type(exc).__name__)
        return NETWORK_ERROR
    if status != 200:
        log.warning("Remotive answered with HTTP status %d", status)
        return HTTP_ERROR
    if len(body) > MAX_RESPONSE_BYTES:
        return RESPONSE_TOO_LARGE
    try:
        feed = parse_feed(body)
    except FeedError as exc:
        return exc.code

    moment = datetime.fromtimestamp(now, tz=timezone.utc)
    _write_atomic(folder / f"{_stamp(moment)}{CACHE_SUFFIX}", body)
    for _, old in _cache_files(folder)[CACHE_KEEP:]:
        old.unlink(missing_ok=True)
    log.info("fetched %d Remotive postings (%d skipped)", len(feed.postings), feed.skipped)
    return feed


def _result(found: tuple[datetime, ParsedFeed], *, from_cache: bool, stale: bool,
            warnings: list[str]) -> FetchResult:
    fetched_at, feed = found
    if feed.skipped:
        warnings = [*warnings, SKIPPED_POSTINGS]
    return FetchResult(
        postings=feed.postings, fetched_at=fetched_at, from_cache=from_cache, stale=stale,
        skipped=feed.skipped, reported_count=feed.reported_count,
        legal_notice=feed.legal_notice, warnings=warnings,
    )


def run_fetch() -> dict[str, Any]:
    """What the Airflow task runs: fetch, then report a small summary (never the postings,
    which go through XCom into the database)."""
    result = fetch_postings(max_age_s=SCHEDULED_MAX_AGE_S)
    return {
        "source": SOURCE,
        "count": len(result.postings),
        "skipped": result.skipped,
        "from_cache": result.from_cache,
        "stale": result.stale,
        "fetched_at": result.fetched_at.isoformat(),
        "warnings": result.warnings,
    }
