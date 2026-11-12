"""The common shapes shared by every job source (adapters, fetcher, ingestion)."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"

# Why nothing could be fetched; also used as warnings (see runner.py).
NOT_CONFIGURED = "not_configured"
QUOTA_EXHAUSTED = "quota_exhausted"
TOO_SOON = "too_soon"
NETWORK_ERROR = "network_error"
HTTP_ERROR = "http_error"
RESPONSE_TOO_LARGE = "response_too_large"
INVALID_RESPONSE = "invalid_response"
NO_POSTINGS = "no_postings"
STALE_CACHE = "stale_cache"
SKIPPED_POSTINGS = "skipped_postings"
PARTIAL = "partial"  # warning: fewer pages than planned were fetched


class FetchError(RuntimeError):
    """No postings could be obtained and there is no cache to fall back on."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class FeedError(ValueError):
    """A response is not a usable feed for this source (``code`` says why)."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class SourcePosting(BaseModel):
    """One vacancy, whatever site it came from. (source, source_id) is its identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    source_id: str
    url: str  # the posting (or the provider's redirect to it): always shown as a link back
    title: str
    company: str | None = None
    location_raw: str | None = None  # as the source wrote it, for the eligibility checks
    remote: bool | None = None  # None: the source does not say
    posted_at: datetime | None = None  # timezone-aware
    description: str | None = None  # HTML or text as received
    description_is_excerpt: bool = False  # True: only a snippet; requirement extraction is weaker
    tags: list[str] = []
    category: str | None = None
    employment_type: str | None = None
    salary_raw: str | None = None

    @field_validator("source", "source_id", "title")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("url must be http(s)")
        return value

    @field_validator("posted_at")
    @classmethod
    def _utc(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class ParsedPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    postings: list[SourcePosting]
    skipped: int = 0  # malformed postings dropped
    filtered: int = 0  # well-formed but not wanted (e.g. not remote)
    raw_count: int = 0  # entries the page held, before skipping and filtering


@dataclass(frozen=True, repr=False)
class Request:
    """One HTTP request. The URL, headers and body may hold an API key, so the object never
    shows them: ``repr`` is redacted and nothing should log a request."""

    method: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes | None = None

    def __repr__(self) -> str:
        return f"Request({self.method}, <redacted>)"


@dataclass(frozen=True)
class Policy:
    """How politely a source is fetched. Rate limits of these APIs are not all published,
    so the defaults are deliberately cautious."""

    max_cache_age_s: float = 6 * 3600
    max_requests_per_day: int = 40
    min_interval_s: float = 30.0  # between two runs
    max_pages: int = 5  # requests per run
    pause_s: float = 1.0  # between the pages of one run
    keep: int = 8  # cached runs kept
    timeout_s: float = 30.0
    max_response_bytes: int = 16 * 1024 * 1024


class SourceAdapter(Protocol):
    name: str
    policy: Policy

    def is_configured(self) -> bool:
        """False when a required key is missing: the source is then skipped, not an error."""

    def next_request(self, pages: list[Any]) -> Request | None:
        """The request after ``pages`` (decoded JSON of the pages fetched so far), or None
        when the run is complete."""

    def parse_page(self, page: Any) -> ParsedPage:
        """Postings of one decoded page; raises FeedError if it is not a feed page."""


class SourceResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    postings: list[SourcePosting]
    fetched_at: datetime
    from_cache: bool
    stale: bool
    pages: int
    skipped: int
    filtered: int
    warnings: list[str]  # codes only


# --- small helpers the adapters share ----------------------------------------------


def setting(name: str, environ: Mapping[str, str] | None = None,
            env_file: Path = ENV_FILE) -> str | None:
    """A setting from ``environ``; with ``environ=None`` the process environment, then
    ``.env`` (KEY=VALUE lines). A real variable always wins."""
    if environ is not None:
        return (environ.get(name) or "").strip() or None
    if os.environ.get(name, "").strip():
        return os.environ[name].strip()
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        key, sep, value = line.strip().partition("=")
        if sep and key.strip() == name:
            return value.strip().strip("'\"") or None
    return None


def text_or_none(value: Any) -> str | None:
    """A stripped non-empty string, else None (numbers and objects are not text)."""
    return value.strip() or None if isinstance(value, str) else None


def stable_id(*parts: str) -> str:
    """A short id derived from text, for sources that give no id of their own."""
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:20]


_FRACTION = re.compile(r"(\.\d{6})\d+")  # Jooble sends 7 digits; Python reads at most 6


def parse_time(value: Any) -> datetime | None:
    """ISO text (with or without a 'Z' or offset), or a unix timestamp; None if unreadable."""
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if isinstance(value, str) and value.strip():
            text = _FRACTION.sub(r"\1", value.strip().replace("Z", "+00:00"))
            moment = datetime.fromisoformat(text)
            return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None
    return None


def parse_items(items: Any, convert: Any) -> ParsedPage:
    """Run ``convert(item) -> SourcePosting | None`` over a page's entries. None means
    "well-formed but not wanted" (filtered); an entry that raises or is not an object is
    malformed (skipped). ``items`` that is not a list is not a feed page."""
    if not isinstance(items, list):
        raise FeedError(INVALID_RESPONSE)
    postings: list[SourcePosting] = []
    skipped = filtered = 0
    for item in items:  # an entry that is not an object fails in convert() and is skipped
        try:
            posting = convert(item)
        except (ValidationError, KeyError, TypeError, ValueError, AttributeError):
            skipped += 1
            continue
        if posting is None:
            filtered += 1
        else:
            postings.append(posting)
    return ParsedPage(postings=postings, skipped=skipped, filtered=filtered,
                      raw_count=len(items))


def int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def names(value: Any) -> list[str]:
    """``[{"name": "x"}, ...]`` -> ``["x", ...]``; anything else -> []."""
    if not isinstance(value, list):
        return []
    return [n for item in value
            if isinstance(item, dict) and (n := text_or_none(item.get("name")))]


def nested_text(value: Any, key: str) -> str | None:
    """``value[key]`` as text when ``value`` is an object, else None."""
    return text_or_none(value.get(key)) if isinstance(value, dict) else None
