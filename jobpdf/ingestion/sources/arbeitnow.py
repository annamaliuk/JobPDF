"""Arbeitnow (https://www.arbeitnow.com/api/job-board-api): a free job board feed, mostly
Germany and remote work. No key. Terms and rate limits are not stated on the docs page, so
the policy is cautious.

Answer: ``{"data": [...], "links": {"next": url|null}, "meta": {...}}``; each entry has
``slug``, ``title``, ``company_name``, ``description`` (HTML), ``remote`` (bool), ``url``,
``tags``, ``job_types``, ``location`` and ``created_at`` (unix time). The next page is taken
from ``links.next`` only if it stays on arbeitnow.com.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from jobpdf.ingestion.sources.models import (
    INVALID_RESPONSE,
    FeedError,
    ParsedPage,
    Policy,
    Request,
    SourcePosting,
    parse_items,
    parse_time,
    text_or_none,
)

NAME = "arbeitnow"
API_URL = "https://www.arbeitnow.com/api/job-board-api"
HOST = "www.arbeitnow.com"
POLICY = Policy(max_pages=5, max_requests_per_day=30)


def _strings(value: Any) -> list[str]:
    return [t for t in (text_or_none(v) for v in value) if t] if isinstance(value, list) else []


class Arbeitnow:
    name = NAME

    def __init__(self, *, policy: Policy = POLICY, remote_only: bool = True) -> None:
        self.policy = policy
        self.remote_only = remote_only

    def is_configured(self) -> bool:
        return True

    def next_request(self, pages: list[Any]) -> Request | None:
        if not pages:
            return Request("GET", API_URL)
        last = pages[-1] if isinstance(pages[-1], dict) else {}
        links = last.get("links")
        target = links.get("next") if isinstance(links, dict) else None
        if not isinstance(target, str):
            return None
        parts = urlsplit(target)
        if parts.scheme != "https" or parts.hostname != HOST:
            return None  # never follow a link off the provider's own host
        return Request("GET", target)

    def parse_page(self, page: Any) -> ParsedPage:
        if not isinstance(page, dict):
            raise FeedError(INVALID_RESPONSE)
        return parse_items(page.get("data"), self._convert)

    def _convert(self, item: dict[str, Any]) -> SourcePosting | None:
        remote = item.get("remote")
        remote = remote if isinstance(remote, bool) else None
        if self.remote_only and remote is not True:
            return None
        tags = _strings(item.get("tags"))
        return SourcePosting(
            source=NAME, source_id=item["slug"], url=item["url"], title=item["title"],
            company=text_or_none(item.get("company_name")),
            location_raw=text_or_none(item.get("location")), remote=remote,
            posted_at=parse_time(item.get("created_at")),
            description=text_or_none(item.get("description")), tags=tags,
            employment_type=", ".join(_strings(item.get("job_types"))) or None,
        )
