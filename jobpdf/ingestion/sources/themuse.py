"""The Muse (https://www.themuse.com/developers/api/v2): public job listings.

``GET /api/public/jobs?page=N&location=Flexible / Remote`` answers with
``{"page", "page_count", "results": [...]}``. The API key is optional (it only raises the
rate limit); it is read from ``THE_MUSE_API_KEY`` and sent as ``api_key``.

The Muse's terms are not verified here: check them before publishing results anywhere.
Descriptions are full HTML. Every posting keeps the link back (``refs.landing_page``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlencode

from jobpdf.ingestion.sources.models import (
    INVALID_RESPONSE,
    FeedError,
    ParsedPage,
    Policy,
    Request,
    SourcePosting,
    int_or_none,
    names,
    nested_text,
    parse_items,
    parse_time,
    setting,
    text_or_none,
)

NAME = "themuse"
API_URL = "https://www.themuse.com/api/public/jobs"
API_KEY_ENV = "THE_MUSE_API_KEY"
REMOTE_LOCATION = "Flexible / Remote"  # The Muse's own name for remote jobs
FIRST_PAGE = 1
POLICY = Policy(max_pages=5, max_requests_per_day=30)


class TheMuse:
    name = NAME

    def __init__(self, *, api_key: str | None = None, policy: Policy = POLICY,
                 remote_only: bool = True) -> None:
        self.api_key = api_key
        self.policy = policy
        self.remote_only = remote_only

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> TheMuse:
        return cls(api_key=setting(API_KEY_ENV, environ))

    def is_configured(self) -> bool:
        return True  # the key is optional

    def next_request(self, pages: list[Any]) -> Request | None:
        if not pages:
            number = FIRST_PAGE
        else:
            last = pages[-1] if isinstance(pages[-1], dict) else {}
            current, total = int_or_none(last.get("page")), int_or_none(last.get("page_count"))
            if current is None or total is None or current + 1 >= total + FIRST_PAGE:
                return None
            number = current + 1
        query = {"page": number, "descending": "true"}
        if self.remote_only:
            query["location"] = REMOTE_LOCATION
        if self.api_key:
            query["api_key"] = self.api_key
        return Request("GET", f"{API_URL}?{urlencode(query)}")

    def parse_page(self, page: Any) -> ParsedPage:
        if not isinstance(page, dict):
            raise FeedError(INVALID_RESPONSE)
        return parse_items(page.get("results"), self._convert)

    def _convert(self, item: dict[str, Any]) -> SourcePosting | None:
        locations = names(item.get("locations"))
        remote = any("remote" in place.lower() or "flexible" in place.lower()
                     for place in locations)
        if self.remote_only and not remote:
            return None
        return SourcePosting(
            source=NAME, source_id=str(item["id"]),
            url=item["refs"]["landing_page"], title=item["name"],
            company=nested_text(item.get("company"), "name"),
            location_raw=", ".join(locations) or None, remote=remote,
            posted_at=parse_time(item.get("publication_date")),
            description=text_or_none(item.get("contents")),
            tags=[*names(item.get("levels")), *names(item.get("tags"))],
            category=(names(item.get("categories")) or [None])[0],
            employment_type=text_or_none(item.get("type")),
        )
