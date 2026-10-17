"""Jooble (https://jooble.org/api/about): ``POST https://jooble.org/api/<key>`` with a JSON
body ``{"keywords", "location", "page"}``.

Needs ``JOOBLE_API_KEY`` (in ``.env``). The key is part of the URL, so a request is never
logged. Optional: ``JOOBLE_KEYWORDS`` (default ``remote``) and ``JOOBLE_LOCATION`` (default
empty). Limits per key are not published, so the policy is cautious; Jooble's terms are
not verified here: check them before showing results to anyone.

Answer: ``{"totalCount": n, "jobs": [...]}``; each job has ``title``, ``location``,
``snippet`` (an excerpt), ``salary``, ``source``, ``type``, ``link`` (via Jooble),
``company``, ``updated`` and sometimes ``id``. Without an id the posting's id is derived
from its link.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from jobpdf.ingestion.sources.models import (
    INVALID_RESPONSE,
    FeedError,
    ParsedPage,
    Policy,
    Request,
    SourcePosting,
    int_or_none,
    parse_items,
    parse_time,
    setting,
    stable_id,
    text_or_none,
)

NAME = "jooble"
API_BASE = "https://jooble.org/api/"
API_KEY_ENV = "JOOBLE_API_KEY"
KEYWORDS_ENV = "JOOBLE_KEYWORDS"
LOCATION_ENV = "JOOBLE_LOCATION"
DEFAULT_KEYWORDS = "remote"
POLICY = Policy(max_pages=4, max_requests_per_day=20)


class Jooble:
    name = NAME

    def __init__(self, *, api_key: str | None, keywords: str = DEFAULT_KEYWORDS,
                 location: str = "", policy: Policy = POLICY) -> None:
        self.api_key = api_key
        self.keywords, self.location, self.policy = keywords, location, policy

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> Jooble:
        return cls(
            api_key=setting(API_KEY_ENV, environ),
            keywords=setting(KEYWORDS_ENV, environ) or DEFAULT_KEYWORDS,
            location=setting(LOCATION_ENV, environ) or "",
        )

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def next_request(self, pages: list[Any]) -> Request | None:
        if pages:
            last = pages[-1] if isinstance(pages[-1], dict) else {}
            jobs, total = last.get("jobs"), int_or_none(last.get("totalCount"))
            if not isinstance(jobs, list) or not jobs:
                return None
            seen = sum(len(p.get("jobs") or []) for p in pages if isinstance(p, dict))
            if total is not None and seen >= total:
                return None
        body = json.dumps({"keywords": self.keywords, "location": self.location,
                           "page": str(len(pages) + 1)}).encode("utf-8")
        return Request("POST", API_BASE + quote(self.api_key or "", safe=""),
                       headers={"Content-Type": "application/json"}, body=body)

    def parse_page(self, page: Any) -> ParsedPage:
        if not isinstance(page, dict):
            raise FeedError(INVALID_RESPONSE)
        return parse_items(page.get("jobs"), self._convert)

    def _convert(self, item: dict[str, Any]) -> SourcePosting | None:
        link = item["link"]
        raw_id = item.get("id")
        source_id = str(raw_id) if isinstance(raw_id, (int, str)) and str(raw_id).strip() \
            else stable_id(link)
        return SourcePosting(
            source=NAME, source_id=source_id, url=link, title=item["title"],
            company=text_or_none(item.get("company")),
            location_raw=text_or_none(item.get("location")),
            posted_at=parse_time(item.get("updated")),
            description=text_or_none(item.get("snippet")), description_is_excerpt=True,
            employment_type=text_or_none(item.get("type")),
            salary_raw=text_or_none(item.get("salary")),
        )
