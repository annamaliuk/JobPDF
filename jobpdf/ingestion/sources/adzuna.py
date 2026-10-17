"""Adzuna (https://developer.adzuna.com): ``GET /v1/api/jobs/{country}/search/{page}``.

Needs ``ADZUNA_APP_ID`` and ``ADZUNA_APP_KEY`` (in ``.env``; both go into the query string,
so a request is never logged). Optional: ``ADZUNA_COUNTRY`` (default ``gb``) and
``ADZUNA_QUERY`` (default ``remote``). Adzuna's terms (attribution, display rules) are not
verified here: check them before showing results to anyone.

Answer: ``{"count": n, "results": [...]}``; each entry has ``id``, ``title``, ``description``
(a truncated snippet), ``redirect_url`` (the link back, also how Adzuna counts clicks),
``created``, ``company.display_name``, ``location.display_name``, ``category.label``,
``contract_time``, ``salary_min`` and ``salary_max``.
"""

from __future__ import annotations

import re
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
    nested_text,
    parse_items,
    parse_time,
    setting,
    text_or_none,
)

NAME = "adzuna"
API_BASE = "https://api.adzuna.com/v1/api/jobs"
APP_ID_ENV = "ADZUNA_APP_ID"
APP_KEY_ENV = "ADZUNA_APP_KEY"
COUNTRY_ENV = "ADZUNA_COUNTRY"
QUERY_ENV = "ADZUNA_QUERY"
DEFAULT_COUNTRY = "gb"
DEFAULT_QUERY = "remote"
RESULTS_PER_PAGE = 50
POLICY = Policy(max_pages=4, max_requests_per_day=30)
_COUNTRY = re.compile(r"^[a-z]{2}$")


def _salary(item: dict[str, Any]) -> str | None:
    low, high = item.get("salary_min"), item.get("salary_max")
    numbers = [n for n in (low, high) if isinstance(n, (int, float)) and not isinstance(n, bool)]
    if not numbers:
        return None
    text = " - ".join(f"{n:,.0f}" for n in dict.fromkeys(numbers))
    return text + (" (estimated)" if str(item.get("salary_is_predicted")) == "1" else "")


class Adzuna:
    name = NAME

    def __init__(self, *, app_id: str | None, app_key: str | None,
                 country: str = DEFAULT_COUNTRY, query: str = DEFAULT_QUERY,
                 policy: Policy = POLICY) -> None:
        if not _COUNTRY.match(country):
            raise ValueError("country must be a two-letter code")
        self.app_id, self.app_key = app_id, app_key
        self.country, self.query, self.policy = country, query, policy

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> Adzuna:
        return cls(
            app_id=setting(APP_ID_ENV, environ), app_key=setting(APP_KEY_ENV, environ),
            country=(setting(COUNTRY_ENV, environ) or DEFAULT_COUNTRY).lower(),
            query=setting(QUERY_ENV, environ) or DEFAULT_QUERY,
        )

    def is_configured(self) -> bool:
        return bool(self.app_id and self.app_key)

    def next_request(self, pages: list[Any]) -> Request | None:
        number = len(pages) + 1
        if pages:
            last = pages[-1] if isinstance(pages[-1], dict) else {}
            results, count = last.get("results"), int_or_none(last.get("count"))
            if not isinstance(results, list) or not results:
                return None
            if count is not None and number > -(-count // RESULTS_PER_PAGE):
                return None
        query = urlencode({
            "app_id": self.app_id, "app_key": self.app_key, "what": self.query,
            "results_per_page": RESULTS_PER_PAGE, "content-type": "application/json",
        })
        return Request("GET", f"{API_BASE}/{self.country}/search/{number}?{query}")

    def parse_page(self, page: Any) -> ParsedPage:
        if not isinstance(page, dict):
            raise FeedError(INVALID_RESPONSE)
        return parse_items(page.get("results"), self._convert)

    def _convert(self, item: dict[str, Any]) -> SourcePosting | None:
        return SourcePosting(
            source=NAME, source_id=str(item["id"]), url=item["redirect_url"],
            title=item["title"],
            company=nested_text(item.get("company"), "display_name"),
            location_raw=nested_text(item.get("location"), "display_name"),
            posted_at=parse_time(item.get("created")),
            description=text_or_none(item.get("description")), description_is_excerpt=True,
            category=nested_text(item.get("category"), "label"),
            employment_type=text_or_none(item.get("contract_time")),
            salary_raw=_salary(item),
        )
