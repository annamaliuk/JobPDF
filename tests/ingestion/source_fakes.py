"""Synthetic responses and a scripted transport for the job-source tests.

The shapes follow each provider's documentation; none was captured from a live call, so a
first real run is what confirms them.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from jobpdf.ingestion.sources.models import Policy, Request

T0 = 1_790_000_000.0  # 2026-09-21T14:13:20Z
HOUR = 3600.0

KEY = "sekret-key-0123456789"  # appears in requests only; must never reach logs or cache
APP_ID, APP_KEY = "app-id-777", "app-key-999"


class Clock:
    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Transport:
    """Answers requests from a script: items are ``(status, body)``, an exception to raise,
    or a callable ``(Request) -> (status, body)``. The last item repeats."""

    def __init__(self, *script) -> None:
        self.script = list(script)
        self.requests: list[Request] = []
        self.limits: list[tuple[float, int]] = []

    def __call__(self, request: Request, timeout: float, max_bytes: int):
        self.requests.append(request)
        self.limits.append((timeout, max_bytes))
        item = self.script[min(len(self.requests), len(self.script)) - 1]
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = item(request)
        status, body = item
        return status, body if isinstance(body, bytes) else json.dumps(body).encode()

    @property
    def calls(self) -> int:
        return len(self.requests)


def ok(body) -> tuple[int, bytes]:
    return 200, body if isinstance(body, bytes) else json.dumps(body).encode()


def fast(policy: Policy | None = None, **changes) -> Policy:
    """A policy that never really sleeps or limits unless a test says so."""
    base = policy or Policy()
    values = {**base.__dict__, "pause_s": 0.0, **changes}
    return Policy(**values)


no_sleep: Callable[[float], None] = lambda seconds: None  # noqa: E731


# --- payloads ---------------------------------------------------------------------


def muse_job(i: int = 1, **changes) -> dict:
    data = {
        "id": 1000 + i, "name": f"Data Engineer {i}", "type": "external",
        "contents": f"<p>Build pipelines {i}</p>", "publication_date": "2026-09-20T10:00:00Z",
        "locations": [{"name": "Flexible / Remote"}],
        "categories": [{"name": "Data and Analytics"}],
        "levels": [{"name": "Mid Level", "short_name": "mid"}],
        "tags": [], "refs": {"landing_page": f"https://www.themuse.com/jobs/acme/{i}"},
        "company": {"id": 7, "short_name": "acme", "name": "Acme"},
    }
    return {**data, **changes}


def muse_page(jobs: list[dict], page: int = 1, page_count: int = 1) -> dict:
    return {"page": page, "page_count": page_count, "items_per_page": 20, "total": 99,
            "results": jobs}


def arbeitnow_job(i: int = 1, **changes) -> dict:
    data = {
        "slug": f"python-dev-{i}-berlin", "title": f"Python Developer {i}",
        "company_name": "Beispiel GmbH", "description": f"<p>Wir suchen {i}</p>",
        "remote": True, "url": f"https://www.arbeitnow.com/jobs/companies/x/{i}",
        "tags": ["Python", "Backend"], "job_types": ["full time"], "location": "Berlin",
        "created_at": 1_789_900_000,
    }
    return {**data, **changes}


def arbeitnow_page(jobs: list[dict], next_url: str | None = None) -> dict:
    return {"data": jobs, "links": {"first": "https://www.arbeitnow.com/api/job-board-api?page=1",
                                    "next": next_url},
            "meta": {"current_page": 1}}


def adzuna_job(i: int = 1, **changes) -> dict:
    data = {
        "id": f"48{i}", "title": f"Backend Engineer {i}", "description": "Snippet of the ad…",
        "redirect_url": f"https://www.adzuna.co.uk/jobs/details/48{i}?se=x",
        "created": "2026-09-19T08:30:00Z", "company": {"display_name": "Globex"},
        "location": {"display_name": "London", "area": ["UK", "London"]},
        "category": {"label": "IT Jobs", "tag": "it-jobs"}, "contract_time": "full_time",
        "salary_min": 50000.0, "salary_max": 60000.0, "salary_is_predicted": "0",
    }
    return {**data, **changes}


def adzuna_page(jobs: list[dict], count: int | None = None) -> dict:
    return {"count": len(jobs) if count is None else count, "mean": 55000.0, "results": jobs}


def jooble_job(i: int = 1, **changes) -> dict:
    data = {
        "title": f"ML Engineer {i}", "location": "Remote", "snippet": "We are looking for…",
        "salary": "", "source": "example.com", "type": "Full-time",
        "link": f"https://jooble.org/desc/{i}00", "company": "Initech",
        "updated": "2026-09-18T00:00:00.0000000", "id": 5_000 + i,
    }
    return {**data, **changes}


def jooble_page(jobs: list[dict], total: int | None = None) -> dict:
    return {"totalCount": len(jobs) if total is None else total, "jobs": jobs}
