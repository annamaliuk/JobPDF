"""Builders for the alert and uptime tests: reports, snapshots and a fake HTTP service."""

import json
from datetime import datetime, timezone
from typing import Any

from jobpdf.monitoring.health import Check, HealthReport
from jobpdf.monitoring.metrics import Analyses, MetricsSnapshot, StageStats

NOW = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def check(status: str = "ok", code: str | None = None) -> Check:
    return Check(status=status, code=code)


def health(status: str = "ok", **checks: Check) -> HealthReport:
    return HealthReport(status=status, checks=checks, versions={}, checked_at=NOW,
                        duration_ms=5)


def healthy() -> HealthReport:
    return health("ok", database=check(), vacancies=check(), llm=check())


def snapshot(*, total: int = 0, failed: int = 0, errors: dict[str, int] | None = None,
             tokens: tuple[int, int] = (0, 0), p95_ms: int | None = None, samples: int = 100,
             denied: dict[str, int] | None = None) -> MetricsSnapshot:
    stages = {}
    if p95_ms is not None:
        stages["total"] = StageStats(count=samples, p50_ms=1, p95_ms=p95_ms, max_ms=p95_ms)
    return MetricsSnapshot(
        since=NOW, uptime_s=100, analyses=Analyses(total=total, ok=total - failed, failed=failed),
        errors=errors or {}, warnings={}, access_denied=denied or {}, stages=stages,
        tokens={"input": tokens[0], "output": tokens[1]})


def body(model: Any) -> bytes:
    return model.model_dump_json().encode()


class FakeService:
    """A ``fetch`` for ``probe``: routes maps a path to (status, bytes) or to an exception."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: dict[str, str], timeout: float):
        self.calls.append((url, dict(headers)))
        path = "/" + url.split("/", 3)[3]
        route = self.routes[path]
        if isinstance(route, Exception):
            raise route
        return route


def service(health_report: HealthReport | None = None, status: int = 200,
            metrics: MetricsSnapshot | None = None, **routes: Any) -> FakeService:
    table: dict[str, Any] = {
        "/health": (status, body(health_report or healthy())),
        "/metrics": (200, body(metrics or snapshot())),
    }
    table.update({"/" + k: v for k, v in routes.items()})
    return FakeService(table)


def raw(status: int, payload: Any) -> tuple[int, bytes]:
    return status, payload if isinstance(payload, bytes) else json.dumps(payload).encode()
