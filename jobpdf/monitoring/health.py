"""The health report behind ``/health`` (JM-34): one function, no web framework.

``build_health(...)`` runs a handful of cheap checks and returns a ``HealthReport``. The
FastAPI app (JM-36) serves it as JSON with ``http_status(report)``.

Status of a check:
- ``ok``
- ``degraded``  something reduces quality or freshness, but requests are still served
                (stale vacancies, a missing optional source, the skill index unreachable, ...)
- ``down``      a core function cannot work: the database is unreachable (uploads fail
                closed, JM-30) or the LLM key is missing (a CV cannot be extracted)

The overall status is the worst of the checks. ``http_status`` answers 503 only for
``down``; ``degraded`` still answers 200, in line with "graceful degradation" in the spec.

What it never contains: secrets, URLs, host names, exception messages, CV or posting text.
Only statuses, stable codes, counts, ages and version stamps. A check that raises is turned
into ``check_failed`` (the exception type is logged, never its message), so the endpoint
itself never crashes.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import psycopg
from pydantic import BaseModel, ConfigDict

from jobpdf.api.response import SCHEMA_VERSION
from jobpdf.ingestion.sources.runner import CACHE_STAMP, CACHE_SUFFIX, DAY_S, LEDGER_NAME, cache_dir
from jobpdf.ingestion.vacancies import SOURCES
from jobpdf.ranking.models import RANKING_VERSION
from jobpdf.ranking.semantic import embedding_version

log = logging.getLogger(__name__)

OK = "ok"
DEGRADED = "degraded"
DOWN = "down"
HTTP_STATUS = {OK: 200, DEGRADED: 200, DOWN: 503}
_SEVERITY = {OK: 0, DEGRADED: 1, DOWN: 2}

# How old data may get before the report calls it stale. Fetches run every 8 hours, so a day
# means three missed cycles in a row.
FETCH_STALE_AFTER_S = 24 * 3600
VACANCY_STALE_AFTER_S = 24 * 3600
DB_CONNECT_TIMEOUT_S = 3

# stable codes
NOT_CONFIGURED = "not_configured"
UNREACHABLE = "unreachable"
DATABASE_DOWN = "database_down"
TABLE_MISSING = "table_missing"
NO_VACANCIES = "no_vacancies"
STALE_VACANCIES = "stale_vacancies"
NO_CACHE = "no_cache"
STALE_FETCH = "stale_fetch"
INDEX_UNAVAILABLE = "index_unavailable"
LLM_TIER_DISABLED = "llm_tier_disabled"
KEY_MISSING = "key_missing"
NO_API_KEYS = "no_api_keys"
CHECK_FAILED = "check_failed"

LLM_KEY_ENV = "ANTHROPIC_API_KEY"


class Check(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    code: str | None = None  # why it is not ok; None when ok
    data: dict[str, Any] = {}  # JSON scalars and small maps: counts, ages, versions
    latency_ms: int = 0


class HealthReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    checks: dict[str, Check]
    versions: dict[str, str | None]
    checked_at: datetime
    duration_ms: int

    @property
    def http_status(self) -> int:
        return HTTP_STATUS[self.status]


def http_status(report: HealthReport) -> int:
    return report.http_status


class _Normalizer(Protocol):
    def status(self) -> Mapping[str, str | None]: ...


class _KeyList(Protocol):
    def status(self) -> str: ...


def worst(statuses: list[str]) -> str:
    return max(statuses, key=_SEVERITY.__getitem__, default=OK)


def build_health(
    *,
    normalizer: _Normalizer | None = None,
    api_keys: _KeyList | None = None,
    connect: Callable[[], psycopg.Connection] | None = None,
    cache_base: Path | None = None,
    environ: Mapping[str, str] | None = None,
    clock: Callable[[], float] | None = None,
) -> HealthReport:
    """Run the checks. A dependency that is not passed is not checked (its check is left out
    of the report), so the app must pass what it uses.

    ``environ`` replaces the process environment for tests; the LLM key is read from it the
    way the LLM client reads it (the process environment only, not ``.env``)."""
    now = (clock or time.time)()
    llm_env = os.environ if environ is None else environ
    started = time.perf_counter()
    checks: dict[str, Check] = {}

    conn: psycopg.Connection | None = None
    try:
        checks["database"], conn = _timed(DOWN, _database, connect)
        checks["vacancies"] = _timed(DEGRADED, _vacancies, conn, now)[0]
        if normalizer is not None:
            checks["normalization"] = _timed(DEGRADED, _normalization, normalizer)[0]
        checks["llm"] = _timed(DEGRADED, _llm, llm_env)[0]
        if api_keys is not None:
            checks["api_keys"] = _timed(DEGRADED, _api_keys, api_keys)[0]
        checks["fetch"] = _timed(DEGRADED, _fetch, cache_base, now)[0]
        versions = _versions(checks.get("normalization"))
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # a broken connection is already reported by its check
                pass

    return HealthReport(
        status=worst([c.status for c in checks.values()]), checks=checks, versions=versions,
        checked_at=datetime.fromtimestamp(now, tz=timezone.utc),
        duration_ms=round((time.perf_counter() - started) * 1000),
    )


def _timed(failure_status: str, check: Callable[..., Any], *args: Any) -> Any:
    """Run one check; time it; turn an unexpected exception into ``check_failed``."""
    started = time.perf_counter()
    try:
        outcome = check(*args)
    except Exception as exc:
        log.warning("health check %s failed (%s)", check.__name__, type(exc).__name__)
        outcome = Check(status=failure_status, code=CHECK_FAILED)
    ms = round((time.perf_counter() - started) * 1000)
    if isinstance(outcome, tuple):  # (check, extra)
        return outcome[0].model_copy(update={"latency_ms": ms}), outcome[1]
    return outcome.model_copy(update={"latency_ms": ms}), None


# --- the checks -------------------------------------------------------------------------


def _database(connect: Callable[[], psycopg.Connection] | None
              ) -> tuple[Check, psycopg.Connection | None]:
    """Reachable? Opens a fresh connection (the pool's may be the thing that is broken)."""
    try:
        conn = (connect or _default_connect)()
    except _NotConfigured:
        return Check(status=DOWN, code=NOT_CONFIGURED), None
    except psycopg.Error:
        return Check(status=DOWN, code=UNREACHABLE), None
    try:
        conn.execute("SELECT 1")
        conn.rollback()
    except psycopg.Error:
        conn.close()
        return Check(status=DOWN, code=UNREACHABLE), None
    return Check(status=OK), conn


class _NotConfigured(Exception):
    pass


def _default_connect() -> psycopg.Connection:
    from jobpdf.normalization.db import database_url

    url = database_url()
    if not url:
        raise _NotConfigured
    return psycopg.connect(url, connect_timeout=DB_CONNECT_TIMEOUT_S)


def _vacancies(conn: psycopg.Connection | None, now: float) -> Check:
    if conn is None:
        return Check(status=DEGRADED, code=DATABASE_DOWN)
    try:
        rows = conn.execute(
            "SELECT source, count(*), max(last_seen_at), max(posted_at) "
            "FROM vacancies GROUP BY source ORDER BY source").fetchall()
    except psycopg.errors.UndefinedTable:
        return Check(status=DEGRADED, code=TABLE_MISSING)
    finally:
        conn.rollback()  # also ends the aborted transaction after a missing table
    total = sum(r[1] for r in rows)
    newest_seen = max((r[2] for r in rows), default=None)
    newest_posted = max((r[3] for r in rows if r[3] is not None), default=None)
    data: dict[str, Any] = {
        "total": total,
        "by_source": {r[0]: r[1] for r in rows},
        "newest_seen_age_s": None if newest_seen is None else _age(now, newest_seen),
        "newest_posting_age_s": None if newest_posted is None else _age(now, newest_posted),
    }
    if total == 0:
        return Check(status=DEGRADED, code=NO_VACANCIES, data=data)
    if data["newest_seen_age_s"] > VACANCY_STALE_AFTER_S:
        return Check(status=DEGRADED, code=STALE_VACANCIES, data=data)
    return Check(status=OK, data=data)


def _normalization(normalizer: _Normalizer) -> Check:
    status = dict(normalizer.status())
    data = {key: value for key, value in status.items()
            if value is None or isinstance(value, str)}
    if status.get("index") != "ok":
        return Check(status=DEGRADED, code=INDEX_UNAVAILABLE, data=data)
    if status.get("llm") != "enabled":
        return Check(status=DEGRADED, code=LLM_TIER_DISABLED, data=data)
    return Check(status=OK, data=data)


def _llm(environ: Mapping[str, str]) -> Check:
    """Is the key there? ("set" or "missing", never the value.) Without it no CV can be
    extracted, so this is ``down`` rather than ``degraded``."""
    if (environ.get(LLM_KEY_ENV) or "").strip():
        return Check(status=OK, data={"key": "set"})
    return Check(status=DOWN, code=KEY_MISSING, data={"key": "missing"})


def _api_keys(api_keys: _KeyList) -> Check:
    state = api_keys.status()
    if state == "set":
        return Check(status=OK, data={"keys": "set"})
    return Check(status=DEGRADED, code=NO_API_KEYS, data={"keys": "missing"})


def _fetch(base: Path | None, now: float) -> Check:
    """Age of the newest cached fetch per source and requests used in the last 24 hours.
    Remotive always counts. Another source counts once it has been fetched at least once
    (so a source nobody set up cannot make the service look unhealthy, but one that was
    working and stopped does)."""
    sources: dict[str, dict[str, Any]] = {}
    problem: str | None = None
    for name in SOURCES:
        folder = cache_dir(name, base)
        newest = _newest_cache(folder)
        age = None if newest is None else max(0, round(now - newest.timestamp()))
        counts = name == "remotive" or newest is not None
        sources[name] = {"age_s": age, "requests_24h": _requests_24h(folder, now),
                         "counted": counts}
        if counts and problem is None:
            if age is None:
                problem = NO_CACHE
            elif age > FETCH_STALE_AFTER_S:
                problem = STALE_FETCH
    return Check(status=OK if problem is None else DEGRADED, code=problem,
                 data={"sources": sources})


def _newest_cache(folder: Path) -> datetime | None:
    newest = None
    for path in folder.glob(f"*{CACHE_SUFFIX}"):
        try:
            moment = datetime.strptime(path.stem, CACHE_STAMP).replace(tzinfo=timezone.utc)
        except ValueError:  # the ledger and stray files
            continue
        newest = moment if newest is None or moment > newest else newest
    return newest


def _requests_24h(folder: Path, now: float) -> int | None:
    """From the fetch guard's ledger; None if there is none or it is unreadable."""
    try:
        times = json.loads((folder / LEDGER_NAME).read_text(encoding="utf-8"))
        return sum(1 for t in times if now - float(t) < DAY_S)
    except (OSError, ValueError, TypeError):
        return None


def _age(now: float, moment: datetime) -> int:
    return max(0, round(now - moment.timestamp()))


def _versions(normalization: Check | None) -> dict[str, str | None]:
    """Code versions, plus the normalization ones taken from its check (``status()`` runs a
    live probe, so it is called once)."""
    versions: dict[str, str | None] = {
        "ranking": RANKING_VERSION, "response_schema": SCHEMA_VERSION,
        "embedding": embedding_version(),
    }
    if normalization is not None:
        for key in ("taxonomy", "index", "matcher_prompt"):
            versions[key] = normalization.data.get(f"{key}_version")
    return versions
