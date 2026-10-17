"""Alert rules over the health report and the metrics (JM-40).

``evaluate(health, metrics)`` turns what ``/health`` (JM-34) and ``/metrics`` report into a list
of ``Alert``s. It decides nothing about delivery: ``uptime.py`` fetches, de-duplicates and
posts, this module only says what is wrong and how bad it is.

Severity.
- ``critical``  the service cannot do its job now: it does not answer, the database or the
                LLM key is down, or half the analyses fail because of us.
- ``warning``   something is degrading: stale vacancies or fetches, a share of analyses failing
                on our side, slow or expensive analyses.

What is NOT counted as a fault: a CV that cannot be read (corrupt, encrypted, no text, too
large). Those are the caller's problem and would page you for someone else's PDF. Only
failures on our side count: an LLM error, an invalid extraction, an unexpected exception.

The metrics are per process and reset on restart, so the rates are "since the last restart"
and need ``MIN_ANALYSES`` before they mean anything.

Privacy. An alert carries a code, a severity and a fixed sentence. Anything taken from the
server's answer (a check name, a code) is reduced to ``[a-z0-9_.:-]`` and cut to 64 characters
first, so a surprising answer cannot put text into a notification.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

from jobpdf.monitoring.health import DEGRADED, DOWN, HealthReport
from jobpdf.monitoring.metrics import MetricsSnapshot

CRITICAL = "critical"
WARNING = "warning"
_RANK = {WARNING: 1, CRITICAL: 2}

# Thresholds. Metrics are since the last restart; nothing below is judged on fewer analyses
# than MIN_ANALYSES.
MIN_ANALYSES = 10
FAULT_RATE_WARNING = 0.2  # share of analyses that failed on our side
FAULT_RATE_CRITICAL = 0.5
SLOW_P95_MS = 60_000  # p95 of the whole analysis
TOKENS_PER_ANALYSIS_MAX = 40_000  # input + output, averaged over successful analyses
TOTAL_STAGE = "total"
MIN_STAGE_SAMPLES = MIN_ANALYSES

# Failure codes that mean "we failed", whatever the stage. Parse errors and no_text are the
# caller's file and are deliberately absent.
SERVICE_FAULT_CODES = frozenset({"llm_error", "llm_not_configured", "extraction_invalid"})
UNEXPECTED_KEY = "unexpected"
LIMITER_UNAVAILABLE = "limiter_unavailable"

# Alert codes that do not come from a health check.
UNREACHABLE = "unreachable"
BAD_RESPONSE = "bad_response"
HEALTH_DOWN = "health_down"
METRICS_UNAVAILABLE = "metrics_unavailable"
SERVICE_ERRORS = "service_errors"
SLOW_ANALYSES = "slow_analyses"
TOKEN_SPIKE = "token_spike"
LIMITER_FAILING = "limiter_failing"
CHECK_PREFIX = "check."  # check.<name>.<code>, one per health check that is not ok

MESSAGES = {
    UNREACHABLE: "The service does not answer on /health.",
    BAD_RESPONSE: "/health answered, but not with a health report.",
    HEALTH_DOWN: "The health report says down.",
    METRICS_UNAVAILABLE: "/metrics cannot be read: the monitor key is rejected or the call "
                         "failed.",
    SERVICE_ERRORS: "Analyses are failing on our side (LLM or unexpected errors).",
    SLOW_ANALYSES: "Analyses are slow: the 95th percentile is over the limit.",
    TOKEN_SPIKE: "Analyses use many more LLM tokens than expected.",
    LIMITER_FAILING: "The rate limiter cannot reach its database; uploads are refused.",
}
CHECK_MESSAGE = "Health check reports a problem."  # check.* alerts: name and code say which
CHECK_HINTS = {
    "database_down": "The database is unreachable.",
    "table_missing": "A table is missing: run scripts/migrate.py.",
    "no_vacancies": "There are no vacancies to rank against.",
    "stale_vacancies": "Vacancies have not been refreshed for over a day.",
    "stale_fetch": "No source has been fetched for over a day.",
    "no_cache": "No fetch has ever succeeded.",
    "index_unavailable": "The skill index is not built or not reachable.",
    "key_missing": "The LLM key is not set.",
    "llm_tier_disabled": "The LLM matching tier is off.",
    "no_api_keys": "No API keys are configured.",
    "check_failed": "The check itself crashed.",
}

_SAFE = re.compile(r"[^a-z0-9_.:-]")
MAX_PART_CHARS = 64


@dataclass(frozen=True)
class Alert:
    code: str
    severity: str  # CRITICAL or WARNING

    @property
    def message(self) -> str:
        if self.code in MESSAGES:
            return MESSAGES[self.code]
        if self.code.startswith(CHECK_PREFIX):
            hint = CHECK_HINTS.get(self.code.rsplit(".", 1)[-1])
            return f"{CHECK_MESSAGE} {hint}" if hint else CHECK_MESSAGE
        return CHECK_MESSAGE


def worst(alerts: Collection[Alert]) -> str | None:
    """The highest severity among ``alerts``; None when there are none."""
    if not alerts:
        return None
    return max((a.severity for a in alerts), key=_RANK.__getitem__)


def safe_part(value: object) -> str:
    """A server-supplied name or code reduced to ``[a-z0-9_.:-]`` and cut to 64 characters."""
    if value is None:
        return "unknown"
    return _SAFE.sub("", str(value).lower())[:MAX_PART_CHARS] or "unknown"


def evaluate(
    health: HealthReport | None,
    metrics: MetricsSnapshot | None = None,
    *,
    ignore: Collection[str] = (),
) -> list[Alert]:
    """Every alert that applies, most severe first and then by code. ``health`` is None when
    the service did not answer (the caller adds ``unreachable`` / ``bad_response`` itself).
    Codes in ``ignore`` are dropped, e.g. ``check.fetch.no_cache`` on a site without Airflow."""
    alerts: list[Alert] = []
    if health is not None:
        alerts += _health_alerts(health)
    if metrics is not None:
        alerts += _metrics_alerts(metrics)
    skipped = set(ignore)
    kept = {a.code: a for a in alerts if a.code not in skipped}  # one alert per code
    return sorted(kept.values(), key=lambda a: (-_RANK[a.severity], a.code))


def _health_alerts(health: HealthReport) -> list[Alert]:
    alerts = []
    for name, check in health.checks.items():
        if check.status not in (DEGRADED, DOWN):
            continue
        severity = CRITICAL if check.status == DOWN else WARNING
        code = f"{CHECK_PREFIX}{safe_part(name)}.{safe_part(check.code)}"
        alerts.append(Alert(code, severity))
    if health.status == DOWN and not any(a.severity == CRITICAL for a in alerts):
        alerts.append(Alert(HEALTH_DOWN, CRITICAL))  # down, but no single check says why
    return alerts


def _metrics_alerts(metrics: MetricsSnapshot) -> list[Alert]:
    alerts = []
    total = metrics.analyses.total
    if total >= MIN_ANALYSES:
        rate = _service_faults(metrics.errors) / total
        if rate >= FAULT_RATE_CRITICAL:
            alerts.append(Alert(SERVICE_ERRORS, CRITICAL))
        elif rate >= FAULT_RATE_WARNING:
            alerts.append(Alert(SERVICE_ERRORS, WARNING))
    total_time = metrics.stages.get(TOTAL_STAGE)
    if total_time and total_time.count >= MIN_STAGE_SAMPLES and total_time.p95_ms > SLOW_P95_MS:
        alerts.append(Alert(SLOW_ANALYSES, WARNING))
    if metrics.analyses.ok >= MIN_ANALYSES:
        tokens = sum(metrics.tokens.values()) / metrics.analyses.ok
        if tokens > TOKENS_PER_ANALYSIS_MAX:
            alerts.append(Alert(TOKEN_SPIKE, WARNING))
    if metrics.access_denied.get(LIMITER_UNAVAILABLE, 0) > 0:
        alerts.append(Alert(LIMITER_FAILING, WARNING))
    return alerts


def _service_faults(errors: dict[str, int]) -> int:
    """Analyses that failed because of us: ``stage:code`` keys with a service-fault code, plus
    the ``unexpected`` key."""
    return sum(
        count for key, count in errors.items()
        if key == UNEXPECTED_KEY or key.rpartition(":")[2] in SERVICE_FAULT_CODES
    )
