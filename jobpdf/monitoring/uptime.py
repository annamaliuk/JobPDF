"""Uptime probe, alert tracking and webhook delivery (JM-40). Standard library only.

One cycle:

    probe(url, api_key)  ->  ProbeResult(alerts)         fetch /health (and /metrics)
    tracker.update(alerts, now)  ->  Changes             decide what is worth saying
    format_message(changes, label)  ->  text             fixed sentences, codes, no data
    post_webhook(url, text)                              JSON POST

Why a tracker. A probe every minute must not produce a message every minute, and one slow
answer must not wake anybody. So an alert becomes *active* only after it was seen in
``CONFIRM_CYCLES`` cycles in a row, it is announced once, repeated after ``REMIND_AFTER_S``
while it lasts, and announced as resolved after it was absent for ``CONFIRM_CYCLES`` cycles.
The state can be saved to a file so that this also holds for a cron job that runs once.

Secrets. The monitor key and the webhook URL (it carries a token) are read from the
environment by the script and never appear in a message, a log line or an exception text:
failures are reported by type only.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from jobpdf.monitoring.alerts import (
    BAD_RESPONSE,
    CRITICAL,
    METRICS_UNAVAILABLE,
    UNREACHABLE,
    WARNING,
    Alert,
    evaluate,
)
from jobpdf.monitoring.health import HealthReport
from jobpdf.monitoring.metrics import MetricsSnapshot

log = logging.getLogger(__name__)

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
HEALTH_PATH = "/health"
METRICS_PATH = "/metrics"
KEY_HEADER = "X-API-Key"
TIMEOUT_S = 10
MAX_BODY_BYTES = 1_000_000
CONFIRM_CYCLES = 2  # cycles in a row before an alert is announced or called resolved
REMIND_AFTER_S = 6 * 3600  # repeat a still-active alert this often
WEBHOOK_TIMEOUT_S = 10
STATE_VERSION = 1

# (status, body) of a GET. Raises OSError or HTTPException when there is no HTTP answer.
Fetch = Callable[[str, dict[str, str], float], tuple[int, bytes]]
Post = Callable[[str, bytes, float], None]


# --- fetching ----------------------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: the key header must not travel to another address."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def http_get(url: str, headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return response.status, response.read(MAX_BODY_BYTES)
    except urllib.error.HTTPError as exc:  # 3xx/4xx/5xx is an answer, not a network failure
        with exc:
            return exc.code, exc.read(MAX_BODY_BYTES)


def valid_url(url: str) -> bool:
    parts = urlsplit(url)
    return (parts.scheme in ("http", "https") and bool(parts.hostname)
            and not parts.query and parts.username is None)


@dataclass(frozen=True)
class ProbeResult:
    alerts: list[Alert]
    health: HealthReport | None = None
    metrics: MetricsSnapshot | None = None


def probe(
    base_url: str,
    api_key: str | None = None,
    *,
    ignore: frozenset[str] | set[str] = frozenset(),
    fetch: Fetch = http_get,
    timeout: float = TIMEOUT_S,
) -> ProbeResult:
    """Ask ``base_url`` for its health, and for its metrics when ``api_key`` is given.
    Never raises: every failure becomes an alert."""
    base = base_url.rstrip("/")
    answer = _get_json(fetch, base + HEALTH_PATH, {}, timeout)
    if answer is None:
        return ProbeResult(_filter([Alert(UNREACHABLE, CRITICAL)], ignore))
    status, payload = answer
    health = _parse(HealthReport, payload) if status in (200, 503) else None
    if health is None:
        # A proxy error page (502, 504) is "the app is not answering"; anything else that is
        # not a report is a different kind of wrong.
        code = UNREACHABLE if status in (502, 503, 504) else BAD_RESPONSE
        return ProbeResult(_filter([Alert(code, CRITICAL)], ignore))
    metrics = None
    extra: list[Alert] = []
    if api_key:
        metrics = _fetch_metrics(fetch, base, api_key, timeout)
        if metrics is None:
            extra.append(Alert(METRICS_UNAVAILABLE, WARNING))
    alerts = evaluate(health, metrics, ignore=ignore) + _filter(extra, ignore)
    return ProbeResult(sorted(alerts, key=lambda a: (a.severity != CRITICAL, a.code)),
                       health, metrics)


def _filter(alerts: list[Alert], ignore: Any) -> list[Alert]:
    return [a for a in alerts if a.code not in ignore]


def _get_json(fetch: Fetch, url: str, headers: dict[str, str],
              timeout: float) -> tuple[int, Any] | None:
    """(status, parsed body) or None when nothing answered. A body that is not JSON parses
    to None, so the caller sees a status with no payload."""
    try:
        status, body = fetch(url, headers, timeout)
    except (OSError, http.client.HTTPException) as exc:  # refused, timed out, cut off
        log.warning("probe failed (%s)", type(exc).__name__)
        return None
    try:
        return status, json.loads(body)
    except ValueError:
        return status, None


def _parse(model: type, payload: Any) -> Any:
    try:
        return model.model_validate(payload)
    except ValidationError:
        return None


def _fetch_metrics(fetch: Fetch, base: str, api_key: str,
                   timeout: float) -> MetricsSnapshot | None:
    answer = _get_json(fetch, base + METRICS_PATH, {KEY_HEADER: api_key}, timeout)
    if answer is None or answer[0] != 200:
        return None
    return _parse(MetricsSnapshot, answer[1])


def read_settings(names: tuple[str, ...], environ: Mapping[str, str],
                  env_file: Path = ENV_FILE) -> dict[str, str]:
    """The named settings: a non-empty environment variable wins, else the ``KEY=VALUE`` line
    in ``env_file`` (a tiny reader, as for DATABASE_URL). Missing ones are left out."""
    found: dict[str, str] = {}
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        text = line.strip()
        key, sep, value = text.partition("=")
        key = key.strip()
        if sep and key in names:
            value = value.strip().strip("'\"")
            if value:
                found[key] = value
    for name in names:
        if environ.get(name):
            found[name] = environ[name]
    return found


# --- tracking ----------------------------------------------------------------------------


@dataclass
class Changes:
    """What is worth saying after one cycle."""

    new: list[Alert] = field(default_factory=list)  # became active (or got worse)
    still: list[Alert] = field(default_factory=list)  # reminders for long-lasting alerts
    resolved: list[str] = field(default_factory=list)  # codes

    def __bool__(self) -> bool:
        return bool(self.new or self.still or self.resolved)


class Tracker:
    """Remembers which alerts are active. ``state`` is plain JSON-able data."""

    def __init__(self, state: dict[str, Any] | None = None, *,
                 confirm: int = CONFIRM_CYCLES, remind_after_s: float = REMIND_AFTER_S) -> None:
        state = state if isinstance(state, dict) and state.get("version") == STATE_VERSION else {}
        self._confirm = confirm
        self._remind_after = remind_after_s
        self._seen: dict[str, int] = _int_map(state.get("seen"))  # code -> cycles in a row
        self._clear: dict[str, int] = _int_map(state.get("clear"))  # active code -> clean cycles
        self._active: dict[str, dict[str, Any]] = _active_map(state.get("active"))

    @property
    def active(self) -> list[Alert]:
        return sorted((Alert(code, e["severity"]) for code, e in self._active.items()),
                      key=lambda a: (a.severity != CRITICAL, a.code))

    def update(self, alerts: list[Alert], now: float) -> Changes:
        changes = Changes()
        current = {a.code: a for a in alerts}
        for code, alert in current.items():
            self._clear.pop(code, None)
            entry = self._active.get(code)
            if entry is None:
                self._seen[code] = self._seen.get(code, 0) + 1
                if self._seen[code] >= self._confirm:
                    self._active[code] = {"severity": alert.severity, "sent": None}
                    changes.new.append(alert)
                continue
            if alert.severity != entry["severity"]:
                worse = alert.severity == CRITICAL
                entry["severity"] = alert.severity
                if worse:
                    entry["sent"] = None
                    changes.new.append(alert)
                    continue
            if entry["sent"] is None:  # announced earlier but the message did not go out
                changes.new.append(alert)
            elif now - entry["sent"] >= self._remind_after:
                changes.still.append(alert)
        for code in [c for c in self._seen if c not in current]:
            del self._seen[code]  # a blip: it never became active
        for code in [c for c in self._active if c not in current]:
            self._clear[code] = self._clear.get(code, 0) + 1
            if self._clear[code] >= self._confirm:
                del self._active[code]
                del self._clear[code]
                changes.resolved.append(code)
        changes.new.sort(key=_order)
        changes.still.sort(key=_order)
        changes.resolved.sort()
        return changes

    def mark_sent(self, changes: Changes, now: float) -> None:
        """Call after the message for ``changes`` was delivered."""
        for alert in changes.new + changes.still:
            self._active[alert.code]["sent"] = now

    def state(self) -> dict[str, Any]:
        return {"version": STATE_VERSION, "seen": dict(self._seen), "clear": dict(self._clear),
                "active": {c: dict(e) for c, e in self._active.items()}}


def _order(alert: Alert) -> tuple[bool, str]:
    return alert.severity != CRITICAL, alert.code


def _int_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if type(v) is int and v > 0}


def _active_map(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for code, entry in value.items():
        if not isinstance(entry, dict):
            continue
        sent = entry.get("sent")
        if entry.get("severity") in (CRITICAL, WARNING) and (
                sent is None or type(sent) in (int, float)):
            result[code] = {"severity": entry["severity"], "sent": sent}
    return result


def load_state(path: Path) -> dict[str, Any]:
    """The saved state, or an empty one when the file is missing or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(path: Path, state: dict[str, Any]) -> None:
    """Write atomically, so a crash never leaves half a file. Raises OSError."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(state), encoding="utf-8")
    os.replace(temporary, path)


# --- delivery ----------------------------------------------------------------------------


def format_message(changes: Changes, label: str, active: list[Alert]) -> str:
    """The notification text: one line per alert, codes and fixed sentences only."""
    lines = [f"[{label}]"]
    for alert in changes.new:
        lines.append(f"{alert.severity.upper()} {alert.code}: {alert.message}")
    for alert in changes.still:
        lines.append(f"STILL {alert.severity.upper()} {alert.code}: {alert.message}")
    for code in changes.resolved:
        lines.append(f"RESOLVED {code}")
    if changes.resolved and not active:
        lines.append("All clear.")
    return "\n".join(lines)


def webhook_ok(url: str) -> bool:
    """Webhook URLs carry a token, so they must be https (loopback http is allowed for tests
    of a local receiver)."""
    parts = urlsplit(url)
    if not parts.hostname:
        return False
    return parts.scheme == "https" or (
        parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1", "::1"))


def post_webhook(url: str, body: bytes, timeout: float = WEBHOOK_TIMEOUT_S) -> None:
    """POST ``body`` (JSON). Raises OSError (or HTTPError, an OSError) when it did not go
    through; the message of the error is not to be printed, it contains the address."""
    request = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": "application/json"})
    with _OPENER.open(request, timeout=timeout) as response:
        response.read(MAX_BODY_BYTES)


def webhook_body(text: str) -> bytes:
    """Slack and Mattermost read ``text``, Discord reads ``content``."""
    return json.dumps({"text": text, "content": text}).encode("utf-8")
