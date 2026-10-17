"""Check that the service is up and healthy, and tell someone when it is not (JM-40).

Asks /health (and /metrics when a monitor key is set), applies the alert rules in
``jobpdf/monitoring/alerts.py`` and prints the result. With ALERT_WEBHOOK_URL set it also posts
to a Slack/Mattermost/Discord webhook when an alert starts, every 6 hours while it lasts, and
when it is resolved. An alert starts only after it was seen in 2 cycles in a row, so one slow
answer does not wake anybody.

Usage (from the repo root):
    uv run python scripts/uptime_check.py                       # once; exit code = worst alert now
    uv run python scripts/uptime_check.py --url https://jobs.example.com
    uv run python scripts/uptime_check.py --loop                # every 60 s (the compose service)
    uv run python scripts/uptime_check.py --state data/monitor_state.json
        # for cron: remembers alerts between runs so the confirmation and the webhook work

Settings (environment or .env; secrets are reported as "set" or "missing", never printed):
    MONITOR_URL              default http://127.0.0.1:8000 (the compose service uses http://app:8000)
    JOBPDF_MONITOR_API_KEY   an API key for /metrics; without it only /health is checked
    ALERT_WEBHOOK_URL        https address of the webhook
    ALERT_LABEL              name shown in messages, default JobPDF (e.g. "JobPDF prod")
    MONITOR_IGNORE           comma-separated alert codes to ignore, e.g. check.fetch.no_cache

Exit code (without --loop): 0 no alert, 1 warnings, 2 a critical alert, 3 wrong usage or settings.
Without --state a single run cannot confirm anything, so it reports what it sees right now and
does not post to the webhook. With --state an alert counts (exit code, webhook) once it was seen
in --confirm runs in a row; what a run sees right now is always printed.
"""

from __future__ import annotations

import argparse
import http.client
import os
import re
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NoReturn

from jobpdf.monitoring.alerts import CRITICAL, WARNING, Alert, worst
from jobpdf.monitoring.uptime import (
    CONFIRM_CYCLES,
    ENV_FILE,
    WEBHOOK_TIMEOUT_S,
    Changes,
    Fetch,
    Post,
    Tracker,
    format_message,
    http_get,
    load_state,
    post_webhook,
    probe,
    read_settings,
    save_state,
    valid_url,
    webhook_body,
    webhook_ok,
)

DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_LABEL = "JobPDF"
DEFAULT_INTERVAL_S = 60
MIN_INTERVAL_S = 10
EXIT_OK, EXIT_WARNING, EXIT_CRITICAL, EXIT_USAGE = 0, 1, 2, 3
EXIT_CODES = {None: EXIT_OK, WARNING: EXIT_WARNING, CRITICAL: EXIT_CRITICAL}

URL_ENV = "MONITOR_URL"
KEY_ENV = "JOBPDF_MONITOR_API_KEY"
WEBHOOK_ENV = "ALERT_WEBHOOK_URL"
LABEL_ENV = "ALERT_LABEL"
IGNORE_ENV = "MONITOR_IGNORE"
SETTINGS = (URL_ENV, KEY_ENV, WEBHOOK_ENV, LABEL_ENV, IGNORE_ENV)
LABEL = re.compile(r"[A-Za-z0-9 ._-]{1,40}")


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        print(f"uptime_check.py: error: {message}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)  # 2 means "critical" here, so usage errors use 3


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", help=f"base address of the service (env {URL_ENV})")
    parser.add_argument("--loop", action="store_true", help="keep checking until stopped")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_S,
                        help=f"seconds between checks with --loop (at least {MIN_INTERVAL_S})")
    parser.add_argument("--state", type=Path, help="file that remembers alerts between runs")
    parser.add_argument("--confirm", type=int, default=CONFIRM_CYCLES,
                        help="cycles in a row before an alert starts or ends")
    parser.add_argument("--ignore", action="append", default=[], metavar="CODE",
                        help="alert code to ignore (repeatable; env MONITOR_IGNORE)")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    fetch: Fetch = http_get,
    post: Post = post_webhook,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    cycles: int | None = None,
    env_file: Path = ENV_FILE,
) -> int:
    """``cycles`` stops a ``--loop`` after that many checks (for tests); None runs on."""
    env = read_settings(SETTINGS, os.environ if environ is None else environ, env_file)
    args = _parser().parse_args(argv)
    url = args.url or env.get(URL_ENV) or DEFAULT_URL
    webhook = env.get(WEBHOOK_ENV) or None
    key = env.get(KEY_ENV) or None
    label = env.get(LABEL_ENV) or DEFAULT_LABEL
    ignore = {c.strip() for c in [*args.ignore, *(env.get(IGNORE_ENV) or "").split(",")]}

    problem = _settings_problem(url, webhook, label, args.interval, args.confirm)
    if problem:
        print(problem)
        return EXIT_USAGE

    persistent = bool(args.loop or args.state)
    tracker = Tracker(load_state(args.state) if args.state else None,
                      confirm=args.confirm if persistent else 1)
    sending = bool(webhook) and persistent
    print(f"monitor: metrics key {'set' if key else 'missing'}, "
          f"webhook {'set' if webhook else 'missing'}"
          + ("" if sending or not webhook else " (not used without --state or --loop)"))

    done = 0
    while True:
        result = probe(url, key, ignore=ignore, fetch=fetch)
        now = clock()
        changes = tracker.update(result.alerts, now)
        if args.loop:
            _announce(changes, tracker, label, webhook, post, now)
        else:
            _print_now(result.alerts)
            if sending:
                _announce(changes, tracker, label, webhook, post, now, quiet=True)
        if args.state:
            _save(args.state, tracker)
        done += 1
        if not args.loop:
            return EXIT_CODES[worst(tracker.active)]
        if cycles is not None and done >= cycles:
            return EXIT_CODES[worst(tracker.active)]
        sleep(args.interval)


def _settings_problem(url: str, webhook: str | None, label: str, interval: int,
                      confirm: int) -> str | None:
    if not valid_url(url):
        return "the service address must be an http(s) address without credentials or query"
    if webhook and not webhook_ok(webhook):
        return f"{WEBHOOK_ENV} must be an https address"
    if not LABEL.fullmatch(label):
        return f"{LABEL_ENV} may use letters, digits, spaces, dots, dashes and underscores (40)"
    if interval < MIN_INTERVAL_S:
        return f"--interval must be at least {MIN_INTERVAL_S} seconds"
    if confirm < 1:
        return "--confirm must be at least 1"
    return None


def _print_now(alerts: list[Alert]) -> None:
    if not alerts:
        print("ok")
    for alert in alerts:
        print(f"{alert.severity.upper()} {alert.code}: {alert.message}")


def _announce(changes: Changes, tracker: Tracker, label: str, webhook: str | None,
              post: Post, now: float, *, quiet: bool = False) -> None:
    """Print and, when there is a webhook, post what changed. A message that could not be
    posted stays due: the tracker only marks it sent after delivery."""
    if not changes:
        return
    text = format_message(changes, label, tracker.active)
    if not quiet:
        print(text)
    if webhook is None:
        tracker.mark_sent(changes, now)  # printed is as delivered as it gets
        return
    try:
        post(webhook, webhook_body(text), WEBHOOK_TIMEOUT_S)
    except (OSError, http.client.HTTPException) as exc:
        print(f"webhook failed ({type(exc).__name__})")  # the text holds the address
        return
    tracker.mark_sent(changes, now)


def _save(path: Path, tracker: Tracker) -> None:
    try:
        save_state(path, tracker.state())
    except OSError as exc:
        print(f"state not saved ({type(exc).__name__})")


if __name__ == "__main__":
    sys.exit(main())
