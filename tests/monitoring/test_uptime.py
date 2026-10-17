import http.client
import json
import logging
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jobpdf.monitoring import uptime
from jobpdf.monitoring.alerts import CRITICAL, WARNING, Alert
from jobpdf.monitoring.uptime import (
    CONFIRM_CYCLES,
    MAX_BODY_BYTES,
    REMIND_AFTER_S,
    Changes,
    Tracker,
    format_message,
    http_get,
    load_state,
    post_webhook,
    probe,
    save_state,
    valid_url,
    webhook_body,
    webhook_ok,
)
from tests.monitoring.fakes import (
    FakeService,
    body,
    check,
    health,
    raw,
    service,
    snapshot,
)

BASE = "http://app:8000"
SECRET = "monitor-key-SECRET-VALUE"
DOWN = health("down", database=check("down", "database_down"))
CRIT = Alert("check.database.database_down", CRITICAL)
WARN = Alert("check.fetch.stale_fetch", WARNING)


# --- probe ------------------------------------------------------------------------------------


def test_a_healthy_service_gives_no_alerts_and_without_a_key_metrics_are_not_asked() -> None:
    fetch = service()

    result = probe(BASE, fetch=fetch)

    assert result.alerts == [] and result.health is not None and result.metrics is None
    assert [url for url, _ in fetch.calls] == [BASE + "/health"]


def test_the_key_goes_to_metrics_only_and_a_trailing_slash_is_fine() -> None:
    fetch = service(metrics=snapshot(total=100, failed=60, errors={"unexpected": 60}))

    result = probe(BASE + "/", SECRET, fetch=fetch)

    assert fetch.calls == [(BASE + "/health", {}),
                           (BASE + "/metrics", {"X-API-Key": SECRET})]
    assert result.alerts == [Alert("service_errors", CRITICAL)] and result.metrics is not None


def test_a_503_with_a_report_is_the_report_not_an_outage() -> None:
    result = probe(BASE, fetch=service(DOWN, status=503))

    assert result.alerts == [CRIT]


@pytest.mark.parametrize("error", [ConnectionRefusedError(), TimeoutError(),
                                   urllib.error.URLError("no route"),
                                   http.client.RemoteDisconnected("closed"),
                                   http.client.IncompleteRead(b"x")])
def test_no_answer_is_unreachable_and_the_log_names_the_error_type_only(error, caplog) -> None:
    fetch = FakeService({"/health": error})

    with caplog.at_level(logging.DEBUG):
        result = probe(BASE, SECRET, fetch=fetch)

    assert result.alerts == [Alert("unreachable", CRITICAL)] and result.health is None
    assert len(fetch.calls) == 1  # no point asking for metrics
    assert type(error).__name__ in caplog.text and "no route" not in caplog.text


@pytest.mark.parametrize("answer, code", [
    (raw(502, b"<html>Bad Gateway</html>"), "unreachable"),
    (raw(503, b"<html>unavailable</html>"), "unreachable"),
    (raw(504, b""), "unreachable"),
    (raw(404, b"<html>nope</html>"), "bad_response"),
    (raw(500, {"detail": "x"}), "bad_response"),
    (raw(200, b"not json"), "bad_response"),
    (raw(200, b"\xff\xfe"), "bad_response"),
    (raw(200, {"status": "ok"}), "bad_response"),
    (raw(200, [1, 2]), "bad_response"),
    (raw(200, {"status": "ok", "checks": "x", "versions": {}, "checked_at": "x",
               "duration_ms": 1}), "bad_response"),
])
def test_an_answer_that_is_not_a_health_report_is_one_critical_alert(answer, code) -> None:
    result = probe(BASE, SECRET, fetch=FakeService({"/health": answer}))

    assert result.alerts == [Alert(code, CRITICAL)]


@pytest.mark.parametrize("answer", [
    raw(401, {"error": {"code": "invalid_key"}}), raw(403, {}), raw(500, b"oops"),
    (500, body(snapshot())), (204, body(snapshot())),
    raw(200, b"not json"), raw(200, {"analyses": 1}), ConnectionResetError()])
def test_metrics_that_cannot_be_read_are_a_warning_and_health_alerts_still_count(answer) -> None:
    fetch = FakeService({"/health": (503, body(DOWN)), "/metrics": answer})

    result = probe(BASE, SECRET, fetch=fetch)

    assert result.alerts == [CRIT, Alert("metrics_unavailable", WARNING)]


def test_ignored_codes_apply_to_every_kind_of_alert() -> None:
    assert probe(BASE, fetch=FakeService({"/health": OSError()}),
                 ignore={"unreachable"}).alerts == []
    assert probe(BASE, fetch=FakeService({"/health": raw(404, b"")}),
                 ignore={"bad_response"}).alerts == []
    fetch = FakeService({"/health": (200, body(health())), "/metrics": raw(401, {})})
    assert probe(BASE, SECRET, ignore={"metrics_unavailable"}, fetch=fetch).alerts == []


def test_alerts_are_listed_critical_first_then_by_code() -> None:
    report = health("down", a=check("degraded", "x"), b=check("down", "y"),
                    c=check("degraded", "w"))
    fetch = FakeService({"/health": (503, body(report)), "/metrics": raw(401, {})})

    result = probe(BASE, SECRET, fetch=fetch)

    assert [a.code for a in result.alerts] == [
        "check.b.y", "check.a.x", "check.c.w", "metrics_unavailable"]


def test_the_address_check() -> None:
    assert valid_url("http://app:8000") and valid_url("https://jobs.example.com/api")
    for bad in ("ftp://x", "app:8000", "http://", "", "http://u:p@host", "http://h/?k=v"):
        assert not valid_url(bad), bad


# --- the real HTTP functions, against a local server ------------------------------------------


class Handler(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def log_message(self, *args) -> None:
        pass

    def _answer(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        Handler.seen.append({"method": self.command, "path": self.path,
                             "key": self.headers.get("X-API-Key"),
                             "type": self.headers.get("Content-Type"),
                             "body": self.rfile.read(length)})
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/elsewhere")
            self.end_headers()
            return
        status = {"/ok": 200, "/down": 503, "/missing": 404, "/boom": 500,
                  "/bigboom": 500}.get(self.path, 200)
        big = self.path in ("/big", "/bigboom")
        payload = b"x" * (MAX_BODY_BYTES + 500) if big else b'{"a": 1}'
        self.send_response(status)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = _answer


@pytest.fixture
def server():
    Handler.seen = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


@pytest.mark.parametrize("path, status", [("/ok", 200), ("/down", 503), ("/missing", 404),
                                          ("/boom", 500)])
def test_http_get_returns_every_http_answer_with_its_body(server, path, status) -> None:
    assert http_get(server + path, {}, 5) == (status, b'{"a": 1}')


def test_http_get_sends_the_given_headers_and_caps_the_body(server) -> None:
    _, content = http_get(server + "/big", {"X-API-Key": SECRET}, 5)

    assert len(content) == MAX_BODY_BYTES
    assert Handler.seen[0]["key"] == SECRET and Handler.seen[0]["method"] == "GET"


def test_http_get_caps_the_body_of_an_error_answer_too(server) -> None:
    status, content = http_get(server + "/bigboom", {}, 5)

    assert status == 500 and len(content) == MAX_BODY_BYTES


def test_http_get_does_not_follow_a_redirect_so_the_key_cannot_travel(server) -> None:
    status, _ = http_get(server + "/redirect", {"X-API-Key": SECRET}, 5)

    assert status == 302 and len(Handler.seen) == 1


def test_http_get_raises_an_oserror_when_nothing_listens(server) -> None:
    port = server.rsplit(":", 1)[1]
    with pytest.raises(OSError):
        http_get(f"http://127.0.0.1:{int(port) + 1}/x", {}, 2)  # a neighbouring, unused port


def test_probe_with_the_real_fetch_reports_a_server_that_is_not_ours(server) -> None:
    assert probe(server, SECRET).alerts == [Alert("bad_response", CRITICAL)]


def test_post_webhook_sends_json_and_raises_for_a_refusal_or_a_redirect(server) -> None:
    post_webhook(server + "/ok", b'{"text": "hi"}', 5)
    assert Handler.seen[0]["method"] == "POST" and Handler.seen[0]["type"] == "application/json"
    assert Handler.seen[0]["body"] == b'{"text": "hi"}'

    for path in ("/boom", "/missing", "/redirect"):
        with pytest.raises(OSError):
            post_webhook(server + path, b"{}", 5)


def test_the_webhook_body_has_the_text_for_slack_and_for_discord() -> None:
    assert json.loads(webhook_body("a\nb")) == {"text": "a\nb", "content": "a\nb"}


@pytest.mark.parametrize("url, ok", [
    ("https://hooks.example.com/T0/B0/xyz", True), ("http://localhost:9000/hook", True),
    ("http://127.0.0.1/hook", True), ("http://[::1]:9000/hook", True),
    ("http://hooks.example.com/x", False), ("ftp://hooks.example.com/x", False),
    ("https://", False), ("hooks.example.com", False), ("", False)])
def test_the_webhook_must_be_https_except_on_this_machine(url, ok) -> None:
    assert webhook_ok(url) is ok


# --- the tracker -----------------------------------------------------------------------------


def run(tracker: Tracker, *alerts: Alert, now: float = 0) -> Changes:
    return tracker.update(list(alerts), now)


def test_an_alert_starts_only_after_it_was_seen_in_enough_cycles_in_a_row() -> None:
    tracker = Tracker()
    assert CONFIRM_CYCLES == 2

    assert not run(tracker, CRIT)
    changes = run(tracker, CRIT)

    assert changes.new == [CRIT] and tracker.active == [CRIT]


def test_a_blip_never_becomes_an_alert() -> None:
    tracker = Tracker()
    run(tracker, CRIT)
    run(tracker)  # gone again: the count starts over
    assert not run(tracker, CRIT)
    assert tracker.active == []


def test_with_confirm_one_it_starts_at_once() -> None:
    tracker = Tracker(confirm=1)

    assert run(tracker, CRIT).new == [CRIT]


def test_a_sent_alert_is_not_announced_again_until_the_reminder_is_due() -> None:
    tracker = Tracker(confirm=1)
    first = run(tracker, CRIT, now=1000)
    tracker.mark_sent(first, 1000)

    assert not run(tracker, CRIT, now=1000 + REMIND_AFTER_S - 1)
    reminder = run(tracker, CRIT, now=1000 + REMIND_AFTER_S)
    assert reminder.still == [CRIT] and reminder.new == []
    tracker.mark_sent(reminder, 1000 + REMIND_AFTER_S)
    assert not run(tracker, CRIT, now=1000 + REMIND_AFTER_S + 60)


def test_an_alert_that_could_not_be_sent_is_announced_again_every_cycle_until_it_is() -> None:
    tracker = Tracker(confirm=1)

    assert run(tracker, CRIT, now=1).new == [CRIT]
    again = run(tracker, CRIT, now=2)  # mark_sent was not called: the post failed
    assert again.new == [CRIT]
    tracker.mark_sent(again, 2)
    assert not run(tracker, CRIT, now=3)


def test_a_warning_that_gets_worse_is_announced_as_new_and_a_better_one_is_quiet() -> None:
    tracker = Tracker(confirm=1)
    tracker.mark_sent(run(tracker, Alert("x", WARNING), now=0), 0)

    worse = run(tracker, Alert("x", CRITICAL), now=1)
    assert worse.new == [Alert("x", CRITICAL)]
    tracker.mark_sent(worse, 1)

    assert not run(tracker, Alert("x", WARNING), now=2)
    assert tracker.active == [Alert("x", WARNING)]


def test_a_worse_alert_whose_message_was_not_delivered_is_announced_again() -> None:
    tracker = Tracker(confirm=1)
    tracker.mark_sent(run(tracker, Alert("x", WARNING), now=0), 0)

    assert run(tracker, Alert("x", CRITICAL), now=1).new == [Alert("x", CRITICAL)]
    assert run(tracker, Alert("x", CRITICAL), now=2).new == [Alert("x", CRITICAL)]  # not sent


def test_reminders_and_resolutions_come_in_a_fixed_order() -> None:
    tracker = Tracker(confirm=1)
    alerts = [Alert("b", WARNING), CRIT, Alert("a", WARNING), Alert("c", CRITICAL)]
    tracker.mark_sent(run(tracker, *alerts, now=0), 0)

    reminders = run(tracker, *alerts, now=REMIND_AFTER_S)
    assert reminders.still == [Alert("c", CRITICAL), CRIT, Alert("a", WARNING),
                               Alert("b", WARNING)]
    tracker.mark_sent(reminders, REMIND_AFTER_S)

    assert run(tracker, now=REMIND_AFTER_S + 1).resolved == ["a", "b", "c", CRIT.code]


def test_an_alert_is_resolved_after_enough_clean_cycles_and_a_flap_is_not() -> None:
    tracker = Tracker()
    run(tracker, CRIT)
    tracker.mark_sent(run(tracker, CRIT), 0)

    assert not run(tracker)  # one clean cycle: still active
    assert tracker.active == [CRIT]
    assert not run(tracker, CRIT)  # it came back: the clean count starts over
    assert not run(tracker)
    resolved = run(tracker)

    assert resolved.resolved == [CRIT.code] and tracker.active == []


def test_changes_are_sorted_critical_first() -> None:
    tracker = Tracker(confirm=1)

    changes = run(tracker, WARN, Alert("b", WARNING), CRIT)

    assert changes.new == [CRIT, Alert("b", WARNING), WARN]


def test_the_state_survives_a_restart() -> None:
    first = Tracker()
    run(first, CRIT, WARN)
    run(first, CRIT)  # CRIT is active, WARN was a blip
    first.mark_sent(Changes(new=[CRIT]), 500)

    second = Tracker(json.loads(json.dumps(first.state())))

    assert second.active == [CRIT]
    assert not run(second, CRIT, now=501)  # already sent, not due
    assert run(second, CRIT, now=500 + REMIND_AFTER_S).still == [CRIT]


@pytest.mark.parametrize("state", [
    None, [], "x", 3, {}, {"version": 99}, {"version": 1, "seen": [], "active": "x"},
    {"version": 99, "seen": {"a": 1}, "active": {"a": {"severity": "critical", "sent": None}}},
    {"seen": {"a": 1}, "active": {"a": {"severity": "critical", "sent": None}}},
    {"version": 1, "seen": {"a": "x", "b": -1, "c": True}, "clear": {"a": 1.5},
     "active": {"a": {"severity": "bogus", "sent": None}, "b": 3,
                "c": {"severity": "critical", "sent": "never"}}}])
def test_a_damaged_or_foreign_state_is_read_as_empty(state) -> None:
    tracker = Tracker(state)

    assert tracker.active == []
    assert tracker.state()["seen"] == {} and tracker.state()["active"] == {}


def test_a_state_with_one_good_entry_keeps_it() -> None:
    tracker = Tracker({"version": 1, "active": {
        "keep": {"severity": "warning", "sent": 5.5}, "drop": {"severity": "bogus", "sent": 1}}})

    assert tracker.active == [Alert("keep", WARNING)]


# --- messages and the state file ---------------------------------------------------------------


def test_the_message_has_one_line_per_alert_and_says_when_all_is_clear() -> None:
    changes = Changes(new=[CRIT], still=[WARN], resolved=["unreachable"])

    text = format_message(changes, "JobPDF prod", [CRIT, WARN])

    lines = text.split("\n")
    assert lines[0] == "[JobPDF prod]"
    assert lines[1].startswith("CRITICAL check.database.database_down: ")
    assert lines[2].startswith("STILL WARNING check.fetch.stale_fetch: ")
    assert lines[3] == "RESOLVED unreachable" and len(lines) == 4
    assert format_message(Changes(resolved=["x"]), "L", []).endswith("RESOLVED x\nAll clear.")


def test_all_clear_is_said_only_when_something_was_resolved_and_nothing_is_left() -> None:
    assert "All clear." not in format_message(Changes(new=[CRIT]), "L", [])
    assert "All clear." not in format_message(Changes(resolved=["x"]), "L", [CRIT])


def test_a_save_that_fails_midway_leaves_the_old_state_intact(tmp_path, monkeypatch) -> None:
    path = tmp_path / "state.json"
    save_state(path, {"version": 1, "a": 1})

    def refuse(*args):
        raise PermissionError("no")

    monkeypatch.setattr(uptime.os, "replace", refuse)
    with pytest.raises(PermissionError):
        save_state(path, {"version": 1, "a": 2})

    assert load_state(path) == {"version": 1, "a": 1}


def test_the_state_file_round_trips_and_is_replaced_atomically(tmp_path) -> None:
    path = tmp_path / "state.json"

    save_state(path, {"version": 1, "a": 1})
    save_state(path, {"version": 1, "a": 2})

    assert load_state(path) == {"version": 1, "a": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]  # no .tmp left behind


def test_a_missing_or_broken_state_file_is_empty(tmp_path) -> None:
    assert load_state(tmp_path / "nope.json") == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert load_state(broken) == {}
    broken.write_bytes(b"\xff\xfe")
    assert load_state(broken) == {}


def test_saving_into_a_missing_folder_raises_oserror(tmp_path) -> None:
    with pytest.raises(OSError):
        save_state(tmp_path / "no" / "state.json", {})


def test_the_module_keeps_its_thresholds_in_constants() -> None:
    assert uptime.CONFIRM_CYCLES >= 2 and uptime.REMIND_AFTER_S == 6 * 3600


# --- settings from the environment and .env ------------------------------------------------------


NAMES = ("A", "B", "C")


def test_settings_come_from_the_env_file_and_a_real_variable_wins(tmp_path) -> None:
    file = tmp_path / ".env"
    file.write_text("# A=commented\nA=from file\nB = 'quoted'\nC=\nOTHER=x\n  D=1\n"
                    "export A=no\nbroken line\n", encoding="utf-8")

    assert uptime.read_settings(NAMES, {}, file) == {"A": "from file", "B": "quoted"}
    assert uptime.read_settings(NAMES, {"A": "real", "C": "set"}, file) == {
        "A": "real", "B": "quoted", "C": "set"}
    assert uptime.read_settings(NAMES, {"A": "", "OTHER": "y"}, file)["A"] == "from file"


def test_a_missing_env_file_is_not_an_error(tmp_path) -> None:
    assert uptime.read_settings(NAMES, {"B": "1"}, tmp_path / "none") == {"B": "1"}
    assert uptime.read_settings(NAMES, {}, tmp_path / "none") == {}


def test_the_default_env_file_is_the_repository_one() -> None:
    assert uptime.ENV_FILE.name == ".env" and (uptime.ENV_FILE.parent / "pyproject.toml").is_file()
