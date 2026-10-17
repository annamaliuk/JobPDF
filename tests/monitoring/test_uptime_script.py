import importlib.util
import json
import urllib.error
from pathlib import Path

import pytest

from jobpdf.monitoring.uptime import REMIND_AFTER_S
from tests.monitoring.fakes import FakeService, body, check, health, healthy, raw, snapshot

ROOT = Path(__file__).resolve().parents[2]
KEY = "monitor-key-SECRET-VALUE"
HOOK = "https://hooks.example.com/services/T000/B000/SECRET-TOKEN"
SECRETS = (KEY, HOOK, "SECRET-TOKEN")
DOWN = health("down", database=check("down", "database_down"))
STALE = health("degraded", fetch=check("degraded", "stale_fetch"))


@pytest.fixture(scope="module")
def script():
    path = ROOT / "scripts" / "uptime_check.py"
    spec = importlib.util.spec_from_file_location("uptime_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Hook:
    """A webhook that remembers what was posted; ``fail`` makes the next posts raise."""

    def __init__(self) -> None:
        self.posts: list[tuple[str, dict]] = []
        self.timeouts: list[float] = []
        self.fail: Exception | None = None

    def __call__(self, url: str, payload: bytes, timeout: float) -> None:
        if self.fail:
            raise self.fail
        self.timeouts.append(timeout)
        self.posts.append((url, json.loads(payload)))

    @property
    def texts(self) -> list[str]:
        return [p["text"] for _, p in self.posts]


class Timeline:
    """A fetch whose answer changes per cycle: ``steps`` is a list of FakeService."""

    def __init__(self, *steps: FakeService) -> None:
        self.steps = list(steps)
        self.index = 0

    def advance(self) -> None:
        self.index = min(self.index + 1, len(self.steps) - 1)

    def __call__(self, url, headers, timeout):
        return self.steps[self.index](url, headers, timeout)


def up() -> FakeService:
    return FakeService({"/health": (200, body(healthy())), "/metrics": (200, body(snapshot()))})


def down() -> FakeService:
    return FakeService({"/health": (503, body(DOWN)), "/metrics": (200, body(snapshot()))})


def unreachable() -> FakeService:
    return FakeService({"/health": ConnectionRefusedError(), "/metrics": ConnectionRefusedError()})


def run(script, argv=(), *, env=None, fetch=None, post=None, **kwargs) -> int:
    return script.main(list(argv), environ=env or {}, fetch=fetch or up(),
                       post=post or Hook(), clock=kwargs.pop("clock", lambda: 1000.0),
                       sleep=kwargs.pop("sleep", lambda s: None), **kwargs)


# --- one run -----------------------------------------------------------------------------------


def test_a_healthy_service_prints_ok_and_exits_0(script, capsys) -> None:
    assert run(script) == 0

    out = capsys.readouterr().out.splitlines()
    assert out == ["monitor: metrics key missing, webhook missing", "ok"]


@pytest.mark.parametrize("service, code, line", [
    (down(), 2, "CRITICAL check.database.database_down: "),
    (unreachable(), 2, "CRITICAL unreachable: "),
    (FakeService({"/health": (200, body(STALE))}), 1, "WARNING check.fetch.stale_fetch: "),
    (FakeService({"/health": raw(404, b"x")}), 2, "CRITICAL bad_response: ")])
def test_the_exit_code_is_the_worst_alert_right_now(script, capsys, service, code, line) -> None:
    assert run(script, fetch=service) == code
    assert line in capsys.readouterr().out


def test_a_single_run_without_state_does_not_post_and_says_so(script, capsys) -> None:
    hook = Hook()

    assert run(script, env={"ALERT_WEBHOOK_URL": HOOK}, fetch=down(), post=hook) == 2

    assert hook.posts == []
    assert "webhook set (not used without --state or --loop)" in capsys.readouterr().out


def test_the_key_is_sent_to_metrics_and_the_address_comes_from_the_flag_or_the_setting(
        script) -> None:
    fetch = up()
    run(script, env={"JOBPDF_MONITOR_API_KEY": KEY, "MONITOR_URL": "http://one:1"}, fetch=fetch)
    assert fetch.calls[0][0] == "http://one:1/health"
    assert fetch.calls[1] == ("http://one:1/metrics", {"X-API-Key": KEY})

    other = up()
    run(script, ["--url", "http://two:2"], env={"MONITOR_URL": "http://one:1"}, fetch=other)
    assert other.calls[0][0] == "http://two:2/health"

    default = up()
    run(script, fetch=default)
    assert default.calls[0][0] == "http://127.0.0.1:8000/health"


def test_ignored_codes_come_from_the_flag_and_from_the_setting(script, capsys) -> None:
    fetch = FakeService({"/health": (200, body(STALE))})

    assert run(script, ["--ignore", "check.fetch.stale_fetch"], fetch=fetch) == 0
    assert run(script, env={"MONITOR_IGNORE": " x , check.fetch.stale_fetch,"}, fetch=fetch) == 0
    assert run(script, ["--ignore", "other"], fetch=fetch) == 1


def test_metrics_alerts_are_reported_when_a_key_is_set(script, capsys) -> None:
    fetch = FakeService({"/health": (200, body(healthy())), "/metrics": (200, body(
        snapshot(total=100, failed=30, errors={"extraction:llm_error": 30})))})

    assert run(script, env={"JOBPDF_MONITOR_API_KEY": KEY}, fetch=fetch) == 1
    assert "WARNING service_errors: " in capsys.readouterr().out


# --- settings ----------------------------------------------------------------------------------


@pytest.mark.parametrize("argv, env, fragment", [
    ([], {"MONITOR_URL": "not a url"}, "service address"),
    (["--url", "http://user:pw@host"], {}, "service address"),
    ([], {"ALERT_WEBHOOK_URL": "http://hooks.example.com/x"}, "must be an https address"),
    ([], {"ALERT_LABEL": "bad <label>"}, "ALERT_LABEL"),
    ([], {"ALERT_LABEL": "x" * 41}, "ALERT_LABEL"),
    (["--loop", "--interval", "9"], {}, "at least 10"),
    (["--confirm", "0"], {}, "--confirm must be at least 1"),
])
def test_wrong_settings_exit_3_and_say_which_without_echoing_secrets(
        script, capsys, argv, env, fragment) -> None:
    assert run(script, argv, env=env) == 3

    out = capsys.readouterr().out
    assert fragment in out and "hooks.example.com" not in out and "pw" not in out


def test_wrong_flags_also_exit_3_because_2_means_critical(script, capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        run(script, ["--nonsense"])

    assert raised.value.code == 3
    assert "unrecognized arguments" in capsys.readouterr().err


def test_a_custom_label_names_the_site_in_every_message(script, tmp_path) -> None:
    hook = Hook()
    state = tmp_path / "s.json"
    run(script, ["--state", str(state), "--confirm", "1"],
        env={"ALERT_WEBHOOK_URL": HOOK, "ALERT_LABEL": "JobPDF prod"}, fetch=down(), post=hook)

    assert hook.texts[0].startswith("[JobPDF prod]\n")


# --- runs with a state file (cron) --------------------------------------------------------------


def test_cron_runs_confirm_before_they_exit_non_zero_or_post(script, tmp_path, capsys) -> None:
    hook, state = Hook(), str(tmp_path / "s.json")
    env = {"ALERT_WEBHOOK_URL": HOOK}

    first = run(script, ["--state", state], env=env, fetch=down(), post=hook)
    assert first == 0 and hook.posts == []
    assert "CRITICAL check.database.database_down" in capsys.readouterr().out  # still printed

    second = run(script, ["--state", state], env=env, fetch=down(), post=hook)
    assert second == 2 and len(hook.posts) == 1
    assert hook.posts[0][0] == HOOK and "CRITICAL check.database.database_down" in hook.texts[0]
    assert hook.timeouts == [10]
    assert "[JobPDF]" not in capsys.readouterr().out  # a run prints what it sees, not the post

    third = run(script, ["--state", state], env=env, fetch=down(), post=hook)
    assert third == 2 and len(hook.posts) == 1  # not repeated

    later = run(script, ["--state", state], env=env, fetch=down(), post=hook,
                clock=lambda: 1000.0 + REMIND_AFTER_S)
    assert later == 2 and len(hook.posts) == 2 and hook.texts[1].startswith("[JobPDF]\nSTILL")


def test_cron_resolution_is_posted_after_two_clean_runs(script, tmp_path) -> None:
    hook, state = Hook(), str(tmp_path / "s.json")
    env = {"ALERT_WEBHOOK_URL": HOOK}
    for _ in range(2):
        run(script, ["--state", state], env=env, fetch=down(), post=hook)

    assert run(script, ["--state", state], env=env, fetch=up(), post=hook) == 2  # still active
    assert run(script, ["--state", state], env=env, fetch=up(), post=hook) == 0

    assert len(hook.posts) == 2
    assert hook.texts[1] == "[JobPDF]\nRESOLVED check.database.database_down\nAll clear."


def test_the_state_file_is_written_and_a_broken_one_is_ignored(script, tmp_path) -> None:
    state = tmp_path / "s.json"
    state.write_text("{broken")

    run(script, ["--state", str(state)], fetch=down())

    assert json.loads(state.read_text())["seen"] == {"check.database.database_down": 1}


def test_an_unwritable_state_file_is_reported_by_type_and_the_run_goes_on(
        script, tmp_path, capsys) -> None:
    state = tmp_path / "missing-folder" / "s.json"

    assert run(script, ["--state", str(state)], fetch=down()) == 0

    assert "state not saved (FileNotFoundError)" in capsys.readouterr().out


# --- the loop ----------------------------------------------------------------------------------


def test_the_loop_sleeps_between_checks_and_posts_once_per_change(script, capsys) -> None:
    hook, sleeps = Hook(), []
    timeline = Timeline(up(), down(), down(), down(), up(), up(), up())

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        timeline.advance()

    code = run(script, ["--loop", "--interval", "30"], env={"ALERT_WEBHOOK_URL": HOOK},
               fetch=timeline, post=hook, sleep=sleep, cycles=7)

    assert code == 0 and sleeps == [30] * 6
    assert len(hook.posts) == 2
    assert "CRITICAL check.database.database_down" in hook.texts[0]
    assert hook.texts[1].endswith("RESOLVED check.database.database_down\nAll clear.")
    out = capsys.readouterr().out
    assert out.count("CRITICAL check.database.database_down") == 1  # printed on change only
    assert "All clear." in out


def test_the_default_interval_is_a_minute(script) -> None:
    sleeps: list[float] = []

    run(script, ["--loop"], sleep=sleeps.append, cycles=3)

    assert sleeps == [60, 60]


def test_the_loop_exit_code_is_the_state_when_it_stops(script) -> None:
    assert run(script, ["--loop"], fetch=down(), cycles=2) == 2
    assert run(script, ["--loop"], fetch=down(), cycles=1) == 0  # not confirmed yet


def test_the_loop_without_a_webhook_prints_each_change_once(script, capsys) -> None:
    run(script, ["--loop"], fetch=unreachable(), cycles=5)

    assert capsys.readouterr().out.count("CRITICAL unreachable") == 1


def test_a_failed_post_is_retried_next_cycle_and_not_repeated_once_delivered(
        script, capsys) -> None:
    hook = Hook()
    hook.fail = urllib.error.URLError(f"cannot reach {HOOK}")
    cycle = []

    def sleep(seconds: float) -> None:
        cycle.append(1)
        if len(cycle) == 1:
            hook.fail = None  # the receiver is back after the first cycle

    run(script, ["--loop", "--confirm", "1"], env={"ALERT_WEBHOOK_URL": HOOK},
        fetch=down(), post=hook, sleep=sleep, cycles=4)

    assert len(hook.posts) == 1  # delivered on the retry, then not again
    assert "CRITICAL check.database.database_down" in hook.texts[0]
    assert capsys.readouterr().out.count("webhook failed (URLError)") == 1


def test_webhook_errors_are_named_by_type_only(script, capsys) -> None:
    hook = Hook()
    hook.fail = urllib.error.URLError(f"cannot reach {HOOK}")

    run(script, ["--loop", "--confirm", "1"], env={"ALERT_WEBHOOK_URL": HOOK,
        "JOBPDF_MONITOR_API_KEY": KEY}, fetch=down(), post=hook, cycles=1)

    out = capsys.readouterr().out
    assert "webhook failed (URLError)" in out
    assert not any(secret in out for secret in SECRETS)


@pytest.mark.parametrize("failure", [OSError("x"), ConnectionResetError(), TimeoutError()])
def test_any_network_failure_of_the_webhook_is_survived(script, capsys, failure) -> None:
    hook = Hook()
    hook.fail = failure

    assert run(script, ["--loop", "--confirm", "1"], env={"ALERT_WEBHOOK_URL": HOOK},
               fetch=down(), post=hook, cycles=1) == 2
    assert f"webhook failed ({type(failure).__name__})" in capsys.readouterr().out


def test_secrets_never_reach_the_output_in_any_mode(script, capsys, tmp_path) -> None:
    env = {"ALERT_WEBHOOK_URL": HOOK, "JOBPDF_MONITOR_API_KEY": KEY}
    run(script, env=env, fetch=down())
    run(script, ["--loop", "--confirm", "1"], env=env, fetch=unreachable(), cycles=2)
    run(script, ["--state", str(tmp_path / "s.json"), "--confirm", "1"], env=env, fetch=down())

    out = capsys.readouterr()
    assert not any(secret in out.out + out.err for secret in SECRETS)
    assert not any(secret in (tmp_path / "s.json").read_text() for secret in SECRETS)


def test_settings_are_read_from_the_env_file_and_a_real_variable_wins(
        script, tmp_path, monkeypatch) -> None:
    for name in script.SETTINGS:
        monkeypatch.delenv(name, raising=False)
    file = tmp_path / ".env"
    file.write_text(f"MONITOR_URL=http://filehost:1\nJOBPDF_MONITOR_API_KEY={KEY}\n")
    fetch = up()

    script.main([], environ=None, fetch=fetch, env_file=file)
    assert fetch.calls[0][0] == "http://filehost:1/health"
    assert fetch.calls[1][1] == {"X-API-Key": KEY}

    monkeypatch.setenv("MONITOR_URL", "http://envhost:2")
    other = up()
    script.main([], environ=None, fetch=other, env_file=file)
    assert other.calls[0][0] == "http://envhost:2/health"
