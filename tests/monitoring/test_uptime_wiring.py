"""The monitor's pieces must agree: compose, .env.example, the README and the code."""

from pathlib import Path

import pytest

from jobpdf.monitoring import alerts, uptime
from jobpdf.monitoring.alerts import MESSAGES

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")
ENV_EXAMPLE = (ROOT / ".env.example").read_text(encoding="utf-8")
SETTINGS = ("JOBPDF_MONITOR_API_KEY", "ALERT_WEBHOOK_URL", "ALERT_LABEL", "MONITOR_IGNORE",
            "MONITOR_URL")


@pytest.fixture(scope="module")
def compose() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))["services"]


def test_the_monitor_service_runs_the_loop_with_a_state_file_and_only_its_own_settings(
        compose) -> None:
    service = compose["monitor"]

    assert service["profiles"] == ["monitor"]
    assert service["command"] == ["python", "scripts/uptime_check.py", "--loop",
                                  "--state", "data/monitor_state.json"]
    assert set(service["environment"]) == {
        "MONITOR_URL", "JOBPDF_MONITOR_API_KEY", "ALERT_WEBHOOK_URL", "ALERT_LABEL",
        "MONITOR_IGNORE"}  # no database address, no LLM key
    assert service["environment"]["MONITOR_URL"] == "http://app:8000"
    assert (ROOT / "scripts" / "uptime_check.py").is_file()


def test_the_monitor_does_not_wait_for_a_healthy_app_and_is_locked_down(compose) -> None:
    service = compose["monitor"]

    assert service["depends_on"] == ["app"]  # a list: started, not healthy
    assert service["restart"] == "unless-stopped"
    assert service["read_only"] is True and service["cap_drop"] == ["ALL"]
    assert "ports" not in service


def test_the_monitor_url_names_the_port_the_app_listens_on(compose) -> None:
    assert compose["app"]["expose"] == ["8000"]


def test_the_env_example_lists_every_setting() -> None:
    for name in SETTINGS[:4]:
        assert f"{name}=" in ENV_EXAMPLE, name


def test_the_readme_documents_every_alert_code_and_setting() -> None:
    for code in MESSAGES:
        assert f"`{code}`" in README, code
    assert "`check.<name>.<code>`" in README
    for name in SETTINGS:
        assert name in README, name
    assert "--profile monitor" in README and "scripts/uptime_check.py" in README


def test_the_readme_states_the_thresholds_that_the_code_uses() -> None:
    section = README[README.index("## 8. Monitoring and alerts"):]

    assert f"warning at {int(alerts.FAULT_RATE_WARNING * 100)}%" in section
    assert f"critical at {int(alerts.FAULT_RATE_CRITICAL * 100)}%" in section
    assert f"over {alerts.SLOW_P95_MS // 1000} seconds" in section
    assert f"over {alerts.TOKENS_PER_ANALYSIS_MAX:,}".replace(",", " ") in section
    assert f"at least {alerts.MIN_ANALYSES} analyses" in section
    assert f"every {uptime.REMIND_AFTER_S // 3600} hours" in section
    assert "twice in a row" in section and uptime.CONFIRM_CYCLES == 2


def test_the_ci_checks_the_monitor_profile_too() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "--profile monitor config" in workflow
