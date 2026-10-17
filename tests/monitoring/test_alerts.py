import pytest

from jobpdf.monitoring import alerts as alerts_module
from jobpdf.monitoring.alerts import (
    CHECK_MESSAGE,
    CRITICAL,
    FAULT_RATE_CRITICAL,
    FAULT_RATE_WARNING,
    MAX_PART_CHARS,
    MESSAGES,
    MIN_ANALYSES,
    SLOW_P95_MS,
    TOKENS_PER_ANALYSIS_MAX,
    WARNING,
    Alert,
    evaluate,
    safe_part,
    worst,
)
from tests.monitoring.fakes import check, health, healthy, snapshot


def codes(found) -> list[str]:
    return [a.code for a in found]


# --- health ---------------------------------------------------------------------------------


def test_a_healthy_service_raises_no_alert() -> None:
    assert evaluate(healthy(), snapshot(total=50)) == []


def test_a_degraded_check_is_a_warning_and_a_down_check_is_critical_and_listed_first() -> None:
    report = health("down", vacancies=check("degraded", "stale_vacancies"),
                    database=check("down", "database_down"), llm=check())

    found = evaluate(report)

    assert found == [Alert("check.database.database_down", CRITICAL),
                     Alert("check.vacancies.stale_vacancies", WARNING)]


def test_a_down_report_with_no_down_check_still_raises_one_critical_alert() -> None:
    found = evaluate(health("down", database=check("degraded", "slow")))

    assert Alert("health_down", CRITICAL) in found


def test_a_down_report_that_names_its_cause_does_not_add_a_second_alert() -> None:
    found = evaluate(health("down", database=check("down", "database_down")))

    assert codes(found) == ["check.database.database_down"]


def test_names_and_codes_from_the_server_are_reduced_before_they_reach_an_alert() -> None:
    report = health("degraded", **{"Fetch Now!": check("degraded", "Stale <b>X</b>: " + "z" * 200)})

    (alert,) = evaluate(report)

    assert alert.code.startswith("check.fetchnow.")
    assert set(alert.code) <= set("abcdefghijklmnopqrstuvwxyz0123456789_.:-")
    assert len(alert.code.split(".", 2)[2]) <= MAX_PART_CHARS
    assert alert.message == CHECK_MESSAGE  # an unknown code gets the fixed sentence only


def test_a_known_check_code_adds_its_hint_to_the_sentence() -> None:
    (alert,) = evaluate(health("degraded", fetch=check("degraded", "stale_fetch")))

    assert alert.message.startswith(CHECK_MESSAGE) and "over a day" in alert.message


def test_a_check_with_no_code_is_called_unknown() -> None:
    (alert,) = evaluate(health("degraded", fetch=check("degraded", None)))

    assert alert.code == "check.fetch.unknown"


@pytest.mark.parametrize("value, expected", [("ok_1.x:y-z", "ok_1.x:y-z"), ("A B", "ab"),
                                             ("", "unknown"), ("!!!", "unknown"),
                                             ("a" * 100, "a" * MAX_PART_CHARS),
                                             (None, "unknown"), (7, "7")])
def test_safe_part(value, expected) -> None:
    assert safe_part(value) == expected


def test_ignored_codes_are_dropped_and_others_stay() -> None:
    report = health("degraded", fetch=check("degraded", "no_cache"),
                    vacancies=check("degraded", "stale_vacancies"))

    found = evaluate(report, ignore={"check.fetch.no_cache", "never_raised"})

    assert codes(found) == ["check.vacancies.stale_vacancies"]


def test_two_names_that_reduce_to_the_same_code_give_one_alert() -> None:
    report = health("degraded", **{"a b": check("degraded", "x"), "ab": check("degraded", "x")})

    assert len(evaluate(report)) == 1


# --- metrics --------------------------------------------------------------------------------


def faulty(count: int, total: int = 100, key: str = "extraction:llm_error"):
    return snapshot(total=total, failed=count, errors={key: count})


@pytest.mark.parametrize("faults, expected", [
    (1, []), (19, []),
    (20, [Alert("service_errors", WARNING)]),
    (49, [Alert("service_errors", WARNING)]),
    (50, [Alert("service_errors", CRITICAL)]),
    (100, [Alert("service_errors", CRITICAL)]),
])
def test_the_service_fault_rate_has_two_thresholds_that_include_their_value(
        faults, expected) -> None:
    assert (FAULT_RATE_WARNING, FAULT_RATE_CRITICAL) == (0.2, 0.5)

    assert evaluate(None, faulty(faults)) == expected


def test_the_rate_is_not_judged_on_fewer_than_the_minimum_number_of_analyses() -> None:
    assert evaluate(None, faulty(MIN_ANALYSES - 1, total=MIN_ANALYSES - 1)) == []
    assert evaluate(None, faulty(MIN_ANALYSES, total=MIN_ANALYSES)) == [
        Alert("service_errors", CRITICAL)]


@pytest.mark.parametrize("key", ["parse:corrupt", "parse:encrypted", "parse:too_large",
                                 "extraction:no_text", "parse:no_text"])
def test_unreadable_cvs_are_not_our_fault(key) -> None:
    assert evaluate(None, faulty(100, key=key)) == []


@pytest.mark.parametrize("key", ["extraction:llm_error", "grounding:llm_error",
                                 "extraction:llm_not_configured",
                                 "extraction:extraction_invalid", "unexpected"])
def test_llm_and_unexpected_failures_are_our_fault_at_any_stage(key) -> None:
    assert evaluate(None, faulty(100, key=key)) == [Alert("service_errors", CRITICAL)]


def test_faults_and_user_errors_are_added_up_separately() -> None:
    found = evaluate(None, snapshot(total=100, failed=50, errors={
        "parse:corrupt": 30, "extraction:llm_error": 10, "unexpected": 10}))

    assert found == [Alert("service_errors", WARNING)]  # 20 of 100, not 50 of 100


@pytest.mark.parametrize("p95, samples, expected", [
    (SLOW_P95_MS, 100, False), (SLOW_P95_MS + 1, 100, True), (SLOW_P95_MS + 1, 9, False),
    (SLOW_P95_MS + 1, 10, True), (None, 100, False)])
def test_slow_analyses_need_enough_samples_and_a_p95_over_the_limit(p95, samples, expected) -> None:
    found = evaluate(None, snapshot(p95_ms=p95, samples=samples))

    assert (Alert("slow_analyses", WARNING) in found) is expected


@pytest.mark.parametrize("total_tokens, ok, expected", [
    (TOKENS_PER_ANALYSIS_MAX * 10, 10, False), (TOKENS_PER_ANALYSIS_MAX * 10 + 1, 10, True),
    (TOKENS_PER_ANALYSIS_MAX * 9 * 5, 9, False), (0, 10, False)])
def test_token_spikes_compare_the_average_per_successful_analysis(
        total_tokens, ok, expected) -> None:
    found = evaluate(None, snapshot(total=ok, tokens=(total_tokens - 7, 7)))

    assert (Alert("token_spike", WARNING) in found) is expected


def test_failed_analyses_do_not_dilute_the_token_average() -> None:
    snap = snapshot(total=30, failed=20, tokens=(TOKENS_PER_ANALYSIS_MAX * 10 + 1, 0))

    assert Alert("token_spike", WARNING) in evaluate(None, snap)


def test_a_failing_rate_limiter_is_a_warning_but_ordinary_refusals_are_not() -> None:
    assert evaluate(None, snapshot(denied={"limiter_unavailable": 1})) == [
        Alert("limiter_failing", WARNING)]
    assert evaluate(None, snapshot(denied={"rate_limited": 900, "missing_key": 50})) == []


def test_health_and_metrics_alerts_come_together_most_severe_first() -> None:
    found = evaluate(health("degraded", fetch=check("degraded", "stale_fetch")),
                     faulty(60))

    assert found == [Alert("service_errors", CRITICAL), Alert("check.fetch.stale_fetch", WARNING)]


def test_equal_severities_are_listed_by_code_whatever_their_source_or_order() -> None:
    report = health("degraded", zeta=check("degraded", "x"), alpha=check("degraded", "y"))
    snap = snapshot(total=100, failed=30, errors={"unexpected": 30},
                    denied={"limiter_unavailable": 1})

    found = evaluate(report, snap)

    assert codes(found) == ["check.alpha.y", "check.zeta.x", "limiter_failing", "service_errors"]


def test_no_report_and_no_metrics_means_no_alerts_here() -> None:
    assert evaluate(None, None) == []  # "unreachable" is added by the probe


# --- the alert type ---------------------------------------------------------------------------


def test_worst_severity() -> None:
    assert worst([]) is None
    assert worst([Alert("a", WARNING)]) == WARNING
    assert worst([Alert("a", WARNING), Alert("b", CRITICAL), Alert("c", WARNING)]) == CRITICAL


def test_every_fixed_alert_code_has_a_sentence() -> None:
    fixed = {getattr(alerts_module, name) for name in (
        "UNREACHABLE", "BAD_RESPONSE", "HEALTH_DOWN", "METRICS_UNAVAILABLE", "SERVICE_ERRORS",
        "SLOW_ANALYSES", "TOKEN_SPIKE", "LIMITER_FAILING")}

    assert fixed == set(MESSAGES)
    assert all(Alert(c, WARNING).message == MESSAGES[c] for c in fixed)
    assert all(MESSAGES[c].endswith(".") for c in fixed)
    assert Alert("anything_else", WARNING).message == CHECK_MESSAGE
