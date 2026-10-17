import json
import logging
import threading
from datetime import datetime, timezone

import pytest

from jobpdf.monitoring.metrics import (
    FAILED_TOTAL,
    MAX_KEY_CHARS,
    MAX_SAMPLES,
    UNEXPECTED,
    MetricsRecorder,
    percentile,
)
from jobpdf.pipeline import PipelineError, PipelineResult, PipelineWarning
from tests.pipeline.fakes import run as analyze

T0 = 1_790_000_000.0


class Clock:
    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def result(timings=None, warnings=(), tokens=(100, 20)) -> PipelineResult:
    base = analyze()
    return base.model_copy(update={
        "timings_ms": {"parse": 10, "extraction": 800, "total": 900} if timings is None
        else timings,
        "warnings": list(warnings), "input_tokens": tokens[0], "output_tokens": tokens[1],
    })


def warn(stage: str, code: str, count: int = 1) -> PipelineWarning:
    return PipelineWarning(stage=stage, code=code, count=count)


@pytest.fixture
def recorder() -> MetricsRecorder:
    return MetricsRecorder(clock=Clock())


# --- percentiles ----------------------------------------------------------------------------


@pytest.mark.parametrize(("values", "p", "expected"), [
    ([5], 50, 5), ([5], 95, 5), ([1, 2], 50, 1), ([1, 2], 95, 2), ([1, 2, 3], 50, 2),
    (list(range(1, 11)), 50, 5), (list(range(1, 11)), 95, 10), (list(range(1, 101)), 95, 95),
    (list(range(1, 101)), 50, 50), (list(range(1, 101)), 99, 99), ([1, 2, 3, 4], 100, 4),
    ([7, 7, 7], 95, 7), ([1, 100], 0, 1),
])
def test_percentiles_use_the_nearest_rank_rule(values, p, expected) -> None:
    assert percentile(values, p) == expected


# --- recording ------------------------------------------------------------------------------


def test_a_fresh_recorder_reports_zeros() -> None:
    snap = MetricsRecorder(clock=Clock()).snapshot()

    assert snap.analyses.model_dump() == {"total": 0, "ok": 0, "failed": 0, "with_warnings": 0}
    assert snap.errors == {} and snap.warnings == {} and snap.access_denied == {}
    assert snap.stages == {} and snap.tokens == {"input": 0, "output": 0}
    assert snap.uptime_s == 0 and snap.since == datetime.fromtimestamp(T0, tz=timezone.utc)


def test_a_successful_analysis_is_counted_with_its_stage_timings_and_tokens(recorder) -> None:
    recorder.record_result(result(tokens=(1200, 300)))

    snap = recorder.snapshot()

    assert (snap.analyses.total, snap.analyses.ok, snap.analyses.failed) == (1, 1, 0)
    assert snap.stages["extraction"].model_dump() == {
        "count": 1, "p50_ms": 800, "p95_ms": 800, "max_ms": 800}
    assert set(snap.stages) == {"parse", "extraction", "total"}
    assert snap.tokens == {"input": 1200, "output": 300}


def test_stage_percentiles_come_from_the_recorded_samples(recorder) -> None:
    for ms in range(1, 101):
        recorder.record_result(result(timings={"ranking": ms, "total": ms * 10}))

    stats = recorder.snapshot().stages["ranking"]

    assert (stats.count, stats.p50_ms, stats.p95_ms, stats.max_ms) == (100, 50, 95, 100)
    assert recorder.snapshot().stages["total"].p95_ms == 950


def test_warnings_count_analyses_not_items_and_each_code_once_per_analysis(recorder) -> None:
    recorder.record_result(result(warnings=[
        warn("normalization", "unmatched_skill", 7), warn("normalization", "unmatched_skill", 2),
        warn("embedding", "embedding_failed")]))
    recorder.record_result(result(warnings=[warn("normalization", "unmatched_skill", 1)]))
    recorder.record_result(result(warnings=[]))

    snap = recorder.snapshot()

    assert snap.warnings == {"embedding:embedding_failed": 1, "normalization:unmatched_skill": 2}
    assert snap.analyses.with_warnings == 2 and snap.analyses.ok == 3


def test_a_failed_analysis_is_counted_by_stage_and_code_and_timed(recorder) -> None:
    recorder.record_error(PipelineError("extraction", "llm_error"), 1500)
    recorder.record_error(PipelineError("extraction", "llm_error"), 2500)
    recorder.record_error(PipelineError("parse", "encrypted"), 40)

    snap = recorder.snapshot()

    assert (snap.analyses.total, snap.analyses.failed, snap.analyses.ok) == (3, 3, 0)
    assert snap.errors == {"extraction:llm_error": 2, "parse:encrypted": 1}
    assert snap.stages[FAILED_TOTAL].count == 3 and snap.stages[FAILED_TOTAL].max_ms == 2500
    assert snap.tokens == {"input": 0, "output": 0}


def test_failures_do_not_pollute_the_timings_of_successes(recorder) -> None:
    recorder.record_result(result(timings={"total": 100}))
    recorder.record_error(PipelineError("parse", "corrupt"), 999999)

    snap = recorder.snapshot()

    assert snap.stages["total"].max_ms == 100 and snap.stages[FAILED_TOTAL].max_ms == 999999


def test_unexpected_failures_have_their_own_key(recorder) -> None:
    recorder.record_unexpected(30)

    snap = recorder.snapshot()

    assert snap.errors == {UNEXPECTED: 1} and snap.analyses.failed == 1


def test_refused_requests_are_counted_by_access_code(recorder) -> None:
    for code in ("rate_limited", "invalid_key", "rate_limited", "too_large"):
        recorder.record_access_denied(code)

    snap = recorder.snapshot()

    assert snap.access_denied == {"invalid_key": 1, "rate_limited": 2, "too_large": 1}
    assert snap.analyses.total == 0  # refused requests never reached the pipeline


def test_negative_numbers_are_clamped_not_recorded_as_is(recorder) -> None:
    recorder.record_result(result(timings={"total": -5}, tokens=(-10, -1)))

    snap = recorder.snapshot()

    assert snap.stages["total"].max_ms == 0 and snap.tokens == {"input": 0, "output": 0}


def test_uptime_follows_the_clock_and_reset_starts_over() -> None:
    clock = Clock()
    recorder = MetricsRecorder(clock=clock)
    recorder.record_access_denied("rate_limited")
    recorder.record_error(PipelineError("parse", "empty_document"), 5)
    recorder.record_result(result(warnings=[warn("grounding", "dropped")]))
    clock.now += 125

    assert recorder.snapshot().uptime_s == 125

    recorder.reset()
    snap = recorder.snapshot()
    assert snap.uptime_s == 0 and snap.access_denied == {}
    assert snap.errors == {} and snap.warnings == {} and snap.stages == {}
    assert snap.analyses.model_dump() == {"total": 0, "ok": 0, "failed": 0, "with_warnings": 0}
    assert snap.tokens == {"input": 0, "output": 0}
    assert snap.since == datetime.fromtimestamp(T0 + 125, tz=timezone.utc)


# --- observe_analysis -------------------------------------------------------------------------


def test_observing_a_successful_call_records_it_and_returns_the_result(recorder) -> None:
    expected = result()

    returned = recorder.observe_analysis(lambda: expected)

    assert returned is expected and recorder.snapshot().analyses.ok == 1


def test_observing_a_pipeline_error_records_it_and_reraises_it_unchanged(recorder) -> None:
    error = PipelineError("parse", "encrypted")

    def failing():
        raise error

    with pytest.raises(PipelineError) as caught:
        recorder.observe_analysis(failing)

    assert caught.value is error
    assert recorder.snapshot().errors == {"parse:encrypted": 1}


def test_observing_an_unexpected_error_records_it_and_reraises(recorder) -> None:
    def failing():
        raise ZeroDivisionError("SecretCvTextXYZ")

    with pytest.raises(ZeroDivisionError):
        recorder.observe_analysis(failing)

    snap = recorder.snapshot()
    assert snap.errors == {UNEXPECTED: 1} and "SecretCvTextXYZ" not in snap.model_dump_json()


def test_observing_times_a_failed_call(recorder) -> None:
    def failing():
        raise PipelineError("extraction", "llm_error")

    with pytest.raises(PipelineError):
        recorder.observe_analysis(failing)

    assert recorder.snapshot().stages[FAILED_TOTAL].count == 1


def test_observing_the_real_pipeline_end_to_end(recorder) -> None:
    returned = recorder.observe_analysis(analyze)

    snap = recorder.snapshot()
    assert returned.profile is not None and snap.analyses.ok == 1
    assert set(snap.stages) >= {"parse", "extraction", "ranking", "total"}


# --- bounded memory and safe keys ---------------------------------------------------------------


def test_only_the_latest_samples_are_kept(recorder) -> None:
    small = MetricsRecorder(max_samples=3, clock=Clock())
    for ms in (1000, 10, 20, 30):
        small.record_result(result(timings={"total": ms}))

    stats = small.snapshot().stages["total"]

    assert (stats.count, stats.max_ms, stats.p50_ms) == (3, 30, 20)
    assert MAX_SAMPLES == 1000


def test_new_keys_beyond_the_cap_are_counted_as_other() -> None:
    capped = MetricsRecorder(max_keys=2, clock=Clock())
    for code in ("a", "b", "c", "d", "a"):
        capped.record_error(PipelineError("parse", code), 1)

    # "other" is the one extra key that may exist beyond the cap
    assert capped.snapshot().errors == {"other": 2, "parse:a": 2, "parse:b": 1}


def test_the_cap_keeps_counting_keys_it_already_knows() -> None:
    capped = MetricsRecorder(max_keys=1, clock=Clock())
    capped.record_access_denied("known")
    capped.record_access_denied("new")
    capped.record_access_denied("known")

    assert capped.snapshot().access_denied == {"known": 2, "other": 1}


@pytest.mark.parametrize("bad", [
    "Has Space", "UPPER", "text with a CV sentence.", "x" * (MAX_KEY_CHARS + 1), "", "a/b",
    "ünï", "a\nb", "john.doe@example.com",
])
def test_keys_that_are_not_simple_codes_never_become_labels(recorder, bad) -> None:
    recorder.record_access_denied(bad)

    assert recorder.snapshot().access_denied == {"other": 1}


def test_a_code_of_exactly_the_maximum_length_is_kept(recorder) -> None:
    code = "a" * MAX_KEY_CHARS

    recorder.record_access_denied(code)

    assert recorder.snapshot().access_denied == {code: 1}


def test_a_non_string_code_is_other(recorder) -> None:
    recorder.record_access_denied(None)  # type: ignore[arg-type]

    assert recorder.snapshot().access_denied == {"other": 1}


def test_unknown_stage_names_are_sanitised_too(recorder) -> None:
    recorder.record_result(result(timings={"Weird Stage": 5, "total": 1}))

    assert set(recorder.snapshot().stages) == {"other", "total"}


# --- privacy and threads -----------------------------------------------------------------------


def test_the_snapshot_holds_only_codes_and_numbers(recorder) -> None:
    full = analyze()
    recorder.record_result(full)
    recorder.record_error(PipelineError("parse", "encrypted"), 5)
    recorder.record_access_denied("rate_limited")

    text = recorder.snapshot().model_dump_json()

    for secret in (full.profile.model_dump_json()[:40], "@", "http"):
        assert secret not in text
    assert set(json.loads(text)) == {
        "since", "uptime_s", "analyses", "errors", "warnings", "access_denied", "stages",
        "tokens"}


def test_nothing_is_logged(recorder, caplog) -> None:
    recorded = result(warnings=[warn("parse", "x")])
    caplog.set_level(logging.DEBUG)
    caplog.clear()

    recorder.record_result(recorded)
    recorder.record_error(PipelineError("parse", "y"), 1)
    recorder.snapshot()

    assert caplog.records == []


def test_concurrent_recording_loses_nothing() -> None:
    recorder = MetricsRecorder(clock=Clock())
    ok = result()
    per_thread = 200

    def work() -> None:
        for _ in range(per_thread):
            recorder.record_result(ok)
            recorder.record_error(PipelineError("parse", "corrupt"), 5)
            recorder.record_access_denied("rate_limited")
            recorder.snapshot()

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = recorder.snapshot()
    n = 8 * per_thread
    assert (snap.analyses.ok, snap.analyses.failed, snap.analyses.total) == (n, n, 2 * n)
    assert snap.errors == {"parse:corrupt": n} and snap.access_denied == {"rate_limited": n}
    assert snap.tokens["input"] == n * 100


def test_a_code_that_is_not_text_is_filed_under_other() -> None:
    recorder = MetricsRecorder()

    recorder.record_access_denied(None)  # type: ignore[arg-type]
    recorder.record_access_denied(42)  # type: ignore[arg-type]

    assert recorder.snapshot().access_denied == {"other": 2}


def test_recording_takes_the_lock() -> None:
    class Probe:
        entered = 0

        def __enter__(self):
            Probe.entered += 1

        def __exit__(self, *exc):
            return False

    recorder = MetricsRecorder()
    recorder._lock = Probe()  # type: ignore[assignment]

    recorder.record_result(result())
    recorder.record_error(PipelineError("parse", "empty_document"), 5)
    recorder.record_unexpected(5)
    recorder.record_access_denied("rate_limited")
    recorder.snapshot()

    assert Probe.entered == 5
