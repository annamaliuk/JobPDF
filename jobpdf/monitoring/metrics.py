"""CV-analysis metrics (JM-34): counters and timing percentiles, in memory, codes only.

``MetricsRecorder`` is what the API (JM-36) calls around the pipeline; a dashboard (JM-39)
or ``/metrics`` endpoint reads ``snapshot()``.

    result = recorder.observe_analysis(lambda: analyze_cv(...))

records the outcome and re-raises any error unchanged. Or call ``record_result``,
``record_error`` and ``record_access_denied`` directly.

What is recorded, per process (it resets on restart; nothing is stored):
- analyses: total / ok / failed, and failures by ``stage:code``;
- warnings: how many analyses carried each ``stage:code`` warning;
- per-stage timings of successful analyses (count, p50, p95, max), plus the total time of
  failed ones, from the last ``MAX_SAMPLES`` per stage (memory stays bounded);
- LLM tokens used (extraction), the cost driver the spec worries about;
- refused requests by access code (``missing_key``, ``rate_limited``, ...).

Privacy: only stage names and stable codes go in, never text. Keys are sanitised and capped
at ``MAX_KEYS`` per counter (the overflow counts as ``other``), so a surprising code cannot
grow memory or carry text. Thread-safe: the API runs the pipeline in a threadpool.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict

from jobpdf.pipeline import PipelineError, PipelineResult

MAX_SAMPLES = 1000
MAX_KEYS = 100
MAX_KEY_CHARS = 64
OTHER = "other"
UNEXPECTED = "unexpected"  # an exception that was not a PipelineError
FAILED_TOTAL = "total_failed"  # stage name for the duration of failed analyses
_KEY = re.compile(r"[a-z0-9_.:-]+")


class StageStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int
    p50_ms: int
    p95_ms: int
    max_ms: int


class Analyses(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: int = 0
    ok: int = 0
    failed: int = 0
    with_warnings: int = 0  # ok analyses that carried at least one warning


class MetricsSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    since: datetime
    uptime_s: int
    analyses: Analyses
    errors: dict[str, int]  # "stage:code" -> analyses that failed this way
    warnings: dict[str, int]  # "stage:code" -> analyses that carried it
    access_denied: dict[str, int]  # access code -> requests refused
    stages: dict[str, StageStats]
    tokens: dict[str, int]  # input / output LLM tokens of ok analyses


def percentile(sorted_values: list[int], p: float) -> int:
    """Nearest-rank percentile: the smallest value with at least p% of the samples at or
    below it. ``sorted_values`` must be non-empty and ascending."""
    rank = max(1, math.ceil(p / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


class MetricsRecorder:
    def __init__(self, *, max_samples: int = MAX_SAMPLES, max_keys: int = MAX_KEYS,
                 clock: Callable[[], float] | None = None) -> None:
        self._max_samples = max_samples
        self._max_keys = max_keys
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._since = self._clock()
            self._analyses = Counter[str]()
            self._errors = Counter[str]()
            self._warnings = Counter[str]()
            self._denied = Counter[str]()
            self._tokens = Counter[str]()
            self._samples: dict[str, deque[int]] = {}

    # -- recording --------------------------------------------------------------------

    def record_result(self, result: PipelineResult) -> None:
        with self._lock:
            self._analyses["total"] += 1
            self._analyses["ok"] += 1
            codes = {self._key(self._warnings, f"{w.stage}:{w.code}") for w in result.warnings}
            for code in codes:
                self._warnings[code] += 1
            if codes:
                self._analyses["with_warnings"] += 1
            for stage, ms in result.timings_ms.items():
                self._sample(stage, ms)
            self._tokens["input"] += max(0, result.input_tokens)
            self._tokens["output"] += max(0, result.output_tokens)

    def record_error(self, error: PipelineError, total_ms: int) -> None:
        self._record_failure(f"{error.stage}:{error.code}", total_ms)

    def record_unexpected(self, total_ms: int) -> None:
        self._record_failure(UNEXPECTED, total_ms)

    def record_access_denied(self, code: str) -> None:
        with self._lock:
            self._denied[self._key(self._denied, code)] += 1

    def observe_analysis(self, call: Callable[[], PipelineResult]) -> PipelineResult:
        """Run ``call``, record how it ended, and return its result or re-raise its error."""
        started = time.perf_counter()
        try:
            result = call()
        except PipelineError as exc:
            self.record_error(exc, _elapsed_ms(started))
            raise
        except Exception:
            self.record_unexpected(_elapsed_ms(started))
            raise
        self.record_result(result)
        return result

    # -- reading ----------------------------------------------------------------------

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            now = self._clock()
            stages = {}
            for stage, samples in sorted(self._samples.items()):
                ordered = sorted(samples)
                stages[stage] = StageStats(
                    count=len(ordered), p50_ms=percentile(ordered, 50),
                    p95_ms=percentile(ordered, 95), max_ms=ordered[-1])
            return MetricsSnapshot(
                since=datetime.fromtimestamp(self._since, tz=timezone.utc),
                uptime_s=max(0, round(now - self._since)),
                analyses=Analyses(**{k: self._analyses[k] for k in Analyses.model_fields}),
                errors=dict(sorted(self._errors.items())),
                warnings=dict(sorted(self._warnings.items())),
                access_denied=dict(sorted(self._denied.items())),
                stages=stages,
                tokens={"input": self._tokens["input"], "output": self._tokens["output"]},
            )

    # -- internals --------------------------------------------------------------------

    def _record_failure(self, key: str, total_ms: int) -> None:
        with self._lock:
            self._analyses["total"] += 1
            self._analyses["failed"] += 1
            self._errors[self._key(self._errors, key)] += 1
            self._sample(FAILED_TOTAL, total_ms)

    def _sample(self, stage: str, ms: int) -> None:
        key = self._key(self._samples, stage)
        samples = self._samples.setdefault(key, deque(maxlen=self._max_samples))
        samples.append(max(0, int(ms)))

    def _key(self, known: dict | Counter, raw: str) -> str:
        """A safe key: lowercase ``[a-z0-9_.:-]`` up to ``MAX_KEY_CHARS``, else ``other``;
        a new key beyond ``max_keys`` also becomes ``other``."""
        key = raw if isinstance(raw, str) else ""
        if len(key) > MAX_KEY_CHARS or not _KEY.fullmatch(key):
            return OTHER
        if key not in known and len(known) >= self._max_keys:
            return OTHER
        return key


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)
