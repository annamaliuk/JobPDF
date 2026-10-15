"""Learned fit weights behind a feature flag (JM-27).

The hand-tuned formula (JM-24) is ``0.7 x skill + 0.3 x semantic``. This module fits
those two weights from "useful / not useful" feedback on match cards, and lets a
deployment switch between the two scorers without a code change:

    JOBPDF_SCORER=linear    # default: the JM-24 constants
    JOBPDF_SCORER=learned   # weights from learned_weights.json, else back to linear
    JOBPDF_LEARNED_WEIGHTS=/path/to/weights.json   # optional override of the file

Why this model: ``Scorer.score`` sees two informative numbers and the report explains
the formula exactly, so the learned scorer is the same linear formula with other
weights. They are fitted as a logistic regression on ``w x skill + (1 - w) x semantic``,
pulled toward the 0.7 prior (few labels barely move them) and rounded to two decimals.
The intercept is the same for every vacancy, so it cannot change an order and is not
part of the scorer.

Safety rules:

- The flag is read by ``select_scorer()``, which never raises. A missing, unreadable or
  stale weights file means the linear scorer plus a warning code in the log.
- Weights are stored with the ranking version and a fingerprint of the semantic
  calibration they were fitted on. They live on the calibrated 0..1 scale, so if
  JM-26 re-measures floor and ceiling, old weights are refused until refitted.
- Feedback rows hold numbers only (``extra="forbid"``): no CV text, skill names or
  user IDs can be stored by accident.

Feedback format, one JSON object per line::

    {"session": "<random id per CV request>", "vacancy_id": "...", "skill": 0.62,
     "semantic": 0.41, "band": "partial", "label": 1}

``skill`` and ``semantic`` are ``FitComponents.skill`` / ``.semantic`` as shown to the
user (null when they could not be computed). ``label`` is 1 for useful, 0 for not.
Caveat: only cards a user saw can be rated, and they saw them in the current order, so
the data leans toward what the current formula already ranks high.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jobpdf.ranking.models import RANKING_VERSION
from jobpdf.ranking.score import W_SKILL, Band, FitComponents, LinearScorer, Scorer
from jobpdf.ranking.semantic import CALIBRATION, UNCALIBRATED, Calibration

log = logging.getLogger(__name__)

SCORER_ENV = "JOBPDF_SCORER"
WEIGHTS_ENV = "JOBPDF_LEARNED_WEIGHTS"
WEIGHTS_PATH = Path(__file__).with_name("learned_weights.json")
LINEAR = "linear"
LEARNED = "learned"
WEIGHTS_FORMAT_VERSION = 1

# Fitting.
GRID = np.round(np.linspace(0.0, 1.0, 101), 2)  # candidate skill weights, two decimals
DEFAULT_PRIOR_STRENGTH = 20.0  # worth about this many rows that agree with the prior
MIN_ROWS = 30  # below this the fit is refused
RECOMMENDED_ROWS = 100  # below this it runs, but the result should not be trusted
MIN_PER_LABEL = 5
RIDGE = 1e-2  # keeps the slope finite when the labels separate perfectly
NEWTON_ITERS = 50
SLOPE_EPS = 1e-6

# Log codes (never skill text, never file contents).
UNKNOWN_SCORER = "unknown_scorer"
WEIGHTS_MISSING = "learned_weights_missing"
WEIGHTS_INVALID = "learned_weights_invalid"
WEIGHTS_STALE_RANKING = "learned_weights_stale_ranking_version"
WEIGHTS_STALE_CALIBRATION = "learned_weights_stale_calibration"


class NotEnoughData(ValueError):
    """The feedback is too small, or has only one label, to fit anything."""


class WeightsUnusable(Exception):
    """The weights file cannot be used; ``code`` says why (one of the codes above)."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


# --- feedback ----------------------------------------------------------------------


class FeedbackRow(BaseModel):
    """One rated card. Numbers and opaque IDs only: extra fields are rejected."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session: str = Field(min_length=1)  # random per CV request; groups one candidate's cards
    vacancy_id: str = Field(min_length=1)
    skill: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic: float | None = Field(default=None, ge=0.0, le=1.0)
    band: Band | None = None
    label: Literal[0, 1]


def read_feedback(path: Path) -> list[FeedbackRow]:
    """Read a JSONL feedback file. Raises ValueError naming bad line numbers only."""
    rows: list[FeedbackRow] = []
    bad: list[int] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(FeedbackRow.model_validate_json(line))
        except ValidationError:
            bad.append(number)  # only the number: the row may hold something it should not
    if bad:
        shown = ", ".join(map(str, bad[:10])) + (", ..." if len(bad) > 10 else "")
        raise ValueError(f"{len(bad)} invalid feedback row(s), at line(s): {shown}")
    return rows


def file_sha256(path: Path) -> str:
    """Hash with LF line endings, so a Windows checkout hashes the same."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# --- fitting -----------------------------------------------------------------------


@dataclass(frozen=True)
class FitSummary:
    w_skill: float
    w_semantic: float
    slope: float  # 0.0 when the feedback does not rise with the score
    n_rows: int  # rows used: both scores present
    n_positive: int
    n_skipped: int  # rows dropped because a score was missing
    prior_skill: float
    prior_strength: float
    log_loss: float  # mean, of the fitted weights on these rows
    log_loss_prior: float  # mean, of the prior weights on the same rows

    @property
    def informative(self) -> bool:
        """False means feedback did not rise with the score, so weights stay at the prior."""
        return self.slope > SLOPE_EPS

    @property
    def below_recommended(self) -> bool:
        return self.n_rows < RECOMMENDED_ROWS


def fit_weights(
    rows: Sequence[FeedbackRow],
    *,
    prior_skill: float = W_SKILL,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    min_rows: int = MIN_ROWS,
) -> FitSummary:
    """Fit ``w`` in ``w x skill + (1 - w) x semantic`` to useful / not-useful labels.

    For every candidate ``w`` on a two-decimal grid, a logistic regression
    (slope >= 0, intercept) is fitted to the single feature, and the ``w`` with the
    lowest ``log loss + prior_strength x (w - prior_skill)^2`` wins. The penalty makes
    the result sit at the prior when there is little or no signal.
    """
    if prior_strength < 0 or not 0.0 <= prior_skill <= 1.0:
        raise ValueError("prior_skill must be in [0, 1] and prior_strength >= 0")
    usable = [r for r in rows if r.skill is not None and r.semantic is not None]
    n_positive = sum(r.label for r in usable)
    if len(usable) < min_rows:
        raise NotEnoughData(
            f"need at least {min_rows} rated cards with both scores, got {len(usable)}")
    if min(n_positive, len(usable) - n_positive) < MIN_PER_LABEL:
        raise NotEnoughData(f"need at least {MIN_PER_LABEL} cards of each label, "
                            f"got {n_positive} useful and {len(usable) - n_positive} not useful")

    skill = np.array([r.skill for r in usable], dtype=np.float64)
    semantic = np.array([r.semantic for r in usable], dtype=np.float64)
    y = np.array([r.label for r in usable], dtype=np.float64)

    best: tuple[tuple[float, float], float, float] | None = None
    for t in GRID:
        slope, _, nll = _logistic_fit(t * skill + (1.0 - t) * semantic, y)
        key = (nll + prior_strength * (float(t) - prior_skill) ** 2, abs(float(t) - prior_skill))
        if best is None or key < best[0]:
            best = (key, float(t), slope)
    assert best is not None
    _, w_skill, slope = best

    _, _, nll_prior = _logistic_fit(prior_skill * skill + (1.0 - prior_skill) * semantic, y)
    _, _, nll_fitted = _logistic_fit(w_skill * skill + (1.0 - w_skill) * semantic, y)
    n = len(usable)
    return FitSummary(
        w_skill=w_skill, w_semantic=round(1.0 - w_skill, 2), slope=slope, n_rows=n,
        n_positive=n_positive, n_skipped=len(rows) - n, prior_skill=prior_skill,
        prior_strength=prior_strength, log_loss=nll_fitted / n, log_loss_prior=nll_prior / n,
    )


def _logistic_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """(slope, intercept, summed log loss) of ``p = sigmoid(intercept + slope x)``.

    Newton's method with a small ridge on the slope. A negative slope means the
    feature works against the labels; it is replaced by 0 (intercept only), so a
    useless or inverted score can never be turned into a weight.
    """
    mean = float(y.mean())
    base_intercept = math.log(mean / (1.0 - mean))
    slope, intercept = 0.0, base_intercept
    for _ in range(NEWTON_ITERS):
        p = 0.5 * (1.0 + np.tanh((intercept + slope * x) / 2.0))  # stable sigmoid
        residual = p - y
        w = p * (1.0 - p)
        gradient = np.array([residual.sum(), (residual * x).sum() + RIDGE * slope])
        hessian = np.array([[w.sum(), (w * x).sum()], [(w * x).sum(), (w * x * x).sum() + RIDGE]])
        step = np.clip(np.linalg.solve(hessian + 1e-12 * np.eye(2), gradient), -5.0, 5.0)
        intercept -= float(step[0])
        slope -= float(step[1])
        if float(np.abs(step).max()) < 1e-9:
            break
    if slope < 0.0:
        slope, intercept = 0.0, base_intercept
    z = intercept + slope * x
    return slope, intercept, float(np.sum(np.logaddexp(0.0, z) - y * z))


# --- the scorer and its file -------------------------------------------------------


class LearnedLinearScorer(LinearScorer):
    """``LinearScorer`` with fitted weights; a missing part falls back exactly as there."""

    name = "learned_linear"

    @classmethod
    def from_summary(cls, summary: FitSummary) -> LearnedLinearScorer:
        return cls(summary.w_skill, summary.w_semantic)


class LearnedWeights(BaseModel):
    """What ``learned_weights.json`` holds: the weights and what they were fitted on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format_version: int
    weights: dict[str, float]  # "skill" and "semantic"
    ranking_version: str
    calibration_fingerprint: str
    dataset_sha256: str | None
    fitted_at: str
    n_rows: int
    n_positive: int
    prior_skill: float
    prior_strength: float
    log_loss: float
    log_loss_prior: float


def calibration_fingerprint(calibration: Calibration | None = None) -> str:
    """Short fingerprint of the semantic scale: floor, ceiling and embedding model."""
    calibration = CALIBRATION if calibration is None else calibration
    if not calibration.is_valid:
        return UNCALIBRATED
    payload = {"floor": calibration.floor, "ceil": calibration.ceil,
               "embedding_version": calibration.embedding_version}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def write_weights(
    summary: FitSummary,
    path: Path = WEIGHTS_PATH,
    *,
    dataset_sha256: str | None = None,
    calibration: Calibration | None = None,
    now: datetime | None = None,
) -> LearnedWeights:
    stored = LearnedWeights(
        format_version=WEIGHTS_FORMAT_VERSION,
        weights={"skill": summary.w_skill, "semantic": summary.w_semantic},
        ranking_version=RANKING_VERSION,
        calibration_fingerprint=calibration_fingerprint(calibration),
        dataset_sha256=dataset_sha256,
        fitted_at=(now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        n_rows=summary.n_rows, n_positive=summary.n_positive,
        prior_skill=summary.prior_skill, prior_strength=summary.prior_strength,
        log_loss=round(summary.log_loss, 6), log_loss_prior=round(summary.log_loss_prior, 6),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stored.model_dump(), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return stored


def load_learned_scorer(
    path: Path = WEIGHTS_PATH, *, calibration: Calibration | None = None
) -> LearnedLinearScorer:
    """The scorer from a weights file. Raises ``WeightsUnusable`` with a code, never a
    traceback about file contents; ``select_scorer`` turns that into a fallback."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise WeightsUnusable(WEIGHTS_MISSING) from None
    except (OSError, UnicodeError):  # unreadable, or not text at all
        raise WeightsUnusable(WEIGHTS_INVALID) from None
    try:
        stored = LearnedWeights.model_validate_json(text)
        if stored.format_version != WEIGHTS_FORMAT_VERSION:
            raise ValueError("format version")
        scorer = LearnedLinearScorer(stored.weights["skill"], stored.weights["semantic"])
    except (ValidationError, ValueError, KeyError):  # incl. weights that are negative or
        raise WeightsUnusable(WEIGHTS_INVALID) from None  # do not add up to 1
    if stored.ranking_version != RANKING_VERSION:
        raise WeightsUnusable(WEIGHTS_STALE_RANKING)
    if stored.calibration_fingerprint != calibration_fingerprint(calibration):
        raise WeightsUnusable(WEIGHTS_STALE_CALIBRATION)
    return scorer


# --- the feature flag --------------------------------------------------------------


def select_scorer(
    name: str | None = None,
    *,
    weights_path: Path | None = None,
    calibration: Calibration | None = None,
) -> Scorer:
    """The scorer the feature flag asks for. Never raises.

    ``name`` overrides ``JOBPDF_SCORER`` (tests, scripts). Anything but "learned"
    gives the linear scorer; "learned" with unusable weights gives the linear scorer
    and a warning code. Build it once at startup, not per request: it reads the file.
    ``RankingResult.scorer_name`` and ``.weights`` always say which scorer ran.
    """
    raw = name if name is not None else os.environ.get(SCORER_ENV, LINEAR)
    requested = (raw or LINEAR).strip().lower() or LINEAR
    if requested == LINEAR:
        return LinearScorer()
    if requested != LEARNED:
        log.warning("%s: %s is not 'linear' or 'learned'; using the linear scorer",
                    UNKNOWN_SCORER, SCORER_ENV)
        return LinearScorer()
    path = Path(weights_path) if weights_path else Path(os.environ.get(WEIGHTS_ENV) or WEIGHTS_PATH)
    try:
        scorer = load_learned_scorer(path, calibration=calibration)
    except WeightsUnusable as exc:
        log.warning("%s: learned scorer requested but weights are unusable; using the "
                    "linear scorer", exc.code)
        return LinearScorer()
    log.info("learned scorer active: weights=%s", scorer.weights)
    return scorer


def components_of(row: FeedbackRow) -> FitComponents:
    """The scorer's input for a feedback row (coverage and raw cosine are not logged)."""
    return FitComponents(skill=row.skill, required_coverage=None, semantic=row.semantic,
                         semantic_raw=None)
