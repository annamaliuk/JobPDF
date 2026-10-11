"""Combined fit score, bands and ranking (JM-24).

fit = W_SKILL x skill overlap + W_SEMANTIC x semantic similarity, and a band
(strong / partial / weak / unscored) from required-skill coverage plus a semantic
minimum. Every number that produces a result is kept on it, so the report (JM-41)
can explain the formula exactly. Two hooks for later tickets: per-vacancy
adjustments (hide or cap the band, for eligibility checks) and a replaceable
``Scorer`` (learned weights, JM-27). Pure Python: no database, network or LLM.
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from jobpdf.ranking.models import RANKING_VERSION, CandidateSkills, VacancyFeatures
from jobpdf.ranking.overlap import SkillOverlap, skill_overlap
from jobpdf.ranking.semantic import (
    Calibration,
    CandidateVectors,
    SemanticScore,
    semantic_scores,
)

log = logging.getLogger(__name__)

# Weights and thresholds are provisional; JM-26 tunes them on JM-25's dataset.
W_SKILL = 0.7
W_SEMANTIC = 0.3
STRONG_COVERAGE = 0.8  # share of required skills for "strong"
PARTIAL_COVERAGE = 0.5  # share of required skills for "partial"
SEM_MIN = 0.3  # on JM-23's calibrated 0..1 scale; provisional, JM-26 tunes it

# Warning codes on a FitResult.
SEMANTIC_MISSING = "semantic_missing"
SKILL_MISSING = "skill_missing"
UNSCORED = "unscored"
BAND_WITHOUT_SEMANTIC = "band_without_semantic"
# Warning codes on a RankingResult.
SEMANTIC_UNAVAILABLE = "semantic_unavailable"
DUPLICATE_VACANCY_ID = "duplicate_vacancy_id"


def _check_weights(w_skill: float, w_semantic: float) -> None:
    if w_skill < 0 or w_semantic < 0 or not math.isclose(w_skill + w_semantic, 1.0):
        raise ValueError(
            f"fit weights must be non-negative and add up to 1, "
            f"got skill={w_skill} + semantic={w_semantic}"
        )


_check_weights(W_SKILL, W_SEMANTIC)


class Band(str, Enum):
    STRONG = "strong"
    PARTIAL = "partial"
    WEAK = "weak"
    UNSCORED = "unscored"


# Lower rank = better band; used for capping and sorting.
BAND_ORDER: dict[Band, int] = {Band.STRONG: 0, Band.PARTIAL: 1, Band.WEAK: 2, Band.UNSCORED: 3}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FitComponents(_Frozen):
    """The inputs of the fit score; None means "could not be computed"."""

    skill: float | None  # SkillOverlap.score
    required_coverage: float | None  # SkillOverlap.required_coverage
    semantic: float | None  # SemanticScore.score (calibrated 0..1)
    semantic_raw: float | None  # SemanticScore.raw (cosine)


class VacancyAdjustment(_Frozen):
    """Per-vacancy outcome of eligibility checks (backlog ticket): hide or cap the band."""

    hidden: bool = False
    band_cap: Band | None = None
    reasons: list[str] = Field(default_factory=list)


class Scorer(Protocol):
    """Turns components into one fit number; JM-27 plugs in learned weights here."""

    name: str

    def score(self, components: FitComponents) -> float | None: ...


class LinearScorer:
    """W_SKILL x skill + W_SEMANTIC x semantic; a missing part falls back to the other."""

    name = "linear"

    def __init__(self, w_skill: float = W_SKILL, w_semantic: float = W_SEMANTIC) -> None:
        _check_weights(w_skill, w_semantic)
        self.w_skill = w_skill
        self.w_semantic = w_semantic

    @property
    def weights(self) -> dict[str, float]:
        return {"skill": self.w_skill, "semantic": self.w_semantic}

    def score(self, components: FitComponents) -> float | None:
        skill, semantic = components.skill, components.semantic
        if skill is not None and semantic is not None:
            return self.w_skill * skill + self.w_semantic * semantic
        if skill is not None:
            return skill
        return semantic  # None when both are missing


def fit_warnings(components: FitComponents, fit: float | None) -> list[str]:
    """Why a fit is partial or missing; independent of the scorer used."""
    if fit is None:
        return [UNSCORED]
    if components.semantic is None:
        return [SEMANTIC_MISSING]
    if components.skill is None:
        return [SKILL_MISSING]
    return []


def assign_band(
    overlap: SkillOverlap, semantic: float | None, cap: Band | None = None
) -> tuple[Band, Band]:
    """(band, band_before_cap). Public so "what if you learned X" (JM-42) uses the
    exact same rule as the ranked cards.

    ``semantic`` is the calibrated semantic score (FitComponents.semantic). A
    missing semantic score never blocks "strong"; a vacancy without listed skills
    is at most "partial", because its requirements can't be confirmed.
    """
    basis = overlap.required_coverage
    if basis is None:
        basis = overlap.score  # only nice-to-have skills listed
    semantic_ok = semantic is None or semantic >= SEM_MIN
    if basis is not None:
        if basis >= STRONG_COVERAGE and semantic_ok:
            band = Band.STRONG
        elif basis >= PARTIAL_COVERAGE:
            band = Band.PARTIAL
        else:
            band = Band.WEAK
    elif semantic is None:
        band = Band.UNSCORED
    else:
        band = Band.PARTIAL if semantic >= SEM_MIN else Band.WEAK
    if cap is not None and BAND_ORDER[cap] > BAND_ORDER[band]:
        return cap, band
    return band, band


# --- ranking -----------------------------------------------------------------------


class FitResult(_Frozen):
    """One vacancy with its fit, band and everything that explains them."""

    vacancy: VacancyFeatures
    fit: float | None
    band: Band
    band_before_cap: Band  # differs from band only when an adjustment capped it
    components: FitComponents
    overlap: SkillOverlap
    semantic: SemanticScore | None  # None when candidate vectors were unavailable
    adjustment_reasons: list[str]
    warnings: list[str]  # codes only
    ranking_version: str


class RankingResult(_Frozen):
    results: list[FitResult]  # visible, sorted best first
    hidden: list[FitResult]  # hidden by adjustments: fully scored, never dropped
    band_counts: dict[Band, int]  # visible results only; every band present
    hidden_count: int
    scorer_name: str
    weights: dict[str, float]  # e.g. {"skill": 0.7, "semantic": 0.3}
    thresholds: dict[str, float]  # STRONG_COVERAGE, PARTIAL_COVERAGE, SEM_MIN
    semantic_calibrated: bool | None  # None when no semantic scores were computed
    ranking_version: str
    warnings: list[str]  # codes only


THRESHOLDS = {
    "strong_coverage": STRONG_COVERAGE,
    "partial_coverage": PARTIAL_COVERAGE,
    "sem_min": SEM_MIN,
}
_DEFAULT_SCORER = LinearScorer()


def rank_vacancies(
    candidate_skills: CandidateSkills,
    candidate_vectors: CandidateVectors | None,
    vacancies: Sequence[VacancyFeatures],
    adjustments: Mapping[str, VacancyAdjustment] | None = None,
    scorer: Scorer = _DEFAULT_SCORER,
    *,
    calibration: Calibration | None = None,
) -> RankingResult:
    """Score, band and sort vacancies for one candidate.

    ``candidate_vectors=None`` means embeddings are unavailable: ranking still
    works on skills alone. ``adjustments`` is keyed by vacancy id (eligibility
    checks); a hidden vacancy is still fully scored and listed in ``hidden``.
    ``calibration`` is passed to JM-23's semantic_scores (default: its file).
    """
    started = time.perf_counter()
    adjustments = adjustments or {}
    warnings: list[str] = []
    if candidate_vectors is None:
        warnings.append(SEMANTIC_UNAVAILABLE)
        semantics: list[SemanticScore | None] = [None] * len(vacancies)
    else:
        semantics = list(semantic_scores(candidate_vectors, vacancies, calibration))
    if any(n > 1 for n in Counter(v.id for v in vacancies).values()):
        warnings.append(DUPLICATE_VACANCY_ID)

    visible: list[FitResult] = []
    hidden: list[FitResult] = []
    for vacancy, semantic in zip(vacancies, semantics, strict=True):
        adjustment = adjustments.get(vacancy.id, VacancyAdjustment())
        result = _fit_result(candidate_skills, vacancy, semantic, adjustment, scorer)
        (hidden if adjustment.hidden else visible).append(result)
    visible.sort(key=_sort_key)
    hidden.sort(key=_sort_key)

    band_counts = dict.fromkeys(Band, 0)
    for result in visible:
        band_counts[result.band] += 1
    scored = [s for s in semantics if s is not None]
    log.info("ranked %d vacancies: %s, hidden=%d, warnings=%s in %.0f ms", len(vacancies),
             {b.value: n for b, n in band_counts.items()}, len(hidden), warnings,
             (time.perf_counter() - started) * 1000)
    return RankingResult(
        results=visible,
        hidden=hidden,
        band_counts=band_counts,
        hidden_count=len(hidden),
        scorer_name=scorer.name,
        weights=dict(getattr(scorer, "weights", {})),
        thresholds=dict(THRESHOLDS),
        semantic_calibrated=scored[0].calibrated if scored else None,
        ranking_version=RANKING_VERSION,
        warnings=warnings,
    )


def _fit_result(
    candidate_skills: CandidateSkills,
    vacancy: VacancyFeatures,
    semantic: SemanticScore | None,
    adjustment: VacancyAdjustment,
    scorer: Scorer,
) -> FitResult:
    overlap = skill_overlap(candidate_skills, vacancy)
    components = FitComponents(
        skill=overlap.score,
        required_coverage=overlap.required_coverage,
        semantic=semantic.score if semantic is not None else None,
        semantic_raw=semantic.raw if semantic is not None else None,
    )
    fit = scorer.score(components)
    band, band_before_cap = assign_band(overlap, components.semantic, adjustment.band_cap)
    warnings = fit_warnings(components, fit)
    if band_before_cap is Band.STRONG and components.semantic is None:
        warnings.append(BAND_WITHOUT_SEMANTIC)
    return FitResult(
        vacancy=vacancy, fit=fit, band=band, band_before_cap=band_before_cap,
        components=components, overlap=overlap, semantic=semantic,
        adjustment_reasons=list(adjustment.reasons), warnings=warnings,
        ranking_version=RANKING_VERSION,
    )


def _sort_key(result: FitResult) -> tuple:
    """Band, then fit (highest first), then newest posting, then id; None sorts last."""
    posted = _timestamp(result.vacancy.posted_at)
    return (
        BAND_ORDER[result.band],
        result.fit is None, -(result.fit or 0.0),
        posted is None, -(posted or 0.0),
        result.vacancy.id,
    )


def _timestamp(value: datetime | None) -> float | None:
    if value is None:
        return None
    # Naive datetimes are taken as UTC, so mixed naive/aware input can't raise.
    return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).timestamp()
