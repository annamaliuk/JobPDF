"""Combined fit score, bands and ranking (JM-24).

fit = W_SKILL x skill overlap + W_SEMANTIC x semantic similarity, and a band
(strong / partial / weak / unscored) from required-skill coverage plus a semantic
minimum. Every number that produces a result is kept on it, so the report (JM-41)
can explain the formula exactly. Two hooks for later tickets: per-vacancy
adjustments (hide or cap the band, for eligibility checks) and a replaceable
``Scorer`` (learned weights, JM-27). Pure Python: no database, network or LLM.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from jobpdf.ranking.overlap import SkillOverlap

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
