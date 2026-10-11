"""Skill-overlap score with matched and missing lists (JM-22).

Every later explanation (JM-24's combined score, JM-41's "7 of 9 required
skills", JM-42's "skills worth learning") is built from ``SkillOverlap``.
Skills are compared by ``SkillRef.key``, so unrecognised skills still count.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from jobpdf.ranking.models import CandidateSkills, SkillRef, VacancyFeatures

REQUIRED_WEIGHT = 1.0
PREFERRED_WEIGHT = 0.5

# Warning codes, in the order they appear in SkillOverlap.warnings.
EMPTY_SKILL = "empty_skill"
DUPLICATE_REQUIREMENT = "duplicate_requirement"
NO_VACANCY_SKILLS = "no_vacancy_skills"
NO_CANDIDATE_SKILLS = "no_candidate_skills"

WeightFn = Callable[[str], float]


def default_weight(key: str) -> float:
    """Every skill counts the same. The hook for rarity or generic-skill weighting:
    an upgrade passes a different function, nothing else changes."""
    return 1.0


class SkillOverlap(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    score: float | None  # None when the vacancy lists no skills
    required_coverage: float | None  # matched required / all required; None if none required
    required_matched: list[SkillRef]  # each list in the vacancy's own order
    required_missing: list[SkillRef]
    preferred_matched: list[SkillRef]
    preferred_missing: list[SkillRef]
    required_matched_count: int
    required_count: int
    preferred_matched_count: int
    preferred_count: int
    warnings: list[str]  # codes only


def skill_overlap(
    candidate: CandidateSkills, vacancy: VacancyFeatures, weight: WeightFn = default_weight
) -> SkillOverlap:
    """score = matched effective weight / total effective weight, where a skill's
    effective weight = base weight (required 1.0, preferred 0.5) x ``weight(key)``.

    ``weight`` must return a positive finite number; anything else raises
    ValueError, because it is a programming error, not bad data.
    """
    warnings: set[str] = set()
    required = _clean(vacancy.required_skills, warnings)
    required_keys = {ref.key for ref in required}
    preferred = []
    for ref in _clean(vacancy.preferred_skills, warnings):
        if ref.key in required_keys:  # counted once, as required
            warnings.add(DUPLICATE_REQUIREMENT)
        else:
            preferred.append(ref)

    have = candidate.keys
    req_matched = [r for r in required if r.key in have]
    req_missing = [r for r in required if r.key not in have]
    pref_matched = [r for r in preferred if r.key in have]
    pref_missing = [r for r in preferred if r.key not in have]

    total = _weight_sum(required, REQUIRED_WEIGHT, weight) + _weight_sum(
        preferred, PREFERRED_WEIGHT, weight)
    matched = _weight_sum(req_matched, REQUIRED_WEIGHT, weight) + _weight_sum(
        pref_matched, PREFERRED_WEIGHT, weight)
    if not required and not preferred:
        warnings.add(NO_VACANCY_SKILLS)
    elif not have:
        warnings.add(NO_CANDIDATE_SKILLS)

    return SkillOverlap(
        score=min(1.0, matched / total) if total else None,
        required_coverage=len(req_matched) / len(required) if required else None,
        required_matched=req_matched,
        required_missing=req_missing,
        preferred_matched=pref_matched,
        preferred_missing=pref_missing,
        required_matched_count=len(req_matched),
        required_count=len(required),
        preferred_matched_count=len(pref_matched),
        preferred_count=len(preferred),
        warnings=[w for w in (EMPTY_SKILL, DUPLICATE_REQUIREMENT, NO_VACANCY_SKILLS,
                              NO_CANDIDATE_SKILLS) if w in warnings],
    )


def _clean(refs: Sequence[SkillRef], warnings: set[str]) -> list[SkillRef]:
    """Vacancy order, first occurrence per key, empty refs ignored."""
    seen: set[str] = set()
    cleaned = []
    for ref in refs:
        if ref.is_empty:
            warnings.add(EMPTY_SKILL)
        elif ref.key not in seen:
            seen.add(ref.key)
            cleaned.append(ref)
    return cleaned


def _weight_sum(refs: Sequence[SkillRef], base: float, weight: WeightFn) -> float:
    return sum(base * _checked_weight(weight, ref.key) for ref in refs)


def _checked_weight(weight: WeightFn, key: str) -> float:
    value = weight(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"weight({key!r}) must be a positive finite number, got {value!r}")
    return float(value)
