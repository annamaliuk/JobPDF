"""The "within reach", "skills worth learning" and "in demand" insights (JM-42).

Each insight answers a "what if" question by re-scoring vacancies for a candidate with
skills added, through ``score_vacancy`` (JM-24), the same code that ranks the cards. No
weight or threshold is repeated here, so the insights follow any change to the formula,
the band rule or the scorer (JM-26, JM-27). The vacancy's own semantic score is passed
back in unchanged: only skill overlap moves when skills are added.

Definitions (limits are module constants):

- Working set: the first ``TOP_N`` visible, scored results in ranking order (band, then
  fit), the same vacancies the report lists. Hidden and unscored vacancies are left out.
- Within reach: partial vacancies that become strong when at most ``MAX_ADDED`` of their
  missing required skills are added. Every set of 1, then 2, missing skills is tried and
  the smallest one that works is kept (highest new fit, then the vacancy's own order).
  Sorted by skills added, then current fit, then vacancy id.
- Skills worth learning: for each required skill missing somewhere in the working set,
  gain = vacancies that become strong with only that skill added, frequency = vacancies
  that require it. Skills that unlock nothing are dropped. Sorted by gain, frequency,
  then label.
- In demand: the candidate's skills by how many working-set vacancies require them.

Trying sets of skills instead of solving "how many more skills reach the threshold"
keeps this correct if coverage gets per-skill weights. An eligibility cap is respected:
pass ``adjustments`` (the mapping given to ranking) or it is read back from the result.

Pure Python, deterministic, no LLM, database or network. Results hold skill references,
vacancy ids and numbers only; the log line holds counts and a timing.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from itertools import combinations

from pydantic import BaseModel, ConfigDict, Field

from jobpdf.ranking.models import CandidateSkills, SkillRef
from jobpdf.ranking.score import (
    Band,
    FitResult,
    LinearScorer,
    RankingResult,
    Scorer,
    VacancyAdjustment,
    score_vacancy,
)

log = logging.getLogger(__name__)

TOP_N = 50  # vacancies the insights look at
MAX_ADDED = 2  # most skills "within reach" may add
WITHIN_REACH_LIMIT = 3
LEARN_LIMIT = 4
IN_DEMAND_LIMIT = 5


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class WithinReachItem(_Frozen):
    vacancy_id: str
    added: list[SkillRef]  # the skills to add, in the vacancy's own order
    fit_now: float | None
    fit_after: float | None
    required_matched_count: int  # now, before adding
    required_count: int


class SkillToLearnItem(_Frozen):
    skill: SkillRef
    gain: int  # vacancies that become strong with only this skill added
    frequency: int  # working-set vacancies that require it
    vacancy_ids: list[str]  # the vacancies it makes strong, in ranking order


class InDemandItem(_Frozen):
    skill: SkillRef
    count: int  # working-set vacancies that require it


class InsightsResult(_Frozen):
    working_set_size: int
    within_reach: list[WithinReachItem] = Field(default_factory=list)
    skills_to_learn: list[SkillToLearnItem] = Field(default_factory=list)
    in_demand: list[InDemandItem] = Field(default_factory=list)


_DEFAULT_SCORER = LinearScorer()


def compute_insights(
    candidate_skills: CandidateSkills,
    ranking: RankingResult,
    scorer: Scorer = _DEFAULT_SCORER,
    adjustments: Mapping[str, VacancyAdjustment] | None = None,
) -> InsightsResult:
    """All three insights for one ranked candidate.

    Pass the same ``scorer`` and ``adjustments`` that produced ``ranking``; the scorer
    only changes ``fit_after`` and the order among equals, never which vacancies count.
    """
    started = time.perf_counter()
    work = working_set(ranking)
    result = InsightsResult(
        working_set_size=len(work),
        within_reach=within_reach(candidate_skills, work, scorer, adjustments),
        skills_to_learn=skills_worth_learning(candidate_skills, work, scorer, adjustments),
        in_demand=in_demand(candidate_skills, work),
    )
    log.info("insights over %d vacancies: within_reach=%d, to_learn=%d, in_demand=%d in %.0f ms",
             len(work), len(result.within_reach), len(result.skills_to_learn),
             len(result.in_demand), (time.perf_counter() - started) * 1000)
    return result


def working_set(ranking: RankingResult) -> list[FitResult]:
    """The first ``TOP_N`` visible results that have a fit, in ranking order."""
    return [r for r in ranking.results if r.fit is not None][:TOP_N]


def within_reach(
    candidate: CandidateSkills,
    work: Sequence[FitResult],
    scorer: Scorer = _DEFAULT_SCORER,
    adjustments: Mapping[str, VacancyAdjustment] | None = None,
) -> list[WithinReachItem]:
    items: list[WithinReachItem] = []
    for fit in work:
        if fit.band is not Band.PARTIAL:
            continue
        adjustment = _adjustment(fit, adjustments)
        found = _smallest_unlock(candidate, fit, scorer, adjustment)
        if found is None:
            continue
        added, after = found
        items.append(WithinReachItem(
            vacancy_id=fit.vacancy.id, added=list(added), fit_now=fit.fit,
            fit_after=after.fit, required_matched_count=fit.overlap.required_matched_count,
            required_count=fit.overlap.required_count,
        ))
    items.sort(key=lambda i: (len(i.added), i.fit_now is None, -(i.fit_now or 0.0),
                              i.vacancy_id))
    return items[:WITHIN_REACH_LIMIT]


def skills_worth_learning(
    candidate: CandidateSkills,
    work: Sequence[FitResult],
    scorer: Scorer = _DEFAULT_SCORER,
    adjustments: Mapping[str, VacancyAdjustment] | None = None,
) -> list[SkillToLearnItem]:
    refs: dict[str, SkillRef] = {}  # key -> first ref seen, for display
    requiring: dict[str, list[FitResult]] = {}
    for fit in work:
        for ref in fit.overlap.required_missing:  # unique per vacancy
            refs.setdefault(ref.key, ref)
            requiring.setdefault(ref.key, []).append(fit)

    items: list[SkillToLearnItem] = []
    for key, fits in requiring.items():
        unlocked = [
            fit.vacancy.id for fit in fits
            if fit.band is not Band.STRONG
            and _rescore(candidate, fit, [refs[key]], scorer,
                         _adjustment(fit, adjustments)).band is Band.STRONG
        ]
        if unlocked:
            items.append(SkillToLearnItem(skill=refs[key], gain=len(unlocked),
                                          frequency=len(fits), vacancy_ids=unlocked))
    items.sort(key=lambda i: (-i.gain, -i.frequency, *_label_key(i.skill)))
    return items[:LEARN_LIMIT]


def in_demand(candidate: CandidateSkills, work: Sequence[FitResult]) -> list[InDemandItem]:
    counts = dict.fromkeys(candidate.refs, 0)
    for fit in work:
        for ref in fit.overlap.required_matched:
            if ref.key in counts:
                counts[ref.key] += 1
    items = [InDemandItem(skill=candidate.refs[key], count=n) for key, n in counts.items() if n]
    items.sort(key=lambda i: (-i.count, *_label_key(i.skill)))
    return items[:IN_DEMAND_LIMIT]


# --- helpers -----------------------------------------------------------------------


def _smallest_unlock(
    candidate: CandidateSkills, fit: FitResult, scorer: Scorer, adjustment: VacancyAdjustment
) -> tuple[tuple[SkillRef, ...], FitResult] | None:
    missing = fit.overlap.required_missing
    for size in range(1, MAX_ADDED + 1):
        best: tuple[tuple[SkillRef, ...], FitResult] | None = None
        for added in combinations(missing, size):  # vacancy order, so ties keep the first
            after = _rescore(candidate, fit, added, scorer, adjustment)
            if after.band is not Band.STRONG:
                continue
            if best is None or (after.fit or 0.0) > (best[1].fit or 0.0):
                best = (added, after)
        if best is not None:
            return best
    return None


def _rescore(
    candidate: CandidateSkills,
    fit: FitResult,
    added: Sequence[SkillRef],
    scorer: Scorer,
    adjustment: VacancyAdjustment,
) -> FitResult:
    """The vacancy scored again with skills added; its semantic score stays the same."""
    return score_vacancy(candidate.with_added(added), fit.vacancy, fit.semantic, adjustment,
                         scorer)


def _adjustment(
    fit: FitResult, adjustments: Mapping[str, VacancyAdjustment] | None
) -> VacancyAdjustment:
    """The adjustment ranking used; without the mapping, a cap is read from the result."""
    if adjustments is not None and fit.vacancy.id in adjustments:
        return adjustments[fit.vacancy.id]
    if fit.band is not fit.band_before_cap:
        return VacancyAdjustment(band_cap=fit.band)
    return VacancyAdjustment()


def _label_key(ref: SkillRef) -> tuple[str, str]:
    return ((ref.label or ref.raw).casefold(), ref.key)
