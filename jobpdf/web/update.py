"""Re-rank a report with skills added or removed, without the CV (JM-53).

The report page keeps what ranking needs in one hidden form field, ``state``: the skills as
they were matched, the past-work vectors (numbers, not text), the counted months of
experience, the summary, the warning codes of the earlier stages, the versions and the page's
choices (country, language). "Update matches" posts it back with the changes, and the vacancies
the server has now are re-ranked with the same code as the first run
(``pipeline.rank_profile``). Nothing is stored and the CV is not needed again.

Typed skills are matched without the LLM: exact names, then vector search
(``normalize_skills(..., use_llm=False)``). A word neither can place is kept as an unrecognised
skill, which counts when a vacancy writes it the same way. So an update makes no model call.

The field is not signed. Whoever edits it only changes their own page; it is size-capped and
validated, and a field from another version of the page is refused with "upload again".
"""

from __future__ import annotations

import base64
import binascii
import logging
import re
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from jobpdf.api.response import (
    AnalysisResponse,
    Meta,
    ProfileOut,
    SkillOut,
    assemble_response,
    insights_out,
    skill_from_match,
)
from jobpdf.normalization.match_cache import SkillMatch
from jobpdf.pipeline import (
    NO_VACANCIES,
    PipelineResult,
    PipelineVersions,
    PipelineWarning,
    rank_profile,
)
from jobpdf.ranking.eligibility import location_adjustments
from jobpdf.ranking.models import RANKING_VERSION, CandidateSkills, SkillRef, VacancyFeatures
from jobpdf.ranking.score import Scorer
from jobpdf.ranking.semantic import CandidateVectors
from jobpdf.web.view import skill_key

log = logging.getLogger(__name__)

STATE_VERSION = 1
MAX_STATE_CHARS = 256 * 1024  # a few jobs' vectors are 20 to 60 KB; this leaves room
MAX_ADDED_SKILLS = 20  # per update
MAX_SKILL_CHARS = 60
ADDED_PREFIX = "added["  # the path of a typed skill: added[0], added[1], ...

# The error code for the page (routes.WEB_DETAILS words it).
REPORT_EXPIRED = "report_expired"


class UpdateError(Exception):
    """The update cannot be made; ``code`` says why and never holds what was sent."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ReportState(BaseModel):
    """What the report page carries so it can be re-ranked."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = STATE_VERSION
    summary: str | None
    total_experience_months: int | None  # shown on the page
    experience_months: int | None  # what JM-47 and JM-50 may trust (counted_experience_months)
    skills: dict[str, SkillMatch]  # profile path -> match, typed ones under added[i]
    vectors: CandidateVectors | None
    warnings: list[PipelineWarning]  # the stages before ranking; ranking adds its own again
    versions: PipelineVersions
    home_country: str | None = None
    language: str


def state_from_result(result: PipelineResult, home_country: str | None,
                      language: str) -> ReportState:
    """The state for a page made from a ``PipelineResult``."""
    return ReportState(
        summary=result.profile.summary,
        total_experience_months=result.profile.total_experience_months,
        experience_months=result.experience_months,
        skills={path: _slim(match) for path, match in result.skill_matches.items()},
        vectors=result.vectors,
        warnings=[w for w in result.warnings if w.stage != "ranking"],
        versions=result.versions,
        home_country=home_country,
        language=language,
    )


def encode_state(state: ReportState) -> str:
    return base64.urlsafe_b64encode(state.model_dump_json().encode("utf-8")).decode("ascii")


def decode_state(text: str) -> ReportState:
    """The state from the form, or ``UpdateError(REPORT_EXPIRED)`` for anything that is not a
    state this version of the page made."""
    if not text or len(text) > MAX_STATE_CHARS:
        raise UpdateError(REPORT_EXPIRED)
    try:
        state = ReportState.model_validate_json(
            base64.urlsafe_b64decode(text.encode("ascii")))
    except (ValueError, binascii.Error):  # bad base64, non-ASCII, bad JSON, wrong shape
        raise UpdateError(REPORT_EXPIRED) from None
    if state.version != STATE_VERSION or state.versions.ranking != RANKING_VERSION:
        raise UpdateError(REPORT_EXPIRED)
    return state


@dataclass(frozen=True)
class Typed:
    names: list[str]
    too_long: int = 0  # left out: longer than MAX_SKILL_CHARS
    over_limit: int = 0  # left out: past the first MAX_ADDED_SKILLS


def typed_skills(text: str, ticked: Iterable[str] = ()) -> Typed:
    """The skill names from the "add skills" line (separated by commas, semicolons or new
    lines) and the "I have this" ticks, trimmed and without repeats. Names over the limits are
    left out and counted, so the page can say so instead of losing the report."""
    names = [n.strip() for n in re.split(r"[,;\n]", text or "")]
    names += [n.strip() for n in ticked]
    unique = list(dict.fromkeys(n for n in names if n))
    fitting = [n for n in unique if len(n) <= MAX_SKILL_CHARS]
    return Typed(fitting[:MAX_ADDED_SKILLS], len(unique) - len(fitting),
                 max(len(fitting) - MAX_ADDED_SKILLS, 0))


def update_report(
    state: ReportState,
    add: Sequence[str],
    remove: Iterable[str],
    *,
    normalizer,
    vacancies: Sequence[VacancyFeatures],
    scorer: Scorer | None = None,
) -> tuple[ReportState, AnalysisResponse]:
    """The changed state and the re-ranked response. ``add`` are skill names, ``remove`` are
    skill keys (``SkillRef.key``) as the page's remove ticks send them."""
    started = time.perf_counter()
    matches = normalizer.normalize_skills(list(add), use_llm=False) if add else []
    state = _changed(state, matches, set(remove))
    response = _rerank(state, vacancies, scorer, started)
    log.info("report updated: typed=%d removed=%d skills=%d vacancies=%d in %d ms",
             len(add), len(set(remove)), len(state.skills), len(vacancies),
             response.meta.timings_ms["total"])
    return state, response


def _key(match: SkillMatch) -> str:
    """The key the page's remove ticks send (``view.skill_key``)."""
    return skill_key(skill_from_match(match))


def _changed(state: ReportState, added: Sequence[SkillMatch], remove: set[str]) -> ReportState:
    skills = {p: m for p, m in state.skills.items() if _key(m) not in remove}
    have = {SkillRef.from_match(m).key for m in skills.values()}
    next_index = 1 + max((int(p[len(ADDED_PREFIX):-1]) for p in state.skills
                          if p.startswith(ADDED_PREFIX)), default=-1)
    for match in added:
        ref = SkillRef.from_match(match)
        if ref.is_empty or ref.key in have:  # already in the CV, or typed twice
            continue
        skills[f"{ADDED_PREFIX}{next_index}]"] = _slim(match)
        have.add(ref.key)
        next_index += 1
    return state.model_copy(update={"skills": skills})


def _rerank(state: ReportState, vacancies: Sequence[VacancyFeatures], scorer: Scorer | None,
            started: float) -> AnalysisResponse:
    adjustments = location_adjustments(vacancies, state.home_country)
    ranking, insights = rank_profile(CandidateSkills.from_skill_matches(state.skills),
                                     state.vectors, vacancies, state.experience_months,
                                     adjustments=adjustments, scorer=scorer)
    codes = Counter(([NO_VACANCIES] if not vacancies else []) + list(ranking.warnings))
    warnings = [*state.warnings,
                *(PipelineWarning(stage="ranking", code=c, count=n) for c, n in codes.items())]
    profile = ProfileOut(
        summary=state.summary, skills=[_skill_out(p, m) for p, m in state.skills.items()],
        experience=[], education=[], languages=[], certifications=[],
        total_experience_months=state.total_experience_months)
    ms = round((time.perf_counter() - started) * 1000)
    versions = state.versions.model_copy(
        update={"ranking": ranking.ranking_version, "scorer": ranking.scorer_name})
    return assemble_response(profile, ranking, warnings, versions,
                             Meta(timings_ms={"total": ms},
                                  input_tokens=0, output_tokens=0),
                             insights=insights_out(insights))


def _skill_out(path: str, match: SkillMatch) -> SkillOut:
    out = skill_from_match(match)
    return out.model_copy(update={"added": True}) if path.startswith(ADDED_PREFIX) else out


def _slim(match: SkillMatch) -> SkillMatch:
    """A match without its candidate list, which ranking never reads and would bloat the page."""
    return match.model_copy(update={"candidates": []})
