"""The one shared place to normalize skills (JM-20).

The CV pipeline (FastAPI) and vacancy ingestion (Airflow) both go through this
module, so they build the matcher the same way and get identical concept IDs for
identical strings. It is an in-process module, not an HTTP service, and adds no
matching behaviour of its own: everything is delegated to JM-18's SkillMatcher.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

import psycopg
from pydantic import BaseModel, ConfigDict

from jobpdf.normalization.match_cache import SkillMatch
from jobpdf.normalization.matcher import SkillMatcher

log = logging.getLogger(__name__)

LLM_DISABLED = "disabled (ANTHROPIC_API_KEY missing)"
DEFAULT_PROFILE_CONTEXT = "CV"


class ServiceVersions(BaseModel):
    """What produced a set of concept IDs; stored next to them (JM-38)."""

    model_config = ConfigDict(frozen=True)

    taxonomy: str | None  # index_meta taxonomy.esco_version, e.g. "v1.2.1"
    index: str  # match_cache.index_version(index_meta), or "unavailable"
    matcher_prompt: str  # MATCHER_PROMPT_VERSION


class ProfileSkills(BaseModel):
    """Every raw skill of a profile with its match, keyed by its path in the profile."""

    matches: dict[str, SkillMatch]  # e.g. "skills[3]", "experience[1].skills_used[0]"
    versions: ServiceVersions


class _SkillNamed(Protocol):
    name: str


class _Job(Protocol):
    skills_used: list[str]


class _Profile(Protocol):
    """CandidateProfile and ExtractedProfile both fit: skills[].name, experience[].skills_used."""

    skills: Sequence[_SkillNamed]
    experience: Sequence[_Job]


class NormalizationService:
    """A thin, shared wrapper around one SkillMatcher.

    Every dependency is passed in. ``index`` is only used by ``status()`` for a
    live database probe; ``llm_model`` is None when the LLM tier is disabled.
    """

    def __init__(
        self,
        matcher: SkillMatcher,
        versions: ServiceVersions,
        *,
        index: object | None = None,
        llm_model: str | None = None,
        cache_type: str = "unknown",
    ) -> None:
        self._matcher = matcher
        self._versions = versions
        self._index = index
        self._llm_model = llm_model
        self._cache_type = cache_type

    @property
    def versions(self) -> ServiceVersions:
        return self._versions

    def normalize_skills(
        self, raws: Sequence[str], context: str | None = None
    ) -> list[SkillMatch]:
        """One SkillMatch per input, in input order."""
        return self._matcher.match_many(list(raws), context)

    def normalize_skill_groups(
        self, groups: Mapping[str, Sequence[str]], context: str | None = None
    ) -> dict[str, list[SkillMatch]]:
        """Several named lists (e.g. a vacancy's required / nice_to_have) in one call.

        One ``match_many`` call means at most one batched LLM call for all groups;
        results are split back with group order and item order kept.
        """
        flat = [raw for raws in groups.values() for raw in raws]
        matches = self.normalize_skills(flat, context) if flat else []
        result: dict[str, list[SkillMatch]] = {}
        at = 0
        for name, raws in groups.items():
            result[name] = matches[at : at + len(raws)]
            at += len(raws)
        return result

    def normalize_profile(
        self, profile: _Profile, context: str | None = DEFAULT_PROFILE_CONTEXT
    ) -> ProfileSkills:
        """Every raw skill string of a CV profile, matched in one call; the profile is untouched."""
        located = list(profile_skill_paths(profile))
        matches = self.normalize_skills([raw for _, raw in located], context) if located else []
        return ProfileSkills(
            matches={path: match for (path, _), match in zip(located, matches, strict=True)},
            versions=self._versions,
        )

    @staticmethod
    def concept_ids(matches: Iterable[SkillMatch]) -> set[str]:
        """Matched IDs only; unmatched results stay in the match lists, just not here."""
        return {m.concept_id for m in matches if m.concept_id is not None}

    def status(self) -> dict[str, str | None]:
        """Health summary for /health (JM-28). Never contains secret values or URLs."""
        return {
            "index": "ok" if _index_reachable(self._index) else "unavailable",
            "taxonomy_version": self._versions.taxonomy,
            "index_version": self._versions.index,
            "matcher_prompt_version": self._versions.matcher_prompt,
            "llm": "enabled" if self._llm_model else LLM_DISABLED,
            "llm_model": self._llm_model,
            "cache": self._cache_type,
        }

    def close(self) -> None:
        close = getattr(self._index, "close", None)
        if callable(close):
            close()


def profile_skill_paths(profile: _Profile) -> Iterable[tuple[str, str]]:
    """(path, raw skill) for every place a JM-11 profile holds a raw skill name."""
    for i, skill in enumerate(profile.skills):
        yield f"skills[{i}]", skill.name
    for j, job in enumerate(profile.experience):
        for k, raw in enumerate(job.skills_used):
            yield f"experience[{j}].skills_used[{k}]", raw


def _index_reachable(index: object | None) -> bool:
    """A live probe, so /health turns red when the database drops after startup."""
    if index is None or getattr(index, "available", True) is False:
        return False
    connection = getattr(index, "connection", None)
    if connection is None:  # nothing to probe (e.g. an in-memory index)
        return True
    try:
        connection.execute("SELECT 1")
    except psycopg.Error:
        return False
    return True
