"""The ranking data contract (JM-22): what candidate skills and vacancies look like
when they reach ranking.

This file is the agreement between the CV side and the vacancy side (JM-21,
JM-32, JM-38 produce ``VacancyFeatures``). Models are frozen and reject unknown
fields, so a misspelled field fails loudly instead of silently disappearing.
Optional fields default to None so later tickets can fill them without breaking
stored data. It deliberately imports nothing from normalization: skill matches
are accepted through the ``SkillLike`` protocol.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

RANKING_VERSION = "1"
RAW_KEY_PREFIX = "raw:"
ADDED_SOURCE = "added"


class _ContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SkillLike(Protocol):
    """What ranking needs from a normalized skill (JM-18's SkillMatch fits)."""

    @property
    def concept_id(self) -> str | None: ...

    @property
    def raw(self) -> str: ...

    @property
    def normalized(self) -> str: ...

    @property
    def preferred_label(self) -> str | None: ...


class SkillRef(_ContractModel):
    """One skill as ranking compares it: by concept ID, or by text when unrecognised."""

    concept_id: str | None = None
    raw: str
    normalized: str
    label: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _drop_serialized_key(cls, data: Any) -> Any:
        # ``key`` is computed and appears in model_dump(); accept it back so stored
        # JSON round-trips despite extra="forbid".
        if isinstance(data, dict) and "key" in data:
            data = {k: v for k, v in data.items() if k != "key"}
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def key(self) -> str:
        """The concept ID, else "raw:<normalized>" so an unrecognised skill still
        counts on both sides instead of vanishing from the requirements."""
        return self.concept_id or RAW_KEY_PREFIX + self.normalized

    @property
    def is_empty(self) -> bool:
        """No concept and no text: names no skill at all."""
        return self.key == RAW_KEY_PREFIX

    @classmethod
    def from_match(cls, match: SkillLike) -> SkillRef:
        return cls(concept_id=match.concept_id, raw=match.raw, normalized=match.normalized,
                   label=match.preferred_label)


class EducationRequirement(_ContractModel):
    level: str | None = None
    field: str | None = None
    quote: str | None = None


class VacancyFeatures(_ContractModel):
    """A vacancy as ranking sees it. Comments say which ticket fills each field."""

    # ingestion
    id: str  # source posting ID
    source: str  # e.g. "remotive", "synthetic"
    title: str
    company: str | None = None
    url: str | None = None
    posted_at: datetime | None = None
    location_raw: str | None = None  # source text, unchanged
    # vacancy extraction + normalization
    required_skills: list[SkillRef] = Field(default_factory=list)
    preferred_skills: list[SkillRef] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    min_years_experience: int | None = None
    education_requirement: EducationRequirement | None = None
    languages: list[str] | None = None  # None = not extracted; [] = none required
    # semantic similarity (JM-23)
    embedding: list[float] | None = None
    embedding_version: str | None = None
    vacancy_text_version: str | None = None
    # which normalization produced the skill IDs (JM-20's ServiceVersions)
    taxonomy_version: str | None = None
    index_version: str | None = None
    matcher_prompt_version: str | None = None


class CandidateSkills(_ContractModel):
    """A candidate's skills by key, with where each one was found in the profile."""

    refs: dict[str, SkillRef] = Field(default_factory=dict)  # key -> first ref, for display
    sources: dict[str, list[str]] = Field(default_factory=dict)  # key -> profile paths

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(self.refs)

    @classmethod
    def from_skill_matches(cls, matches: Mapping[str, SkillLike]) -> CandidateSkills:
        """From path -> match, the shape of JM-20's ``ProfileSkills.matches``.

        A skill found in several places becomes one key with several sources.
        Empty matches (no concept, no text) name no skill and are skipped; they
        remain in the normalization output.
        """
        refs: dict[str, SkillRef] = {}
        sources: dict[str, list[str]] = {}
        for path, match in matches.items():
            ref = SkillRef.from_match(match)
            if ref.is_empty:
                continue
            refs.setdefault(ref.key, ref)
            sources.setdefault(ref.key, []).append(path)
        return cls(refs=refs, sources=sources)

    def with_added(self, refs: Iterable[SkillRef]) -> CandidateSkills:
        """A new instance with extra skills (source "added"), e.g. for "skills worth
        learning" (JM-42). The original is never mutated."""
        new_refs = dict(self.refs)
        new_sources = {key: list(paths) for key, paths in self.sources.items()}
        for ref in refs:
            if ref.is_empty:
                continue
            new_refs.setdefault(ref.key, ref)
            paths = new_sources.setdefault(ref.key, [])
            if ADDED_SOURCE not in paths:
                paths.append(ADDED_SOURCE)
        return CandidateSkills(refs=new_refs, sources=new_sources)
