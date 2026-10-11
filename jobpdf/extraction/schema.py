"""Schemas for structured CV extraction (JM-11).

Two layers: ``Extracted*`` models are what the LLM returns (they become the
tool's input_schema), and ``CandidateProfile`` is the final profile that code
completes (deduped skills, total experience). Neither layer has fields for a
person's name, email, phone or URLs, by design.
"""

from __future__ import annotations

import copy
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"
TOOL_NAME = "record_candidate_profile"
QUOTE_MAX_CHARS = 200

_YEAR_MONTH = r"\d{4}(-(0[1-9]|1[0-2]))?"
START_DATE_PATTERN = rf"^{_YEAR_MONTH}$"
END_DATE_PATTERN = rf"^({_YEAR_MONTH}|present)$"

Quote = Annotated[
    str,
    Field(
        min_length=1,
        max_length=QUOTE_MAX_CHARS,
        description="Exact phrase copied character for character from the CV text.",
    ),
]
StartDate = Annotated[
    str | None,
    Field(pattern=START_DATE_PATTERN, description='"YYYY-MM" or "YYYY"; null if unknown.'),
]
EndDate = Annotated[
    str | None,
    Field(
        pattern=END_DATE_PATTERN,
        description='"YYYY-MM", "YYYY", or "present" for an ongoing role; null if unknown.',
    ),
]
FoundIn = Literal["skills_list", "experience", "projects", "education", "summary", "other"]


class _LLMModel(BaseModel):
    # A field the schema doesn't define (e.g. "email") is a validation error, so
    # the corrective retry can tell the model to drop it.
    model_config = ConfigDict(extra="forbid")


class SkillMention(_LLMModel):
    name: str = Field(description='Skill name verbatim as written, e.g. "GBQ", "Python 3".')
    kind: Literal["hard", "soft"]
    found_in: FoundIn = Field(description="Where in the CV this mention appears.")
    quote: Quote


class ExtractedJob(_LLMModel):
    title: str
    company: str | None
    location: str | None
    start: StartDate
    end: EndDate
    description: str | None = Field(description="Short summary in the CV's own words.")
    skills_used: list[str] = Field(description="Raw skill names mentioned for this job.")
    quote: Quote = Field(description="The title/company line as written in the CV.")


class ExtractedEducation(_LLMModel):
    institution: str | None
    degree: str | None
    field: str | None
    start: StartDate
    end: EndDate
    quote: Quote


class SpokenLanguage(_LLMModel):
    language: str
    level: str | None = Field(description="Level as written (e.g. 'C1', 'native'); null if none.")


class Certification(_LLMModel):
    name: str
    issuer: str | None
    year: str | None = Field(pattern=r"^\d{4}$", description='"YYYY"; null if unknown.')
    quote: Quote


class ExtractedProfile(_LLMModel):
    """What the model records via the tool. Code completes it into CandidateProfile."""

    summary: str | None = Field(description="The CV's own summary/profile text, or null.")
    skills: list[SkillMention]
    experience: list[ExtractedJob]
    education: list[ExtractedEducation]
    languages: list[SpokenLanguage]
    certifications: list[Certification]


class Skill(BaseModel):
    name: str  # first-seen spelling
    kind: Literal["hard", "soft"]
    found_in: list[str]  # merged across mentions, ordered, unique
    quotes: list[str]


class CandidateProfile(BaseModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    summary: str | None
    skills: list[Skill]
    experience: list[ExtractedJob]
    education: list[ExtractedEducation]
    languages: list[SpokenLanguage]
    certifications: list[Certification]
    total_experience_months: int | None


class ExtractionResult(BaseModel):
    profile: CandidateProfile
    warnings: list[str]
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    from_cache: bool


class VacancySkill(_LLMModel):
    name: str = Field(description="Skill name verbatim as written in the vacancy.")
    quote: Quote


class VacancyProfile(_LLMModel):
    """Vacancy-side counterpart; its prompt comes in Sprint 3."""

    required_skills: list[VacancySkill]
    nice_to_have_skills: list[VacancySkill]
    responsibilities: list[str]
    seniority: str | None


def tool_schema(model: type[BaseModel] = ExtractedProfile) -> dict[str, Any]:
    """A self-contained tool definition for forced tool use.

    All ``$ref``s are inlined (and ``$defs`` dropped) so the schema stands on its
    own; Pydantic's ``title`` keys are dropped because they only cost tokens.
    """
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})
    return {
        "name": TOOL_NAME,
        "description": (
            "Record the structured profile extracted from the CV. Call it exactly once. "
            "Every quote must be copied verbatim from the CV."
        ),
        "input_schema": _inline(raw, defs),
    }


def _inline(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            resolved = _inline(copy.deepcopy(defs[name]), defs)
            extra = {k: v for k, v in node.items() if k != "$ref"}
            return {**resolved, **_inline(extra, defs)}
        # Pydantic's annotation titles are strings; a *property* named "title"
        # (ExtractedJob.title) is a dict and must stay.
        return {
            k: _inline(v, defs)
            for k, v in node.items()
            if not (k == "title" and isinstance(v, str))
        }
    if isinstance(node, list):
        return [_inline(item, defs) for item in node]
    return node
