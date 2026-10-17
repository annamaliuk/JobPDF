"""What a posting asks for, as plain strings (before normalization).

``RequirementExtractor`` is the seam for JM-48 (one LLM call per posting turning the
description into required and nice-to-have skills, responsibilities, minimum years,
education and languages, each with a quote). Until it exists, ``TagsExtractor`` applies the
spec's own fallback: "postings without a usable description fall back to the source's tags".
It reads tags only, so responsibilities stay empty (the embedding text is then the title)
and postings from sources that send no tags (Adzuna, Jooble) have no skills and rank as
"unscored" until JM-48 is in.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from jobpdf.ingestion.sources.models import SourcePosting
from jobpdf.ranking.models import EducationRequirement

TAGS_FALLBACK = "tags_fallback"
NO_REQUIREMENTS = "no_requirements"
MAX_TAG_SKILLS = 25
MAX_TAG_CHARS = 60  # longer than this is a sentence, not a skill


class Requirements(BaseModel):
    model_config = ConfigDict(frozen=True)

    required_skills: list[str] = []
    preferred_skills: list[str] = []
    responsibilities: list[str] = []
    min_years_experience: int | None = None
    education: EducationRequirement | None = None
    languages: list[str] | None = None  # None = not extracted
    warnings: list[str] = []  # codes only


class RequirementExtractor(Protocol):
    def extract(self, posting: SourcePosting) -> Requirements:
        """May raise; the caller counts that posting as failed and goes on."""


class TagsExtractor:
    """Required skills = the posting's tags, cleaned: trimmed, blanks and over-long entries
    dropped, repeats (ignoring case) removed, at most ``MAX_TAG_SKILLS``."""

    def extract(self, posting: SourcePosting) -> Requirements:
        seen: set[str] = set()
        skills: list[str] = []
        for tag in posting.tags:
            clean = " ".join(tag.split())
            if not clean or len(clean) > MAX_TAG_CHARS or clean.casefold() in seen:
                continue
            seen.add(clean.casefold())
            skills.append(clean)
        return Requirements(
            required_skills=skills[:MAX_TAG_SKILLS],
            warnings=[TAGS_FALLBACK] if skills else [TAGS_FALLBACK, NO_REQUIREMENTS],
        )
