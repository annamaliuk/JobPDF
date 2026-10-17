"""What a posting asks for, as plain strings (before normalization).

``RulesExtractor`` (JM-48) reads the description with word lists, regular expressions and
the skill taxonomy (see ``requirement_rules``): required and nice-to-have skills,
responsibilities, minimum years, education and languages, each with the line it came from.
Postings without a usable description fall back to ``TagsExtractor``, which reads the
source's tags only, so their responsibilities stay empty (the embedding text is then the
title).

Each extractor has a ``version``; it is stored with the vacancy, so changing the rules (or
switching extractor) rebuilds the stored vacancies once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from jobpdf.ingestion import requirement_rules as rules
from jobpdf.ingestion.posting_text import is_usable, posting_lines
from jobpdf.ingestion.sources.models import SourcePosting
from jobpdf.ranking.models import EducationRequirement

TAGS_VERSION = "tags"
RULES_VERSION = "rules-1"  # bump when a rule or word list changes

# Warning codes.
TAGS_FALLBACK = "tags_fallback"
NO_REQUIREMENTS = "no_requirements"
TAGS_ADDED = "tags_added"  # the description named few skills, so the tags were added
PARTIAL_VOCABULARY = "partial_vocabulary"  # no taxonomy build: custom skills only
FEW_SKILLS = 3  # fewer skills than this found in the text: the tags are added
MAX_SKILLS = 25
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
    quotes: dict[str, str] = {}  # item -> the posting line it came from; never stored


class RequirementExtractor(Protocol):
    # Stamped on each stored vacancy (``extraction_version``); None = not tracked.
    version: str | None

    def extract(self, posting: SourcePosting) -> Requirements:
        """May raise; the caller counts that posting as failed and goes on."""


class TagsExtractor:
    """Required skills = the posting's tags, cleaned: trimmed, blanks and over-long entries
    dropped, repeats (ignoring case) removed, at most ``MAX_TAG_SKILLS``."""

    version = TAGS_VERSION

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


class RulesExtractor:
    """Requirements read from the description by rules (JM-48); tags when it is unusable.

    Skills are found by the taxonomy vocabulary, loaded once (``SkillVocabulary.load``).
    A skill is nice-to-have when it sits in a nice-to-have section or in a sentence with a
    cue such as "is a plus"; anything named in both lists stays required. Years, education
    and languages are read from requirement and general lines only; lines that only state a
    preference never set them. When the text names fewer than ``FEW_SKILLS`` skills, the
    source's tags are added as required skills (``tags_added``).
    """

    version = RULES_VERSION

    def __init__(self, vocabulary: rules.SkillVocabulary | None = None,
                 taxonomy_dir: Path = Path("data/taxonomy")) -> None:
        self._vocabulary = vocabulary
        self._taxonomy_dir = taxonomy_dir
        self._tags = TagsExtractor()

    @property
    def vocabulary(self) -> rules.SkillVocabulary:
        if self._vocabulary is None:
            self._vocabulary = rules.SkillVocabulary.load(self._taxonomy_dir)
        return self._vocabulary

    def extract(self, posting: SourcePosting) -> Requirements:
        lines = posting_lines(posting.description)
        if not is_usable(lines):
            return self._tags.extract(posting)
        vocabulary = self.vocabulary
        placed = rules.split_sections(lines)
        quotes: dict[str, str] = {}
        required: dict[str, str] = {}  # norm -> text as written
        preferred: dict[str, str] = {}
        years: list[int] = []
        education: EducationRequirement | None = None
        languages: list[str] = []
        for item in placed:
            if item.section == rules.OTHER:
                continue
            nice = item.section == rules.NICE_TO_HAVE
            for hit in vocabulary.scan(item.line.text, preferred=nice):
                if rules.is_language_name(hit.text):
                    continue
                target = preferred if hit.preferred else required
                if hit.norm not in required and hit.norm not in target:
                    target[hit.norm] = hit.text
                    quotes.setdefault(hit.text, hit.quote)
            if nice or item.section == rules.RESPONSIBILITIES:
                continue
            for sentence in rules.sentences(item.line.text):
                if rules.has_preferred_cue(sentence):
                    continue
                found = rules.years_in(sentence)
                if found:
                    years += found
                    quotes.setdefault(f"years:{max(found)}", sentence)
                if education is None and (hit := rules.education_in(sentence)):
                    education = EducationRequirement(level=hit.level, field=hit.field,
                                                     quote=hit.quote)
                for language in rules.languages_in(sentence):
                    if language not in languages:
                        languages.append(language)
                        quotes.setdefault(f"language:{language}", sentence)
        for norm in list(preferred):
            if norm in required:
                del preferred[norm]

        warnings: list[str] = []
        skills = list(required.values())[:MAX_SKILLS]
        if len(skills) + len(preferred) < FEW_SKILLS:
            known = {s.casefold() for s in [*skills, *preferred.values()]}
            extra = [t for t in self._tags.extract(posting).required_skills
                     if t.casefold() not in known]
            if extra:
                skills = (skills + extra)[:MAX_SKILLS]
                warnings.append(TAGS_ADDED)
        if not vocabulary.complete:
            warnings.append(PARTIAL_VOCABULARY)
        if not skills and not preferred:
            warnings.append(NO_REQUIREMENTS)
        return Requirements(
            required_skills=skills,
            preferred_skills=list(preferred.values())[:MAX_SKILLS],
            responsibilities=rules.responsibilities_of(placed),
            min_years_experience=max(years) if years else None,
            education=education,
            languages=languages,
            warnings=warnings,
            quotes=quotes,
        )
