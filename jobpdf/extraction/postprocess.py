"""Turn the LLM's ExtractedProfile into the final CandidateProfile (JM-11).

Nothing here drops data: problems (unfindable quotes, PII in the output,
undatable jobs) become warnings and the items stay. JM-12 does the real offset
location of quotes later.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

from jobpdf.extraction.prompt import MASK_CHAR, mask_pii
from jobpdf.extraction.schema import (
    CandidateProfile,
    ExtractedJob,
    ExtractedProfile,
    Skill,
    SkillMention,
)
from jobpdf.normalization.text import normalize

QUOTE_PREVIEW_CHARS = 60


def build_profile(
    extracted: ExtractedProfile, masked_text: str, today: dt.date
) -> tuple[CandidateProfile, list[str]]:
    months, warnings = total_experience_months(extracted.experience, today)
    warnings += check_quotes(extracted, masked_text)
    warnings += check_pii_leaks(extracted)
    profile = CandidateProfile(
        summary=extracted.summary,
        skills=dedupe_skills(extracted.skills),
        experience=extracted.experience,
        education=extracted.education,
        languages=extracted.languages,
        certifications=extracted.certifications,
        total_experience_months=months,
    )
    return profile, warnings


def total_experience_months(
    jobs: list[ExtractedJob], today: dt.date
) -> tuple[int | None, list[str]]:
    """Months of experience with overlapping jobs counted once (inclusive).

    A year-only start counts from January and a year-only end until December;
    "present" means ``today``. Jobs without a usable start or end are skipped
    with a warning rather than guessed. None when no job can be counted.
    """
    warnings: list[str] = []
    intervals: list[tuple[int, int]] = []
    for job in jobs:
        label = f"{job.title!r}"
        if job.start is None:
            warnings.append(f"experience {label}: no start date, not counted in total months")
            continue
        if job.end is None:
            warnings.append(f"experience {label}: no end date, not counted in total months")
            continue
        start = _month_index(job.start, is_end=False, today=today)
        end = _month_index(job.end, is_end=True, today=today)
        if end < start:
            warnings.append(f"experience {label}: ends before it starts, not counted")
            continue
        intervals.append((start, end))
    if not intervals:
        return None, warnings
    total = 0
    current_start, current_end = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start <= current_end + 1:  # overlapping or back-to-back months merge
            current_end = max(current_end, end)
        else:
            total += current_end - current_start + 1
            current_start, current_end = start, end
    total += current_end - current_start + 1
    return total, warnings


def _month_index(value: str, *, is_end: bool, today: dt.date) -> int:
    if value == "present":
        return today.year * 12 + today.month - 1
    year, _, month = value.partition("-")
    month_number = int(month) if month else (12 if is_end else 1)
    return int(year) * 12 + month_number - 1


def dedupe_skills(mentions: list[SkillMention]) -> list[Skill]:
    """One Skill per normalize(name): first spelling, merged places and quotes.

    If mentions disagree on kind, "hard" wins: a tool named once among soft
    skills is still a tool.
    """
    skills: dict[str, Skill] = {}
    for mention in mentions:
        key = normalize(mention.name)
        if not key:
            continue
        skill = skills.get(key)
        if skill is None:
            skills[key] = Skill(
                name=mention.name.strip(),
                kind=mention.kind,
                found_in=[mention.found_in],
                quotes=[mention.quote],
            )
            continue
        if mention.found_in not in skill.found_in:
            skill.found_in.append(mention.found_in)
        if mention.quote not in skill.quotes:
            skill.quotes.append(mention.quote)
        if mention.kind == "hard":
            skill.kind = "hard"
    return list(skills.values())


def check_quotes(extracted: ExtractedProfile, masked_text: str) -> list[str]:
    """Warn for each quote found in the masked CV neither exactly nor whitespace-collapsed."""
    collapsed_text = _collapse(masked_text)
    warnings = []
    for quote in _quotes(extracted):
        if quote in masked_text or _collapse(quote) in collapsed_text:
            continue
        warnings.append(f"quote not found: {quote[:QUOTE_PREVIEW_CHARS]}")
    return warnings


def _quotes(extracted: ExtractedProfile) -> Iterator[str]:
    for item in (*extracted.skills, *extracted.experience, *extracted.education,
                 *extracted.certifications):
        yield item.quote


def _collapse(text: str) -> str:
    return " ".join(text.split())


def check_pii_leaks(extracted: ExtractedProfile) -> list[str]:
    """Warn if any output string contains an email, phone number, URL or redaction mark."""
    warnings = []
    for path, value in _strings(extracted.model_dump()):
        if mask_pii(value)[1] or MASK_CHAR in value:
            warnings.append(f"possible personal data in output at {path}")
    return warnings


def _strings(node: object, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _strings(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _strings(value, f"{path}[{i}]")
