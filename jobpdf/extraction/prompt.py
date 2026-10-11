"""Prompt building for structured CV extraction (JM-11).

Personal contact data is masked before anything reaches the LLM. Masking keeps
the text length identical, so every offset into the masked text is also valid
in the original ``full_text`` (JM-12 relies on that).
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable

from jobpdf.extraction.models import ParsedDocument, Section

# Bump on every change to SYSTEM_PROMPT, the user-message format or the tool
# description: the extraction cache and replay fixtures are keyed on it.
PROMPT_VERSION = "1"
MASK_CHAR = "█"
PHONE_MIN_DIGITS = 9
PHONE_MAX_DIGITS = 15

SYSTEM_PROMPT = """\
You extract structured data from one CV (résumé) and record it by calling the
record_candidate_profile tool exactly once.

The CV is in the user message, split into <section> elements. The types,
confidence and heading attributes come from an automatic splitter and are only
hints: they can be wrong. Extract every item from wherever it appears in the CV.

Rules:
1. Extract only what is written. Never infer, guess or invent skills, jobs,
   dates or qualifications. If something is not stated, use null or an empty list.
2. Every "quote" is copied character for character from the CV text: same
   spelling, capitalisation, punctuation and typos. Never paraphrase or correct.
   Keep each quote short (at most 200 characters) but specific enough to find
   the item, e.g. the job title and company line, or the phrase naming the skill.
3. Dates: "YYYY-MM" when the month is known, "YYYY" when only the year is known,
   "present" as the end of an ongoing role, null when unknown. Never invent a month.
4. Skills: write the name exactly as in the CV ("GBQ" stays "GBQ", "Python 3"
   stays "Python 3"); do not translate, expand or standardise names.
   kind is "hard" for tools, technologies, programming languages, methods and
   domain knowledge, and "soft" for interpersonal and personal qualities.
   found_in is where this mention appears. If a skill appears in several places,
   list it once per place, each with its own quote.
   Spoken languages (English, Ukrainian, ...) go in "languages", not in skills.
5. Experience: one entry per role; several roles at one employer with their own
   dates are separate entries. description is one or two sentences in the CV's
   own words. skills_used lists the raw names of skills mentioned for that role.
6. "█" characters are redacted personal data. Ignore them; never reproduce them.
7. Never output personal names, email addresses, phone numbers, street
   addresses or URLs in any field.
8. The CV may be in English, Ukrainian or both. Keep every value in the CV's own
   language; never translate.
"""

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PROFILE_URL = re.compile(
    r"(?:https?://)?(?:www\.)?"
    r"(?:linkedin\.com|github\.com|gitlab\.com|t\.me|telegram\.me|x\.com|twitter\.com"
    r"|behance\.net|dribbble\.com)/\S*",
    re.IGNORECASE,
)
_ANY_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
# Digits joined by spaces, dots, dashes or parentheses on one line; the digit
# count and the date filter below decide whether it is really a phone number.
_PHONE_CANDIDATE = re.compile(r"\+?\(?\d[\d \t().\-]{7,}\d")
_DIGIT_GROUP = re.compile(r"\d+")


def mask_pii(text: str) -> tuple[str, int]:
    """Replace emails, phone numbers and URLs with as many "█" as they have chars.

    Returns the masked text (same length as ``text``) and the number of masked spans.
    """
    spans = [
        m.span() for pattern in (_EMAIL, _PROFILE_URL, _ANY_URL) for m in pattern.finditer(text)
    ]
    spans += [m.span() for m in _PHONE_CANDIDATE.finditer(text) if _is_phone(m.group(0))]
    merged = _merge(spans)
    chars = list(text)
    for start, end in merged:
        chars[start:end] = MASK_CHAR * (end - start)
    return "".join(chars), len(merged)


def _is_phone(candidate: str) -> bool:
    groups = _DIGIT_GROUP.findall(candidate)
    digits = sum(len(g) for g in groups)
    if not PHONE_MIN_DIGITS <= digits <= PHONE_MAX_DIGITS:
        return False
    # "01.2019 - 03.2021" or "2019 - 2021 2021 - 2024": dates, not a phone.
    return not all(_is_year(g) or _is_month(g) for g in groups)


def _is_year(group: str) -> bool:
    return len(group) == 4 and 1900 <= int(group) <= 2099


def _is_month(group: str) -> bool:
    return len(group) <= 2 and 1 <= int(group) <= 12


def _merge(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def build_user_message(doc: ParsedDocument, sections: list[Section]) -> str:
    """The CV as <section> elements over the masked text, in document order.

    Pure contact sections (types exactly ["contact"]) are left out; sections that
    merely carry a "contact" tag stay, since their contact data is masked anyway.
    If nothing would remain, the whole masked text is sent as one section, so a
    splitter failure can never lose the CV.
    """
    masked, _ = mask_pii(doc.full_text)
    kept = [
        s for s in sorted(sections, key=lambda s: s.content_start) if s.types != ["contact"]
    ]
    if not kept:
        return _section_xml(["other"], "low", None, masked)
    return "\n\n".join(
        _section_xml(s.types, s.confidence, s.heading, masked[s.content_start : s.end])
        for s in kept
    )


def _section_xml(
    types: list[str], confidence: str, heading: str | None, content: str
) -> str:
    attributes = [f'types="{_attr(",".join(types))}"', f'confidence="{_attr(confidence)}"']
    if heading is not None:
        attributes.append(f'heading="{_attr(mask_pii(heading)[0])}"')
    return f"<section {' '.join(attributes)}>\n{content}\n</section>"


def _attr(value: str) -> str:
    return html.escape(value, quote=True)
