"""Split a ParsedDocument into typed sections (JM-10).

Heading keywords and the document's own learned heading style decide section
boundaries. Formatting alone never does: job titles and company names are
short and bold too. Section types are hints for JM-11, which sees the full
text, so every block must land in exactly one section and nothing is dropped.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Literal, NamedTuple

from jobpdf.extraction.headings import HeadingMatch, match_heading, normalize_heading
from jobpdf.extraction.models import (
    Confidence,
    LabelSource,
    ParsedDocument,
    Section,
    SectionType,
    TextBlock,
)

# Heading candidates: short, single-line, not sentence- or list-like.
MAX_HEADING_WORDS = 5
MAX_HEADING_CHARS = 40
# A block this much larger than the body text counts as a "large font" signal.
FONT_RATIO = 1.15
# If more candidates than this share the heading style, the style isn't
# distinctive (job titles probably share it), so unknown headings are not guessed.
STYLE_GUARD_MAX = 12
# Font sizes within this distance count as the same size.
SIZE_TOLERANCE_PT = 0.5
# Fewer digits than this is a date range or an ID, not a phone number.
PHONE_MIN_DIGITS = 9

NO_HEADINGS_WARNING = "No section headings found; the whole CV is one 'other' section."
STYLE_SKIPPED_WARNING = (
    "Fewer than 2 keyword headings; heading style not learned, so unknown "
    "headings and sub-headings were not detected."
)
STYLE_GUARD_WARNING = (
    "Heading style is shared by too many blocks to be distinctive; "
    "unknown headings were not detected."
)

_BULLETS = "•●▪■◦○♦►▸*·–—-"
# "Skills: Python, SQL": short heading part, a separator, then content.
_INLINE_HEADING = re.compile(
    rf"^([^\n:–—-]{{1,{MAX_HEADING_CHARS}}}?)\s*[:–—-]\s*(\S.*)$", re.DOTALL
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"\+?\(?\d[\d\s().-]{6,}\d")
_PROFILE_URL = re.compile(r"(?:linkedin\.com/in|github\.com|t\.me|telegram\.me)/\S+", re.I)
_YEAR = re.compile(r"(?:19|20)\d\d")

HeadingKind = Literal["standalone", "inline", "first_line", "style"]


class Fingerprint(NamedTuple):
    size: int | None
    bold: bool
    caps: bool


@dataclass(frozen=True)
class _Heading:
    kind: HeadingKind
    text: str
    types: list[SectionType]
    confidence: Confidence
    source: LabelSource
    end: int  # end offset of the heading text, used for empty sections
    content: TextBlock | None = None  # remainder of a split inline/first-line block


@dataclass
class _Draft:
    heading: _Heading | None
    blocks: list[TextBlock] = field(default_factory=list)


def split_sections(doc: ParsedDocument) -> tuple[list[Section], list[str]]:
    """Cut the ordered blocks into typed sections.

    Returns the sections and section-level warnings. Warnings are returned
    rather than appended to ``doc.warnings`` so the document is never mutated
    and repeated calls can't duplicate them.
    """
    warnings: list[str] = []
    headings = _find_headings(doc.blocks, warnings)
    drafts = _assemble(doc.blocks, headings)

    if not any(d.heading for d in drafts):
        if doc.blocks:
            warnings.append(NO_HEADINGS_WARNING)
        sections = [_fallback_section(d) for d in drafts]
    else:
        sections = [_to_section(d) for d in drafts]
    return _split_off_contacts(sections), warnings


# --- heading detection -------------------------------------------------------


def _find_headings(blocks: list[TextBlock], warnings: list[str]) -> dict[int, _Heading]:
    """Decide which blocks start a section; keys are block indexes."""
    body_size = body_font_size(blocks)
    standalone = {
        i: match for i, b in enumerate(blocks)
        if is_candidate(b.text) and (match := match_heading(b.text)) is not None
    }
    level1 = _learn_level1([fingerprint(blocks[i]) for i in standalone], warnings)
    detect_unknown = level1 is not None and _style_is_distinctive(blocks, level1, warnings)

    headings: dict[int, _Heading] = {}
    seen_keyword_heading = False
    for i, block in enumerate(blocks):
        heading = _keyword_heading(block, standalone.get(i), level1, body_size)
        if heading is None and detect_unknown and seen_keyword_heading:
            heading = _style_heading(block, level1)
        if heading is not None:
            headings[i] = heading
            seen_keyword_heading = seen_keyword_heading or heading.kind != "style"
    return headings


def _keyword_heading(
    block: TextBlock,
    standalone_match: HeadingMatch | None,
    level1: Fingerprint | None,
    body_size: float | None,
) -> _Heading | None:
    fp = fingerprint(block)
    if standalone_match is not None:
        if level1 is not None and is_weaker(fp, level1):
            return None  # sub-heading: stays in the current section
        return _Heading(
            kind="standalone",
            text=block.text,
            types=standalone_match.types,
            confidence=_confidence(block, block.text, body_size),
            source=standalone_match.source,
            end=block.end,
        )

    inline = _INLINE_HEADING.match(block.text)
    if inline and is_candidate(inline.group(1)):
        match = match_heading(inline.group(1))
        if match is not None:
            if level1 is not None and is_weaker(fp, level1):
                return None  # e.g. "Languages: Python, Java" inside Skills
            return _Heading(
                kind="inline",
                text=inline.group(1),
                types=match.types,
                confidence=_confidence(block, inline.group(1), body_size),
                source=match.source,
                end=block.start + inline.end(1),
                content=_tail(block, inline.start(2)),
            )

    if "\n" in block.text:
        first_line = block.text.split("\n", 1)[0]
        match = match_heading(first_line) if is_candidate(first_line) else None
        if match is not None:
            # PyMuPDF merged the heading with the next line, so the block's
            # formatting describes both lines: we can't verify it's a heading style.
            return _Heading(
                kind="first_line",
                text=first_line,
                types=match.types,
                confidence="low",
                source=match.source,
                end=block.start + len(first_line),
                content=_tail(block, len(first_line) + 1),
            )
    return None


def _style_heading(block: TextBlock, level1: Fingerprint | None) -> _Heading | None:
    if is_candidate(block.text) and fingerprint(block) == level1:
        return _Heading(
            kind="style",
            text=block.text,
            types=["other"],
            confidence="low",
            source="style",
            end=block.end,
        )
    return None


def _learn_level1(fingerprints: list[Fingerprint], warnings: list[str]) -> Fingerprint | None:
    """Most common keyword-heading style; ties go to the strongest style."""
    if len(fingerprints) < 2:
        # With zero headings the "no headings" warning already says it all.
        if fingerprints:
            warnings.append(STYLE_SKIPPED_WARNING)
        return None
    counts = Counter(fingerprints)
    return max(counts, key=lambda fp: (counts[fp], fp.size or 0, fp.bold, fp.caps))


def _style_is_distinctive(
    blocks: list[TextBlock], level1: Fingerprint, warnings: list[str]
) -> bool:
    shared = sum(1 for b in blocks if is_candidate(b.text) and fingerprint(b) == level1)
    if shared > STYLE_GUARD_MAX:
        warnings.append(STYLE_GUARD_WARNING)
        return False
    return True


def is_candidate(text: str) -> bool:
    """Could this text be a heading, judging by its shape alone?"""
    text = text.strip()
    if not text or "\n" in text or len(text) > MAX_HEADING_CHARS:
        return False
    if text[-1] in ".," or text[0] in _BULLETS:
        return False
    words = normalize_heading(text).split()
    return 0 < len(words) <= MAX_HEADING_WORDS


def fingerprint(block: TextBlock) -> Fingerprint:
    size = round(block.font_size) if block.font_size is not None else None
    return Fingerprint(size=size, bold=block.is_bold, caps=is_all_caps(block.text))


def is_weaker(fp: Fingerprint, level1: Fingerprint) -> bool:
    """Smaller font, or same size but missing bold/caps that level-1 headings have."""
    if fp.size is not None and level1.size is not None:
        if fp.size < level1.size - SIZE_TOLERANCE_PT:
            return True
        if fp.size > level1.size + SIZE_TOLERANCE_PT:
            return False
    return (level1.bold and not fp.bold) or (level1.caps and not fp.caps)


def is_all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def body_font_size(blocks: list[TextBlock]) -> float | None:
    """Character-weighted median font size: what most of the text looks like."""
    sized = sorted((b.font_size, len(b.text)) for b in blocks if b.font_size is not None)
    total = sum(weight for _, weight in sized)
    seen = 0
    for size, weight in sized:
        seen += weight
        if seen * 2 >= total:
            return size
    return None


def _confidence(block: TextBlock, heading_text: str, body_size: float | None) -> Confidence:
    large = (
        block.font_size is not None
        and body_size is not None
        and block.font_size >= FONT_RATIO * body_size
    )
    signals = sum([block.is_bold, large, is_all_caps(heading_text)])
    return "high" if signals >= 2 else "low"


def _tail(block: TextBlock, offset: int) -> TextBlock:
    """The part of a block after an inline heading, with exact offsets."""
    return replace(block, text=block.text[offset:], start=block.start + offset)


# --- assembly ----------------------------------------------------------------


def _assemble(blocks: list[TextBlock], headings: dict[int, _Heading]) -> list[_Draft]:
    drafts = [_Draft(heading=None)]
    for i, block in enumerate(blocks):
        heading = headings.get(i)
        if heading is None:
            drafts[-1].blocks.append(block)
            continue
        drafts.append(_Draft(heading=heading))
        if heading.content is not None:
            drafts[-1].blocks.append(heading.content)
    if not drafts[0].blocks:
        drafts.pop(0)  # nothing before the first heading
    return drafts


def _to_section(draft: _Draft) -> Section:
    if draft.heading is None:
        # Text before the first heading: usually name and contact details.
        if any(is_contact_block(b, majority=False) for b in draft.blocks):
            return _section(draft.blocks, ["contact"], None, "high", "pattern")
        return _section(draft.blocks, ["summary"], None, "low", "fallback")
    h = draft.heading
    return _section(draft.blocks, list(h.types), h.text, h.confidence, h.source, empty_at=h.end)


def _fallback_section(draft: _Draft) -> Section:
    return _section(draft.blocks, ["other"], None, "low", "fallback")


def _section(
    blocks: list[TextBlock],
    types: list[SectionType],
    heading: str | None,
    confidence: Confidence,
    source: LabelSource,
    empty_at: int = 0,
) -> Section:
    start = blocks[0].start if blocks else empty_at
    end = blocks[-1].end if blocks else empty_at
    return Section(
        types=types,
        heading=heading,
        blocks=blocks,
        content_start=start,
        end=end,
        confidence=confidence,
        label_source=source,
    )


# --- unlabelled contact details ------------------------------------------------


def _split_off_contacts(sections: list[Section]) -> list[Section]:
    """Give unlabelled contact blocks their own section, or tag their section.

    A run of contact blocks at the end of a section (bottom of a sidebar, a
    footer) becomes its own "contact" section. Contact blocks elsewhere stay put,
    since splitting would cut the section in two; the section gets a "contact"
    type instead.
    """
    result: list[Section] = []
    for section in sections:
        if "contact" in section.types:
            result.append(section)
            continue
        flags = [is_contact_block(b) for b in section.blocks]
        trailing = len(flags) - next(
            (i for i in range(len(flags), 0, -1) if not flags[i - 1]), 0
        )
        if 0 < trailing < len(flags):
            kept, contacts = section.blocks[:-trailing], section.blocks[-trailing:]
            result.append(replace(section, blocks=kept, end=kept[-1].end))
            result.append(_section(contacts, ["contact"], None, "high", "pattern"))
        elif any(flags):
            result.append(replace(section, types=[*section.types, "contact"]))
        else:
            result.append(section)
    return result


def is_contact_block(block: TextBlock, majority: bool = True) -> bool:
    """Does the block consist of contact lines (email, phone, profile URL)?

    With ``majority`` most lines must be contact lines, so a paragraph that
    merely mentions an email doesn't count; otherwise one line is enough.
    """
    lines = block.text.split("\n")
    contact_lines = sum(1 for line in lines if is_contact_line(line))
    if majority:
        return contact_lines * 2 > len(lines)
    return contact_lines > 0


def is_contact_line(line: str) -> bool:
    if _EMAIL.search(line) or _PROFILE_URL.search(line):
        return True
    return any(_looks_like_phone(m.group(0)) for m in _PHONE.finditer(line))


def _looks_like_phone(candidate: str) -> bool:
    digits = re.sub(r"\D", "", candidate)
    if len(digits) < PHONE_MIN_DIGITS:
        return False
    # "2019 - 2021 2021 - 2024" has enough digits but is only years.
    groups = re.findall(r"\d+", candidate)
    return not all(len(g) == 4 and _YEAR.fullmatch(g) for g in groups)
