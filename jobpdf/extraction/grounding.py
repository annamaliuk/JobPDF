"""Grounding: locate every extracted quote in the CV text (JM-12).

The LLM's "verbatim" quotes differ cosmetically from the parsed text (collapsed
line breaks, straightened quotes, joined hyphenation), so plain ``find`` misses
many of them. Matching therefore runs on a normalized copy of the text that keeps
an index map back to the original, so every hit still yields offsets into
``ParsedDocument.full_text``.

This deliberately does not use ``normalization.text.normalize()``: that one is
for skill lookups and cache keys and cannot map positions back.
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel
from rapidfuzz import fuzz

from jobpdf.extraction.models import ParsedDocument, Section

# Fuzzy matching only accepts a long enough quote that aligns this well: short
# quotes ("Pyhton") fuzzy-match too many unrelated words to mean anything.
FUZZY_MIN_SCORE = 90
MIN_FUZZY_QUOTE_LEN = 12

WARNING_UNGROUNDED = "ungrounded_quote"
WARNING_FUZZY = "fuzzy_quote"
WARNING_AMBIGUOUS = "ambiguous_quote"
WARNING_LENGTH_MISMATCH = "searched_text_length_mismatch"

# The LLM sees the CV wrapped in <section ...>…</section> (prompt.build_user_message)
# and sometimes copies a tag into a quote.
_SECTION_TAG = re.compile(r"</?section\b[^>]*>")

# Which section types an item is expected in: by the list it sits in, and by a
# skill's found_in value (its own vocabulary; "other" says nothing).
_LIST_SECTION_TYPES = {
    "experience": {"experience"},
    "education": {"education"},
    "certifications": {"certifications"},
}
_FOUND_IN_SECTION_TYPE = {
    "skills_list": "skills",
    "experience": "experience",
    "projects": "projects",
    "education": "education",
    "summary": "summary",
}

# Quote and dash variants an LLM tends to straighten. Applied after NFKC, which
# already folds ligatures, full-width forms and the non-breaking space.
_CHAR_MAP = {
    **dict.fromkeys("‘’‚‛", "'"),
    **dict.fromkeys("“”„‟«»", '"'),
    **dict.fromkeys("‐‑‒–—―−", "-"),
}
# Invisible characters that never survive into a quote: soft hyphen, zero-width
# space / non-joiner / joiner, word joiner and BOM.
_INVISIBLE = frozenset(chr(c) for c in (0xAD, 0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF))
_LINE_BREAKS = frozenset("\n\r\v\f\x85  ")


class GroundingStatus(str, Enum):
    EXACT = "exact"
    NORMALIZED = "normalized"
    FUZZY = "fuzzy"
    UNGROUNDED = "ungrounded"


class Grounding(BaseModel):
    """Where one quote sits in ``full_text``; ``full_text[start:end] == matched_text``."""

    path: str  # e.g. "experience[2]" or "skills[5].quotes[1]"
    quote: str
    status: GroundingStatus
    start: int | None = None
    end: int | None = None
    matched_text: str | None = None
    score: float | None = None  # fuzzy only
    ambiguous: bool = False
    low_confidence: bool = False


class GroundingWarning(BaseModel):
    # Code and path only: warnings may reach logs, and quotes can hold CV text.
    code: str
    path: str | None = None


class GroundingResult(BaseModel):
    groundings: list[Grounding]
    counts: dict[GroundingStatus, int]
    warnings: list[GroundingWarning]


@dataclass(frozen=True)
class _Searchable:
    full_text: str
    searched: str  # what the LLM saw (masked), same length as full_text
    normalized: str
    index_map: list[int]
    sections: list[Section]

    def in_preferred(self, span: tuple[int, int], preferred: set[str]) -> bool:
        start, end = span
        return any(
            preferred.intersection(s.types) and s.content_start <= start and end <= s.end
            for s in self.sections
        )


def ground(
    extraction: BaseModel | Mapping[str, Any],
    doc: ParsedDocument,
    sections: list[Section],
    searched_text: str | None = None,
) -> GroundingResult:
    """Locate every quote of an extraction in ``doc.full_text``.

    ``extraction`` is walked generically (any object with a ``quote`` string or a
    ``quotes`` list), so CandidateProfile, ExtractedProfile and later schema
    versions all work. ``searched_text`` is the masked text the LLM saw; masking
    keeps the length, so its offsets are valid in ``full_text``. Unfindable
    quotes are kept as ``ungrounded`` with a warning, never dropped.
    """
    warnings: list[GroundingWarning] = []
    full_text = doc.full_text
    if searched_text is None:
        searched_text = full_text
    elif len(searched_text) != len(full_text):
        warnings.append(GroundingWarning(code=WARNING_LENGTH_MISMATCH))
        searched_text = full_text
    normalized, index_map = normalize_with_map(searched_text)
    text = _Searchable(full_text, searched_text, normalized, index_map, sections)

    data = extraction.model_dump() if isinstance(extraction, BaseModel) else extraction
    groundings = [
        _ground_quote(text, path, quote, preferred)
        for path, quote, preferred in _quoted_items(data)
    ]
    counts = dict.fromkeys(GroundingStatus, 0)
    for g in groundings:
        counts[g.status] += 1
        if g.status is GroundingStatus.UNGROUNDED:
            warnings.append(GroundingWarning(code=WARNING_UNGROUNDED, path=g.path))
        elif g.status is GroundingStatus.FUZZY:
            warnings.append(GroundingWarning(code=WARNING_FUZZY, path=g.path))
        if g.ambiguous:
            warnings.append(GroundingWarning(code=WARNING_AMBIGUOUS, path=g.path))
    return GroundingResult(groundings=groundings, counts=counts, warnings=warnings)


def _quoted_items(
    node: Any, path: str = "", list_name: str | None = None
) -> Iterator[tuple[str, str, set[str]]]:
    if isinstance(node, Mapping):
        quote, quotes = node.get("quote"), node.get("quotes")
        if isinstance(quote, str) or isinstance(quotes, list):
            preferred = _preferred_types(list_name, node.get("found_in"))
            if isinstance(quote, str):
                yield path, quote, preferred
            if isinstance(quotes, list):
                for j, q in enumerate(quotes):
                    if isinstance(q, str):
                        yield f"{path}.quotes[{j}]", q, preferred
        for key, value in node.items():
            if key in ("quote", "quotes"):
                continue
            child = f"{path}.{key}" if path else str(key)
            yield from _quoted_items(value, child, key if isinstance(value, list) else list_name)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _quoted_items(value, f"{path}[{i}]", list_name)


def _preferred_types(list_name: str | None, found_in: object) -> set[str]:
    preferred = set(_LIST_SECTION_TYPES.get(list_name or "", ()))
    places = found_in if isinstance(found_in, list) else [found_in]
    for place in places:
        if isinstance(place, str) and place in _FOUND_IN_SECTION_TYPE:
            preferred.add(_FOUND_IN_SECTION_TYPE[place])
    return preferred


def _ground_quote(text: _Searchable, path: str, quote: str, preferred: set[str]) -> Grounding:
    needle = _SECTION_TAG.sub("", quote).strip()

    def located(status: GroundingStatus, span: tuple[int, int], **extra: Any) -> Grounding:
        start, end = span
        return Grounding(
            path=path, quote=quote, status=status, start=start, end=end,
            matched_text=text.full_text[start:end],
            low_confidence=status is GroundingStatus.FUZZY, **extra,
        )

    if needle:
        hits = [(at, at + len(needle)) for at in _occurrences(text.searched, needle)]
        if hits:
            span, ambiguous = _pick(text, hits, preferred)
            return located(GroundingStatus.EXACT, span, ambiguous=ambiguous)

        normalized_needle = normalize_with_map(needle)[0].strip()
        if normalized_needle:
            hits = [
                span for at in _occurrences(text.normalized, normalized_needle)
                if (span := _to_original(text, at, at + len(normalized_needle)))
            ]
            if hits:
                span, ambiguous = _pick(text, hits, preferred)
                return located(GroundingStatus.NORMALIZED, span, ambiguous=ambiguous)

            if len(normalized_needle) >= MIN_FUZZY_QUOTE_LEN:
                fuzzy = _fuzzy_find(text, normalized_needle, preferred)
                if fuzzy is not None:
                    span, score = fuzzy
                    return located(GroundingStatus.FUZZY, span, score=score)

    return Grounding(path=path, quote=quote, status=GroundingStatus.UNGROUNDED,
                     low_confidence=True)


def _occurrences(haystack: str, needle: str) -> Iterator[int]:
    at = haystack.find(needle)
    while at != -1:
        yield at
        at = haystack.find(needle, at + 1)


def _pick(
    text: _Searchable, hits: list[tuple[int, int]], preferred: set[str]
) -> tuple[tuple[int, int], bool]:
    """One hit, or the only one inside an expected section; else the first, ambiguous."""
    if len(hits) == 1:
        return hits[0], False
    inside = [h for h in hits if text.in_preferred(h, preferred)]
    if len(inside) == 1:
        return inside[0], False
    return (inside or hits)[0], True


def _fuzzy_find(
    text: _Searchable, needle: str, preferred: set[str]
) -> tuple[tuple[int, int], float] | None:
    """Best alignment in the expected sections first, then in the whole document."""
    windows = [
        (bisect.bisect_left(text.index_map, s.content_start),
         bisect.bisect_left(text.index_map, s.end))
        for s in text.sections
        if preferred.intersection(s.types)
    ]
    for candidates in (windows, [(0, len(text.normalized))]):
        best: tuple[float, int, int] | None = None
        for lo, hi in candidates:
            # A window shorter than the quote would be aligned *inside* the quote
            # and score 100 for any substring, so it can't hold the quote.
            if hi - lo < len(needle):
                continue
            alignment = fuzz.partial_ratio_alignment(
                needle, text.normalized[lo:hi], score_cutoff=FUZZY_MIN_SCORE
            )
            if alignment is not None and (best is None or alignment.score > best[0]):
                best = (alignment.score, lo + alignment.dest_start, lo + alignment.dest_end)
        if best is not None:
            span = _to_original(text, best[1], best[2])
            if span is not None:
                return span, round(best[0], 2)
    return None


def _to_original(text: _Searchable, start: int, end: int) -> tuple[int, int] | None:
    """Map a normalized span back to ``full_text``; None if only whitespace is left."""
    if start >= end:
        return None
    original = text.searched
    begin, stop = text.index_map[start], text.index_map[end - 1] + 1
    while stop < len(original) and unicodedata.combining(original[stop]):
        stop += 1
    while begin < stop and original[begin].isspace():
        begin += 1
    while stop > begin and original[stop - 1].isspace():
        stop -= 1
    return (begin, stop) if begin < stop else None


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize ``text`` for matching; ``index_map[i]`` is where char ``i`` came from.

    Steps: NFKC, unified quotes and dashes, invisible characters dropped,
    line-break hyphenation resolved the way ``cleaning.clean_text`` does it, and
    every whitespace run collapsed to one space. The map is non-decreasing, so a
    normalized span ``[s, e)`` maps back to ``[map[s], map[e - 1] + 1)``.
    """
    chars, origin = _fold_characters(text)
    return _fold_whitespace(chars, origin)


def _fold_characters(text: str) -> tuple[list[str], list[int]]:
    # NFKC per base character plus its combining marks, so a decomposed "й"
    # (и + U+0306) composes like the precomposed one without merging clusters.
    chars: list[str] = []
    origin: list[int] = []
    i, n = 0, len(text)
    while i < n:
        j = i + 1
        while j < n and unicodedata.combining(text[j]):
            j += 1
        for ch in unicodedata.normalize("NFKC", text[i:j]):
            ch = _CHAR_MAP.get(ch, ch)
            if ch not in _INVISIBLE:
                chars.append(ch)
                origin.append(i)
        i = j
    return chars, origin


def _fold_whitespace(chars: list[str], origin: list[int]) -> tuple[str, list[int]]:
    out: list[str] = []
    out_map: list[int] = []
    k, n = 0, len(chars)
    while k < n:
        ch = chars[k]
        if ch.isspace():
            run_end = _skip_space(chars, k)
            out.append(" ")
            out_map.append(origin[k])
            k = run_end
            continue
        if ch == "-" and k > 0 and chars[k - 1].isalnum():
            run_end = _skip_space(chars, k + 1)
            breaks_line = any(c in _LINE_BREAKS for c in chars[k + 1 : run_end])
            if breaks_line and run_end < n:
                # "develop-\nment" was one word; "Front-\nEnd" and "2019-\n2021"
                # keep their hyphen (same rule as clean_text).
                if not (chars[k - 1].isalpha() and chars[run_end].islower()):
                    out.append("-")
                    out_map.append(origin[k])
                k = run_end
                continue
        out.append(ch)
        out_map.append(origin[k])
        k += 1
    return "".join(out), out_map


def _skip_space(chars: list[str], k: int) -> int:
    while k < len(chars) and chars[k].isspace():
        k += 1
    return k
