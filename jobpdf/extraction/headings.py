"""Heading normalisation and keyword matching for section chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from jobpdf.extraction.models import LabelSource, SectionType
from jobpdf.extraction.section_keywords import SECTION_KEYWORDS

# Minimum rapidfuzz ratio for a fuzzy keyword match ("experiences" ~ "experience").
FUZZY_MIN = 90
# Shorter parts only match exactly: fuzzy matching short words is too noisy.
FUZZY_MIN_LEN = 4

_LEADING_MARKER = re.compile(r"^(?:(?:\d{1,2}|[ivx]{1,4})[.)]|[•●▪■◦○♦►▸*·–—-])\s*")
_TRAILING_MARKER = re.compile(r"\s*[:–—-]+$")
_APOSTROPHES = re.compile(r"['’ʼ`]")
# Keep the separators split_parts() needs; any other punctuation becomes a space.
_PUNCTUATION = re.compile(r"[^\w\s&/|,]")
_SPACES = re.compile(r"\s+")
_PART_SEPARATOR = re.compile(r"\s*[&/|,]\s*|\s+(?:and|та|і)\s+")


@dataclass(frozen=True)
class HeadingMatch:
    types: list[SectionType]
    source: LabelSource  # "keyword" or "fuzzy"


def normalize_heading(text: str) -> str:
    """Reduce a heading to the plain lowercase form used for keyword lookup."""
    text = text.strip().lower()
    text = _LEADING_MARKER.sub("", text)
    text = _TRAILING_MARKER.sub("", text)
    text = _collapse_letter_spacing(text)
    text = _APOSTROPHES.sub("", text)
    text = _PUNCTUATION.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def _collapse_letter_spacing(text: str) -> str:
    # JM-8's cleaning already turned "W O R K   E X P E R I E N C E" into
    # single-spaced letters, so word gaps are gone: join everything and rely on
    # the space-free keyword index. Only all-single-character text is touched,
    # so a normal "work experience" keeps its space.
    tokens = text.split()
    if len(tokens) >= 3 and all(len(token) == 1 for token in tokens):
        return "".join(tokens)
    return text


def split_parts(normalized: str) -> list[str]:
    """Split a compound heading ("education & experience") into parts, deduped."""
    parts = (part.strip() for part in _PART_SEPARATOR.split(normalized))
    return list(dict.fromkeys(part for part in parts if part))


def _build_index() -> tuple[dict[str, SectionType], dict[str, SectionType]]:
    exact: dict[str, SectionType] = {}
    despaced: dict[str, SectionType] = {}
    for section_type, keywords in SECTION_KEYWORDS.items():
        for keyword in keywords:
            key = normalize_heading(keyword)
            exact[key] = section_type
            despaced[key.replace(" ", "")] = section_type
    return exact, despaced


_EXACT, _DESPACED = _build_index()
_FUZZY_CHOICES = {**_DESPACED, **_EXACT}


def match_part(part: str) -> tuple[SectionType, LabelSource] | None:
    """Exact match, then space-free match, then fuzzy for longer parts."""
    if part in _EXACT:
        return _EXACT[part], "keyword"
    despaced = part.replace(" ", "")
    if despaced in _DESPACED:
        return _DESPACED[despaced], "keyword"
    if len(part) < FUZZY_MIN_LEN:
        return None
    best = process.extractOne(
        part, list(_FUZZY_CHOICES), scorer=fuzz.ratio, score_cutoff=FUZZY_MIN
    )
    if best is None:
        return None
    return _FUZZY_CHOICES[best[0]], "fuzzy"


def match_heading(text: str) -> HeadingMatch | None:
    """Match a heading's parts against the dictionary.

    One matching part is enough. Unmatched parts add "other" rather than being
    dropped, so "Education & Volunteering" doesn't claim to be pure education.
    """
    matches = [match_part(part) for part in split_parts(normalize_heading(text))]
    if not any(matches):
        return None
    types: list[SectionType] = []
    for match in matches:
        section_type: SectionType = match[0] if match else "other"
        if section_type not in types:
            types.append(section_type)
    fuzzy = any(match is not None and match[1] == "fuzzy" for match in matches)
    return HeadingMatch(types=types, source="fuzzy" if fuzzy else "keyword")
