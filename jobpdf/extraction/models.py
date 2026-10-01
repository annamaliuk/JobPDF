"""Data contract for parsed CV documents.

JM-10 (section chunking) relies on ``font_size``/``is_bold`` to spot headers and
JM-12 (grounding) relies on ``start``/``end`` pointing into ``full_text``, so
these fields are part of a cross-ticket contract — change them deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SourceType = Literal["pdf", "docx"]
ParseErrorReason = Literal[
    "encrypted", "corrupt", "unsupported_type", "too_large", "too_many_pages"
]
BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class RawBlock:
    """A block as a format parser sees it, before offsets exist.

    Format parsers return these so that offsets are assigned in exactly one
    place (``parser._assemble``) and can never drift from ``full_text``.
    """

    text: str
    page: int | None
    bbox: BBox | None
    font_size: float | None
    is_bold: bool


@dataclass(frozen=True)
class TextBlock:
    """One block of text; invariant: ``full_text[start:end] == text``."""

    text: str
    page: int | None
    bbox: BBox | None
    font_size: float | None
    is_bold: bool
    start: int
    end: int


@dataclass
class ParsedDocument:
    """A CV as ordered text blocks plus the joined text they index into."""

    source_type: SourceType
    blocks: list[TextBlock]
    full_text: str
    has_text_layer: bool
    warnings: list[str] = field(default_factory=list)


class ParseError(Exception):
    """Raised when a file cannot or must not be parsed.

    ``reason`` is machine-readable so the API can map it to a user message
    without parsing exception text.
    """

    def __init__(self, reason: ParseErrorReason, message: str = "") -> None:
        super().__init__(message or reason)
        self.reason: ParseErrorReason = reason
