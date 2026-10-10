"""Entry point for CV parsing: dispatch by file type and assemble offsets."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from jobpdf.extraction.models import ParsedDocument, ParseError, RawBlock, SourceType, TextBlock
from jobpdf.extraction.parse_docx import parse_docx
from jobpdf.extraction.parse_pdf import parse_pdf

MAX_FILE_BYTES = 20 * 1024 * 1024
BLOCK_SEPARATOR = "\n\n"

FormatResult = tuple[list[RawBlock], bool, list[str]]
FormatParser = Callable[[Path], FormatResult]


_PARSERS: dict[str, tuple[SourceType, FormatParser]] = {
    ".pdf": ("pdf", parse_pdf),
    ".docx": ("docx", parse_docx),
}


def parse(path: str | Path) -> ParsedDocument:
    """Parse a CV file into ordered text blocks with offsets into ``full_text``.

    Size is checked before opening so oversized uploads are rejected without
    handing them to a native parser.
    """
    path = Path(path)
    entry = _PARSERS.get(path.suffix.lower())
    if entry is None:
        raise ParseError("unsupported_type", f"Unsupported file type: {path.suffix or '<none>'}")
    source_type, format_parser = entry

    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ParseError("too_large", f"File is {size} bytes; limit is {MAX_FILE_BYTES}")

    raw_blocks, has_text_layer, warnings = format_parser(path)
    return _assemble(source_type, raw_blocks, has_text_layer, warnings)


def _assemble(
    source_type: SourceType,
    raw_blocks: list[RawBlock],
    has_text_layer: bool,
    warnings: list[str],
) -> ParsedDocument:
    """Join blocks and compute offsets in one place, shared by all formats."""
    blocks: list[TextBlock] = []
    parts: list[str] = []
    cursor = 0
    for raw in raw_blocks:
        # A separator inside a block would make block boundaries ambiguous downstream.
        if not raw.text or BLOCK_SEPARATOR in raw.text:
            raise ValueError(f"Block text must be non-empty without {BLOCK_SEPARATOR!r}")
        if parts:
            parts.append(BLOCK_SEPARATOR)
            cursor += len(BLOCK_SEPARATOR)
        start, end = cursor, cursor + len(raw.text)
        parts.append(raw.text)
        cursor = end
        blocks.append(
            TextBlock(
                text=raw.text,
                page=raw.page,
                bbox=raw.bbox,
                font_size=raw.font_size,
                is_bold=raw.is_bold,
                start=start,
                end=end,
            )
        )
    full_text = "".join(parts)
    return ParsedDocument(
        source_type=source_type,
        blocks=blocks,
        full_text=full_text,
        has_text_layer=has_text_layer,
        warnings=list(warnings),
    )
