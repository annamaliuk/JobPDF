"""Entry point for CV parsing: dispatch by file type and assemble offsets."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import pymupdf

from jobpdf.extraction.models import ParsedDocument, ParseError, RawBlock, SourceType, TextBlock
from jobpdf.extraction.ocr import OCR_DPI
from jobpdf.extraction.parse_docx import parse_docx
from jobpdf.extraction.parse_pdf import parse_pdf

MAX_FILE_BYTES = 20 * 1024 * 1024
BLOCK_SEPARATOR = "\n\n"

# (blocks, has_text_layer, warnings) plus, for PDFs, the 0-based OCR'd pages (JM-9).
FormatResult = tuple[list[RawBlock], bool, list[str]] | tuple[
    list[RawBlock], bool, list[str], list[int]
]
FormatParser = Callable[[Path], FormatResult]


def parse_image(path: Path) -> FormatResult:
    """A photo or scan of a CV (JM-9): wrap it in a one-page PDF and parse that.

    Without OCR (developer machines) this degrades exactly like a scanned PDF.
    """
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "image.pdf"
        pdf_path.write_bytes(image_to_pdf(path))
        return parse_pdf(pdf_path)


def image_to_pdf(path: Path) -> bytes:
    """One page that shows the image at exactly OCR_DPI.

    Not plain convert_to_pdf(): that sizes the page from the image's DPI tag,
    and photos/screenshots tagged 72 dpi would become poster-sized pages that
    OCR renders at ~4x the real resolution (slow, memory-hungry, no gain).
    """
    try:
        pixmap = pymupdf.Pixmap(str(path))
    except Exception as exc:  # unreadable or not really an image
        raise ParseError("corrupt", f"Cannot open image: {exc}") from exc
    doc = pymupdf.open()
    try:
        scale = 72 / OCR_DPI
        page = doc.new_page(width=pixmap.width * scale, height=pixmap.height * scale)
        page.insert_image(page.rect, pixmap=pixmap)
        return doc.tobytes(garbage=3, deflate=True)
    finally:
        doc.close()


_PARSERS: dict[str, tuple[SourceType, FormatParser]] = {
    ".pdf": ("pdf", parse_pdf),
    ".docx": ("docx", parse_docx),
    # Images become a one-page PDF, so they report source_type "pdf".
    ".png": ("pdf", parse_image),
    ".jpg": ("pdf", parse_image),
    ".jpeg": ("pdf", parse_image),
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

    raw_blocks, has_text_layer, warnings, *rest = format_parser(path)
    ocr_pages = rest[0] if rest else []
    return _assemble(source_type, raw_blocks, has_text_layer, warnings, ocr_pages)


def _assemble(
    source_type: SourceType,
    raw_blocks: list[RawBlock],
    has_text_layer: bool,
    warnings: list[str],
    ocr_pages: Sequence[int] = (),
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
        ocr_pages=list(ocr_pages),
    )
