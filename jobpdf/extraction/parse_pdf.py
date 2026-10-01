"""PDF parsing with PyMuPDF: text blocks with font metadata, cleaned text."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf

from jobpdf.extraction.models import ParseError, RawBlock

MAX_PAGES = 15
# Below this many characters per page the PDF is almost certainly a scan.
MIN_CHARS_PER_PAGE = 50
NO_TEXT_LAYER_WARNING = "No text layer detected (likely a scanned PDF); OCR is out of scope."

_BOLD_FLAG = 16
# Some embedded fonts never set the bold flag, so fall back to the font name.
_BOLD_FONT_NAME = re.compile(r"bold|black|heavy", re.IGNORECASE)
# Skip image payloads: we never use them and they bloat the dict output.
_TEXT_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES
_HYPHEN_BREAK = re.compile(r"([^\W\d_])[-­]\n([^\W\d_])")
_INLINE_SPACE = re.compile(r"[^\S\n]+")


def parse_pdf(path: Path) -> tuple[list[RawBlock], bool, list[str]]:
    """Extract text blocks from every page in reading order."""
    doc = _open(path)
    try:
        blocks: list[RawBlock] = []
        for page in doc:
            blocks.extend(_order_page(_page_blocks(page), page.rect.width))
        page_count = doc.page_count
    finally:
        doc.close()

    warnings: list[str] = []
    total_chars = sum(len(b.text) for b in blocks)
    has_text_layer = total_chars / page_count >= MIN_CHARS_PER_PAGE
    if not has_text_layer:
        warnings.append(NO_TEXT_LAYER_WARNING)
    return blocks, has_text_layer, warnings


def _open(path: Path) -> pymupdf.Document:
    """Open the PDF and enforce the safety limits before any text extraction."""
    try:
        doc = pymupdf.open(path, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several unrelated types for broken files
        raise ParseError("corrupt", f"Cannot open PDF: {exc}") from exc

    # Only a user password blocks reading; owner-password (permissions-only)
    # PDFs are common for exported CVs and parse fine.
    if doc.needs_pass:
        doc.close()
        raise ParseError("encrypted", "PDF requires a password")
    if doc.page_count == 0:
        doc.close()
        raise ParseError("corrupt", "PDF has no pages")
    if doc.page_count > MAX_PAGES:
        count = doc.page_count
        doc.close()
        raise ParseError("too_many_pages", f"PDF has {count} pages; limit is {MAX_PAGES}")
    return doc


def _page_blocks(page: pymupdf.Page) -> list[RawBlock]:
    """Turn MuPDF text blocks into RawBlocks, dropping ones that clean to nothing."""
    blocks: list[RawBlock] = []
    for block in page.get_text("dict", flags=_TEXT_FLAGS)["blocks"]:
        if block.get("type") != 0:
            continue
        raw = _to_raw_block(block, page.number)
        if raw is not None:
            blocks.append(raw)
    return blocks


def _to_raw_block(block: dict[str, Any], page_number: int) -> RawBlock | None:
    lines: list[str] = []
    size_chars: Counter[float] = Counter()
    bold_chars = 0
    total_chars = 0
    for line in block["lines"]:
        lines.append("".join(span["text"] for span in line["spans"]))
        for span in line["spans"]:
            n = len(span["text"].strip())
            if n == 0:
                continue
            total_chars += n
            # Round to 0.5pt so tiny rendering differences don't split the vote.
            size_chars[round(span["size"] * 2) / 2] += n
            if _is_bold(span):
                bold_chars += n

    text = clean_text("\n".join(lines))
    if not text:
        return None
    x0, y0, x1, y1 = block["bbox"]
    return RawBlock(
        text=text,
        page=page_number,
        bbox=(x0, y0, x1, y1),
        font_size=size_chars.most_common(1)[0][0] if size_chars else None,
        # Majority vote: one bold word inside a body paragraph must not make it a header.
        is_bold=bold_chars * 2 > total_chars,
    )


def _is_bold(span: dict[str, Any]) -> bool:
    return bool(span["flags"] & _BOLD_FLAG) or bool(_BOLD_FONT_NAME.search(span["font"]))


def clean_text(text: str) -> str:
    """Normalise extracted text while keeping one line per visual line.

    Line breaks are kept (single ``\\n``) because JM-10 uses bullet lines; blank
    lines are dropped so a block can never contain the block separator.
    """
    text = unicodedata.normalize("NFKC", text)  # also expands ligatures like "ﬁ"
    text = _HYPHEN_BREAK.sub(_join_hyphenated, text)
    text = text.replace("­", "")
    lines = (_INLINE_SPACE.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def _join_hyphenated(match: re.Match[str]) -> str:
    # Only join when the next word continues in lowercase ("develop-\nment");
    # "Front-\nEnd" or "Senior-\nLevel" keep their hyphen.
    before, after = match.group(1), match.group(2)
    if after.islower():
        return before + after
    return match.group(0)


def _order_page(blocks: list[RawBlock], page_width: float) -> list[RawBlock]:
    return sorted(blocks, key=_top_left)


def _top_left(block: RawBlock) -> tuple[float, float]:
    assert block.bbox is not None
    return (round(block.bbox[1], 1), block.bbox[0])
