"""PDF parsing with PyMuPDF: text blocks with font metadata, cleaned text."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf

from jobpdf.extraction.cleaning import clean_text
from jobpdf.extraction.models import ParseError, RawBlock

MAX_PAGES = 15
# Below this many characters per page the PDF is almost certainly a scan.
MIN_CHARS_PER_PAGE = 50
NO_TEXT_LAYER_WARNING = "No text layer detected (likely a scanned PDF); OCR is out of scope."

# Reading-order tuning knobs (fractions are of the page width).
# A block wider than this is "full width" and may cross a column gutter.
FULL_WIDTH_RATIO = 0.6
# Gutters are only searched for in this horizontal band of the page.
GUTTER_SEARCH = (0.25, 0.75)
# Narrower gaps are treated as word/tab spacing, not a column break.
MIN_GUTTER_PT = 4.0
GUTTER_SCAN_STEP_PT = 1.0

_BOLD_FLAG = 16
# Some embedded fonts never set the bold flag, so fall back to the font name.
_BOLD_FONT_NAME = re.compile(r"bold|black|heavy", re.IGNORECASE)
# Skip image payloads: we never use them and they bloat the dict output.
_TEXT_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES


def parse_pdf(path: Path) -> tuple[list[RawBlock], bool, list[str]]:
    """Extract text blocks from every page in reading order."""
    doc = _open(path)
    try:
        blocks: list[RawBlock] = []
        for page in doc:
            blocks.extend(order_blocks(_page_blocks(page), page.rect.width))
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


def order_blocks(blocks: list[RawBlock], page_width: float) -> list[RawBlock]:
    """Order one page's blocks the way a person reads a CV.

    MuPDF's ``sort=True`` sorts by position, which interleaves the lines of a
    two-column or sidebar layout. Instead we look for a vertical gutter; if one
    exists, full-width blocks crossing it split the page into horizontal bands,
    and each band is read left column first, then right column.
    """
    gutter = find_gutter(blocks, page_width)
    if gutter is None:
        return sorted(blocks, key=_top_left)

    left = [b for b in blocks if _x1(b) <= gutter]
    right = [b for b in blocks if _x0(b) >= gutter]
    spanning = sorted((b for b in blocks if _x0(b) < gutter < _x1(b)), key=_top_left)

    ordered: list[RawBlock] = []
    band_top = float("-inf")
    for separator in [*spanning, None]:
        band_bottom = _center_y(separator) if separator else float("inf")
        for column in (left, right):
            in_band = (b for b in column if band_top <= _center_y(b) < band_bottom)
            ordered.extend(sorted(in_band, key=_top_left))
        if separator is not None:
            ordered.append(separator)
            band_top = band_bottom
    return ordered


def find_gutter(blocks: list[RawBlock], page_width: float) -> float | None:
    """Return the x of the widest column gutter in the search range, or None.

    Scans candidate x positions and keeps the widest contiguous run of valid
    ones, so the gutter lands in the middle of the gap rather than at its edge.
    """
    lo = page_width * GUTTER_SEARCH[0]
    hi = page_width * GUTTER_SEARCH[1]
    steps = int((hi - lo) / GUTTER_SCAN_STEP_PT) + 1
    best: tuple[float, float] | None = None  # (width, midpoint)
    run_start: float | None = None
    for i in range(steps + 1):
        x = lo + i * GUTTER_SCAN_STEP_PT
        valid = i < steps and _is_gutter(blocks, x, page_width)
        if valid and run_start is None:
            run_start = x
        elif not valid and run_start is not None:
            run_end = x - GUTTER_SCAN_STEP_PT
            width = run_end - run_start
            if width >= MIN_GUTTER_PT and (best is None or width > best[0]):
                best = (width, (run_start + run_end) / 2)
            run_start = None
    return best[1] if best else None


def _is_gutter(blocks: list[RawBlock], x: float, page_width: float) -> bool:
    """A gutter needs content on both sides, side by side vertically.

    Full-width blocks may cross it (they become band separators). A narrow block
    may cross it only if nothing in the columns sits at the same height — e.g. a
    short centred name above the columns — otherwise x runs through a column.
    """
    left = [b for b in blocks if _x1(b) <= x]
    right = [b for b in blocks if _x0(b) >= x]
    if not left or not right or not _y_ranges_overlap(left, right):
        return False
    side_blocks = left + right
    for block in blocks:
        crosses = _x0(block) < x < _x1(block)
        if not crosses or _width(block) > FULL_WIDTH_RATIO * page_width:
            continue
        if any(_overlaps_vertically(block, other) for other in side_blocks):
            return False
    return True


def _y_ranges_overlap(a: list[RawBlock], b: list[RawBlock]) -> bool:
    # Rules out e.g. a lone right-aligned date at the top of a single-column CV.
    a_top, a_bottom = min(_y0(x) for x in a), max(_y1(x) for x in a)
    b_top, b_bottom = min(_y0(x) for x in b), max(_y1(x) for x in b)
    return a_top < b_bottom and b_top < a_bottom


def _overlaps_vertically(a: RawBlock, b: RawBlock) -> bool:
    return _y0(a) < _y1(b) and _y0(b) < _y1(a)


def _bbox(block: RawBlock) -> tuple[float, float, float, float]:
    assert block.bbox is not None, "PDF blocks always carry a bbox"
    return block.bbox


def _x0(block: RawBlock) -> float:
    return _bbox(block)[0]


def _y0(block: RawBlock) -> float:
    return _bbox(block)[1]


def _x1(block: RawBlock) -> float:
    return _bbox(block)[2]


def _y1(block: RawBlock) -> float:
    return _bbox(block)[3]


def _width(block: RawBlock) -> float:
    return _x1(block) - _x0(block)


def _center_y(block: RawBlock) -> float:
    return (_y0(block) + _y1(block)) / 2


def _top_left(block: RawBlock) -> tuple[float, float]:
    # Round y so blocks on the same visual line but with different ascenders
    # are still ordered left to right.
    return (round(_y0(block), 1), _x0(block))
