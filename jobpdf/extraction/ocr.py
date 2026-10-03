"""OCR fallback for pages without usable text (JM-9).

OCR runs through PyMuPDF, whose MuPDF build embeds the Tesseract engine; the
language data comes from a system Tesseract install. By team decision that
install exists only in Docker images and CI, so on developer machines
``ocr_available()`` returning False is the normal case, not an error.
"""

from __future__ import annotations

import unicodedata
from dataclasses import replace
from functools import cache
from pathlib import Path

import pymupdf

# parse_pdf imports this module lazily inside its OCR hook, so this import is not a cycle.
from jobpdf.extraction.models import RawBlock
from jobpdf.extraction.parse_pdf import _TEXT_FLAGS, MIN_CHARS_PER_PAGE

# A page whose text is mostly unmappable glyphs (broken font encodings) is not usable.
MAX_GARBAGE_RATIO = 0.2
OCR_LANGUAGES = "eng+ukr"
OCR_DPI = 300
# OCR costs seconds per page; a CV rarely has more scanned pages than this.
OCR_MAX_PAGES = 5

_REQUIRED_LANGUAGES = tuple(OCR_LANGUAGES.split("+"))
_SMOKE_SIZE = 50  # px; a blank image OCR'd once to prove engine + data load


def garbage_ratio(text: str) -> float:
    """Share of non-whitespace characters that are U+FFFD, private-use or control chars.

    Those are what PyMuPDF yields for fonts without a Unicode mapping: the text
    layer exists but reads as noise, so the page needs OCR like a scan.
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(_is_garbage(c) for c in chars) / len(chars)


def _is_garbage(char: str) -> bool:
    return (
        char == "�"
        or 0xE000 <= ord(char) <= 0xF8FF
        or unicodedata.category(char) == "Cc"
    )


def non_space_chars(text: str) -> int:
    return sum(not c.isspace() for c in text)


def has_clean_text(text: str) -> bool:
    """Any real text at all: the bar for "unreadable", lower than is_usable_text's."""
    return non_space_chars(text) > 0 and garbage_ratio(text) <= MAX_GARBAGE_RATIO


def is_usable_text(text: str) -> bool:
    return (
        non_space_chars(text) >= MIN_CHARS_PER_PAGE
        and garbage_ratio(text) <= MAX_GARBAGE_RATIO
    )


def ocr_available() -> tuple[bool, str | None]:
    """(True, None) if OCR works here, else (False, reason). Checked once per process."""
    tessdata, reason = _probe()
    return tessdata is not None, reason


@cache
def _probe() -> tuple[str | None, str | None]:
    """Resolve the tessdata folder once.

    pymupdf.get_tessdata() shells out to `tesseract --list-langs` on every call,
    so the path is cached here and passed explicitly to each OCR call.
    """
    try:
        tessdata = pymupdf.get_tessdata()
    except Exception:  # RuntimeError when nothing is found; never fatal for parsing
        return None, "tesseract not installed"
    for language in _REQUIRED_LANGUAGES:
        if not (Path(tessdata) / f"{language}.traineddata").is_file():
            return None, f"{language} language data missing"
    try:
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, _SMOKE_SIZE, _SMOKE_SIZE), False)
        pixmap.clear_with(255)
        pixmap.pdfocr_tobytes(language=OCR_LANGUAGES, tessdata=tessdata)
    except Exception as exc:  # engine or data present but unusable
        return None, f"OCR engine failed: {exc}"
    return tessdata, None


def ocr_textpage(page: pymupdf.Page) -> pymupdf.TextPage:
    """A text layer for the whole page from OCR.

    full=True ignores the page's own (missing or broken) text layer; the result
    is in the original page's coordinates, so reading order works unchanged.
    """
    tessdata, reason = _probe()
    if tessdata is None:
        raise RuntimeError(f"OCR unavailable: {reason}")
    return page.get_textpage_ocr(
        flags=_TEXT_FLAGS, language=OCR_LANGUAGES, dpi=OCR_DPI, full=True, tessdata=tessdata
    )


def strip_style(blocks: list[RawBlock]) -> list[RawBlock]:
    """OCR can't tell bold or font size reliably; unset them so JM-10 uses keywords only."""
    return [replace(block, is_bold=False, font_size=None) for block in blocks]
