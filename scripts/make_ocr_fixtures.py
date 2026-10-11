"""Generate the synthetic scanned-CV fixtures for the OCR tests (JM-9).

Needs only PyMuPDF (no Tesseract): each page is typeset, rendered to a
grayscale image and placed as the only content of a new page, like a scan.
Cyrillic text needs a TTF with Cyrillic glyphs, taken from the system:

    uv run python scripts/make_ocr_fixtures.py --font C:/Windows/Fonts/arial.ttf
    ... --font /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf

The generated PDFs are committed, so tests, Docker and CI never need the font.
All names, companies and contact details are invented.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf

OUT_DIR = Path("tests/extraction/fixtures/ocr")
SCAN_DPI = 200
MAX_BYTES = 300 * 1024
_METADATA = {
    "title": "Synthetic scanned CV fixture",
    "author": "JobPDF test suite",
    "creator": "scripts/make_ocr_fixtures.py",
    "producer": "JobPDF",
    "creationDate": "D:20260101000000Z",
    "modDate": "D:20260101000000Z",
}

EN_LINES = [
    (22, "Taras Vyhadanyi"),
    (11, "taras.vyhadanyi@example.com  |  Lviv, Ukraine"),
    (16, "EXPERIENCE"),
    (12, "Backend Developer, Example Systems LLC, 2020 - 2024"),
    (11, "Built internal reporting services and maintained the release pipeline."),
    (16, "EDUCATION"),
    (12, "BSc Computer Science, Example National University, 2016 - 2020"),
    (16, "SKILLS"),
    (12, "Python, SQL, Docker"),
]
UK_LINES = [
    (22, "Тарас Вигаданий"),
    (11, "taras.vyhadanyi@example.com  |  Львів"),
    (16, "Досвід роботи"),
    (12, "Розробник, ТОВ Приклад Системи, 2020 - 2024"),
    (11, "Підтримка внутрішніх сервісів звітності."),
    (16, "Освіта"),
    (12, "Бакалавр комп'ютерних наук, Вигаданий університет, 2016 - 2020"),
]
NATIVE_LINES = [
    (16, "SUMMARY"),
    (11, "Backend developer with four years of experience building internal services"),
    (11, "for reporting and data pipelines at a fictional software company."),
]


def typeset(lines: list[tuple[int, str]], font: Path | None) -> pymupdf.Document:
    """A normal text page. ``font`` (a TTF) is needed for Cyrillic; None uses the
    built-in Helvetica, which keeps native-text pages small (nothing embedded)."""
    doc = pymupdf.open()
    page = doc.new_page()  # A4-ish, 595 x 842 pt
    y = 70.0
    for size, text in lines:
        if font is None:
            page.insert_text((60, y), text, fontsize=size, fontname="helv")
        else:
            page.insert_text((60, y), text, fontsize=size, fontname="cvfont", fontfile=str(font))
        y += size * 2.0
    return doc


def scan(source: pymupdf.Document, target: pymupdf.Document) -> None:
    """Append source's first page to target as an image-only page."""
    src_page = source[0]
    pixmap = src_page.get_pixmap(dpi=SCAN_DPI, colorspace=pymupdf.csGRAY)
    page = target.new_page(width=src_page.rect.width, height=src_page.rect.height)
    page.insert_image(page.rect, stream=pixmap.tobytes("png"))


def save(doc: pymupdf.Document, name: str) -> None:
    doc.set_metadata(_METADATA)
    path = OUT_DIR / name
    doc.save(path, garbage=4, deflate=True, no_new_id=True)
    doc.close()
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise SystemExit(f"{path} is {size} bytes; keep fixtures under {MAX_BYTES}")
    print(f"{path}: {size // 1024} KB")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic scanned-CV fixtures.")
    parser.add_argument("--font", type=Path, required=True, help="TTF with Cyrillic glyphs")
    args = parser.parse_args()
    if not args.font.is_file():
        raise SystemExit(f"Font not found: {args.font}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scanned_en = pymupdf.open()
    scan(typeset(EN_LINES, args.font), scanned_en)
    save(scanned_en, "scanned_en.pdf")

    scanned_uk = pymupdf.open()
    scan(typeset(UK_LINES, args.font), scanned_uk)
    save(scanned_uk, "scanned_uk.pdf")

    mixed = typeset(NATIVE_LINES, None)  # page 0: real text layer, English only
    scan(typeset(EN_LINES, args.font), mixed)  # page 1: image only
    save(mixed, "mixed.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
