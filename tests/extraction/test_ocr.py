import time
import weakref
from pathlib import Path

import pymupdf
import pytest
from rapidfuzz import fuzz

from jobpdf.extraction import ParseError, parse
from jobpdf.extraction import ocr as ocr_module
from jobpdf.extraction.ocr import (
    MAX_GARBAGE_RATIO,
    OCR_MAX_PAGES,
    garbage_ratio,
    is_usable_text,
    ocr_available,
)
from jobpdf.extraction.parse_pdf import _page_blocks, order_blocks
from jobpdf.extraction.sections import split_sections

OCR_FIXTURES = Path(__file__).parent / "fixtures" / "ocr"
OCR_OK, OCR_REASON = ocr_available()
needs_ocr = pytest.mark.skipif(not OCR_OK, reason="tesseract not installed (runs in Docker/CI)")
FAKE_OCR_TEXT = [
    "EXPERIENCE",
    "Backend Developer, Example Systems LLC, 2020 - 2024",
    "Python, SQL, Docker",
]


def assert_offsets(doc) -> None:
    for block in doc.blocks:
        assert doc.full_text[block.start : block.end] == block.text


def image_only_pdf(path: Path, pages: int, *, blank: bool = False) -> Path:
    """Image-only pages built in the test, so nothing is written into the repo."""
    doc = pymupdf.open()
    for i in range(pages):
        source = pymupdf.open()
        source_page = source.new_page()
        if not blank:
            source_page.insert_text((60, 80), f"Scanned page {i} of a fake CV", fontsize=14)
        pixmap = source_page.get_pixmap(dpi=72, colorspace=pymupdf.csGRAY)
        page = doc.new_page(width=source_page.rect.width, height=source_page.rect.height)
        page.insert_image(page.rect, stream=pixmap.tobytes("png"))
        source.close()
    doc.save(path)
    doc.close()
    return path


# --- pure tests: no Tesseract needed ---------------------------------------------------


@pytest.mark.parametrize(
    ("text", "ratio"),
    [
        ("Backend developer with Python", 0.0),
        ("", 0.0),
        ("   \n\t ", 0.0),
        ("���ab", 0.6),
        ("abcd", 3 / 7),
        ("ab\x07c", 0.25),  # control character
    ],
)
def test_garbage_ratio(text: str, ratio: float) -> None:
    assert garbage_ratio(text) == pytest.approx(ratio)


def test_is_usable_text() -> None:
    normal = "Backend developer building reporting services in Python and SQL daily."
    assert is_usable_text(normal)
    assert not is_usable_text("")
    assert not is_usable_text("Short page")
    assert not is_usable_text("�" * 60)
    assert not is_usable_text("" * 30 + "a" * 30)
    noisy_but_ok = "a" * 80 + "�" * int(80 * MAX_GARBAGE_RATIO / (1 - MAX_GARBAGE_RATIO))
    assert is_usable_text(noisy_but_ok)


class FakeOcr:
    """Stands in for ocr_textpage: a real TextPage typeset from known text."""

    def __init__(self, lines: list[str] | None = None) -> None:
        self.lines = FAKE_OCR_TEXT if lines is None else lines
        self.calls: list[int] = []
        self._keep_alive: list[pymupdf.Document] = []

    def __call__(self, page: pymupdf.Page) -> pymupdf.TextPage:
        self.calls.append(page.number)
        source = pymupdf.open()
        self._keep_alive.append(source)
        source_page = source.new_page(width=page.rect.width, height=page.rect.height)
        for i, line in enumerate(self.lines):
            # Bold and large on purpose: strip_style must remove both.
            source_page.insert_text((60, 80 + 30 * i), line, fontsize=18, fontname="hebo")
        textpage = source_page.get_textpage(flags=pymupdf.TEXTFLAGS_DICT)
        textpage.parent = weakref.proxy(page)  # what PyMuPDF's own OCR does
        return textpage


@pytest.fixture
def fake_ocr(monkeypatch: pytest.MonkeyPatch) -> FakeOcr:
    fake = FakeOcr()
    monkeypatch.setattr(ocr_module, "ocr_available", lambda: (True, None))
    monkeypatch.setattr(ocr_module, "ocr_textpage", fake)
    return fake


def test_degrades_gracefully_without_tesseract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_module, "ocr_available", lambda: (False, "tesseract not installed"))

    doc = parse(OCR_FIXTURES / "scanned_en.pdf")

    assert doc.has_text_layer is False
    assert doc.ocr_pages == []
    assert doc.blocks == []  # JM-8 behaviour: nothing readable, but no crash
    assert any("tesseract not installed" in w for w in doc.warnings)
    assert sum("OCR unavailable" in w for w in doc.warnings) == 1


def test_ocr_blocks_flow_through_the_normal_pipeline(fake_ocr: FakeOcr) -> None:
    doc = parse(OCR_FIXTURES / "scanned_en.pdf")

    assert doc.ocr_pages == [0]
    assert fake_ocr.calls == [0]
    assert doc.has_text_layer is False  # about the original file
    assert [b.text for b in doc.blocks] == FAKE_OCR_TEXT
    assert all(b.is_bold is False and b.font_size is None for b in doc.blocks)
    assert all(b.page == 0 for b in doc.blocks)
    assert_offsets(doc)


def test_native_pages_are_not_ocrd(fake_ocr: FakeOcr) -> None:
    doc = parse(OCR_FIXTURES / "mixed.pdf")

    assert fake_ocr.calls == [1]
    assert doc.ocr_pages == [1]
    assert any("SUMMARY" in b.text and b.page == 0 for b in doc.blocks)


def test_ocr_page_limit(tmp_path: Path, fake_ocr: FakeOcr) -> None:
    path = image_only_pdf(tmp_path / "seven.pdf", OCR_MAX_PAGES + 2)

    doc = parse(path)

    assert len(fake_ocr.calls) == OCR_MAX_PAGES
    assert doc.ocr_pages == list(range(OCR_MAX_PAGES))
    limit = [w for w in doc.warnings if "OCR limit" in w]
    assert len(limit) == 1 and "[5, 6]" in limit[0]


def test_clean_short_native_page_keeps_its_text(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "short_last_page.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text(
        (60, 80), "Backend developer building reporting services in Python and SQL."
    )
    doc.new_page().insert_text((60, 80), "References available on request")  # < 50 chars
    doc.save(path)
    fake = FakeOcr(["References"])  # OCR returns less than the native text
    monkeypatch.setattr(ocr_module, "ocr_available", lambda: (True, None))
    monkeypatch.setattr(ocr_module, "ocr_textpage", fake)

    parsed = parse(path)

    assert fake.calls == [1]  # OCR was tried on the short page...
    assert parsed.ocr_pages == []  # ...but its exact native text was kept
    assert parsed.full_text.endswith("References available on request")


def test_ocr_lines_become_separate_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tight line spacing makes MuPDF group these into one block, like Tesseract
    # gluing "EXPERIENCE" onto the paragraph above it.
    lines = ["Lviv, Ukraine", "EXPERIENCE", "Backend Developer, Example Systems LLC"]
    keep_alive: list[pymupdf.Document] = []

    def tight(page: pymupdf.Page) -> pymupdf.TextPage:
        source = pymupdf.open()
        keep_alive.append(source)
        source_page = source.new_page(width=page.rect.width, height=page.rect.height)
        for i, line in enumerate(lines):
            source_page.insert_text((60, 80 + 14 * i), line, fontsize=12)
        textpage = source_page.get_textpage(flags=pymupdf.TEXTFLAGS_DICT)
        assert len([b for b in textpage.extractDICT()["blocks"] if b["type"] == 0]) == 1
        textpage.parent = weakref.proxy(page)
        return textpage

    monkeypatch.setattr(ocr_module, "ocr_available", lambda: (True, None))
    monkeypatch.setattr(ocr_module, "ocr_textpage", tight)

    doc = parse(OCR_FIXTURES / "scanned_en.pdf")

    assert [b.text for b in doc.blocks] == lines
    assert len({b.bbox for b in doc.blocks}) == 3  # each line keeps its own bbox
    sections, _ = split_sections(doc)
    assert any("experience" in s.types for s in sections)


def test_short_clean_single_page_is_not_unreadable(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "short.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((60, 80), "References available on request")  # < 50 chars
    doc.save(path)
    monkeypatch.setattr(ocr_module, "ocr_available", lambda: (True, None))
    monkeypatch.setattr(ocr_module, "ocr_textpage", FakeOcr(["References"]))

    parsed = parse(path)

    assert parsed.full_text == "References available on request"
    assert parsed.ocr_pages == []


def test_has_clean_text() -> None:
    assert ocr_module.has_clean_text("References")
    assert not ocr_module.has_clean_text("   ")
    assert not ocr_module.has_clean_text("���")


def test_nothing_readable_after_ocr_is_unreadable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ocr_module, "ocr_available", lambda: (True, None))
    monkeypatch.setattr(ocr_module, "ocr_textpage", FakeOcr([]))

    with pytest.raises(ParseError) as exc:
        parse(image_only_pdf(tmp_path / "blank.pdf", 1, blank=True))
    assert exc.value.reason == "unreadable"


def test_ocr_failure_on_a_page_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(page: pymupdf.Page) -> pymupdf.TextPage:
        raise RuntimeError("engine crashed")

    monkeypatch.setattr(ocr_module, "ocr_available", lambda: (True, None))
    monkeypatch.setattr(ocr_module, "ocr_textpage", broken)

    with pytest.raises(ParseError) as exc:  # the only page stayed unreadable
        parse(OCR_FIXTURES / "scanned_en.pdf")
    assert exc.value.reason == "unreadable"


# --- real OCR: skipped on laptops by design, run by scripts/test_ocr_docker.sh ----------


def fuzzy_in(needle: str, text: str) -> bool:
    return fuzz.partial_ratio(needle.lower(), text.lower()) >= 85


@needs_ocr
def test_real_ocr_english_scan() -> None:
    started = time.perf_counter()
    doc = parse(OCR_FIXTURES / "scanned_en.pdf")
    print(f"\nOCR scanned_en.pdf: {time.perf_counter() - started:.2f}s for 1 page")

    assert doc.ocr_pages == [0]
    for word in ["experience", "python", "sql", "docker"]:
        assert fuzzy_in(word, doc.full_text), word
    assert all(b.is_bold is False and b.font_size is None for b in doc.blocks)
    assert_offsets(doc)


@needs_ocr
def test_real_ocr_mixed_keeps_native_page() -> None:
    doc = parse(OCR_FIXTURES / "mixed.pdf")

    assert doc.ocr_pages == [1]
    source = pymupdf.open(OCR_FIXTURES / "mixed.pdf")
    native = order_blocks(_page_blocks(source[0]), source[0].rect.width)
    source.close()
    assert [b.text for b in doc.blocks if b.page == 0] == [b.text for b in native]
    assert fuzzy_in("python", doc.full_text)


@needs_ocr
def test_real_ocr_ukrainian_scan() -> None:
    doc = parse(OCR_FIXTURES / "scanned_uk.pdf")

    assert doc.ocr_pages == [0]
    assert fuzzy_in("досвід", doc.full_text)
    assert fuzzy_in("освіта", doc.full_text)


@needs_ocr
def test_real_ocr_end_to_end_sections() -> None:
    sections, _ = split_sections(parse(OCR_FIXTURES / "scanned_en.pdf"))

    experience = [s for s in sections if "experience" in s.types]
    assert experience
    assert experience[0].label_source in {"keyword", "fuzzy"}


@needs_ocr
def test_real_ocr_blank_page_is_unreadable(tmp_path: Path) -> None:
    with pytest.raises(ParseError) as exc:
        parse(image_only_pdf(tmp_path / "blank.pdf", 1, blank=True))
    assert exc.value.reason == "unreadable"


# --- image uploads (.png / .jpg / .jpeg) -----------------------------------------------


def scan_image(path: Path, dpi: int, tag_dpi: int | None = None) -> Path:
    """A PNG/JPEG 'photo' of the English scan fixture, written to tmp_path."""
    source = pymupdf.open(OCR_FIXTURES / "scanned_en.pdf")
    pixmap = source[0].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    source.close()
    if tag_dpi is not None:
        pixmap.set_dpi(tag_dpi, tag_dpi)
    pixmap.save(path)
    return path


def test_image_page_matches_ocr_resolution(tmp_path: Path) -> None:
    from jobpdf.extraction.ocr import OCR_DPI
    from jobpdf.extraction.parser import image_to_pdf

    image = scan_image(tmp_path / "photo.png", dpi=150, tag_dpi=72)  # misleading 72-dpi tag
    pixmap = pymupdf.Pixmap(str(image))

    page = pymupdf.open("pdf", image_to_pdf(image))[0]

    assert page.rect.width == pytest.approx(pixmap.width * 72 / OCR_DPI)
    assert page.rect.height == pytest.approx(pixmap.height * 72 / OCR_DPI)


@pytest.mark.parametrize("name", ["cv.png", "cv.jpg", "cv.JPEG"])
def test_image_upload_degrades_like_a_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setattr(ocr_module, "ocr_available", lambda: (False, "tesseract not installed"))

    doc = parse(scan_image(tmp_path / name, dpi=100))

    assert doc.source_type == "pdf"
    assert doc.has_text_layer is False
    assert doc.blocks == []
    assert any("tesseract not installed" in w for w in doc.warnings)


def test_image_upload_uses_ocr(tmp_path: Path, fake_ocr: FakeOcr) -> None:
    # 300 dpi -> an A4-sized page, wide enough for the fake's typeset lines.
    doc = parse(scan_image(tmp_path / "cv.png", dpi=300))

    assert doc.ocr_pages == [0]
    assert [b.text for b in doc.blocks] == FAKE_OCR_TEXT


def test_broken_image_is_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "cv.png"
    path.write_bytes(b"not an image at all")

    with pytest.raises(ParseError) as exc:
        parse(path)
    assert exc.value.reason == "corrupt"


@needs_ocr
def test_real_ocr_image_upload(tmp_path: Path) -> None:
    doc = parse(scan_image(tmp_path / "cv.jpg", dpi=200))

    assert doc.ocr_pages == [0]
    for word in ["experience", "python", "docker"]:
        assert fuzzy_in(word, doc.full_text), word
