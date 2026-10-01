from pathlib import Path

import pymupdf
import pytest

from jobpdf.extraction import ParseError, parse
from jobpdf.extraction.parse_pdf import MAX_PAGES

FIXTURES = Path(__file__).parent / "fixtures"
TEXT_FIXTURES = ["single_column.pdf", "two_column.pdf", "left_sidebar.pdf"]


@pytest.mark.parametrize("name", [*TEXT_FIXTURES, "image_only.pdf"])
def test_offsets_index_into_full_text(name: str) -> None:
    doc = parse(FIXTURES / name)

    assert doc.source_type == "pdf"
    for block in doc.blocks:
        assert doc.full_text[block.start : block.end] == block.text


@pytest.mark.parametrize("name", TEXT_FIXTURES)
def test_text_fixtures_have_text_layer(name: str) -> None:
    doc = parse(FIXTURES / name)

    assert doc.has_text_layer
    assert doc.warnings == []
    assert all(block.page == 0 and block.bbox is not None for block in doc.blocks)


@pytest.mark.parametrize("name", ["single_column.pdf", "two_column.pdf"])
def test_experience_header_precedes_first_job_and_stands_out(name: str) -> None:
    doc = parse(FIXTURES / name)

    header = next(b for b in doc.blocks if b.text == "Experience")
    assert header.start < doc.full_text.index("Data Engineer")
    body = next(b for b in doc.blocks if b.text.startswith("- "))
    assert header.is_bold or (header.font_size or 0) > (body.font_size or 0)
    assert not body.is_bold


def test_hyphenated_line_break_is_joined() -> None:
    doc = parse(FIXTURES / "single_column.pdf")

    assert "development of nightly" in doc.full_text


def test_image_only_pdf_has_no_text_layer() -> None:
    doc = parse(FIXTURES / "image_only.pdf")

    assert doc.has_text_layer is False
    assert any("text layer" in w for w in doc.warnings)


def test_corrupt_pdf() -> None:
    with pytest.raises(ParseError) as exc:
        parse(FIXTURES / "corrupt.pdf")
    assert exc.value.reason == "corrupt"


def _write_pdf(path: Path, pages: int = 1, **save_kwargs: object) -> Path:
    doc = pymupdf.open()
    for i in range(pages):
        doc.new_page().insert_text((50, 72), f"Synthetic page {i} for Olena Testenko")
    doc.save(path, **save_kwargs)
    doc.close()
    return path


def test_too_many_pages(tmp_path: Path) -> None:
    path = _write_pdf(tmp_path / "long.pdf", pages=MAX_PAGES + 1)

    with pytest.raises(ParseError) as exc:
        parse(path)
    assert exc.value.reason == "too_many_pages"


def test_user_password_pdf_is_rejected(tmp_path: Path) -> None:
    path = _write_pdf(
        tmp_path / "locked.pdf",
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="user-secret",
        owner_pw="owner-secret",
    )

    with pytest.raises(ParseError) as exc:
        parse(path)
    assert exc.value.reason == "encrypted"


def test_owner_password_only_pdf_is_parsed(tmp_path: Path) -> None:
    path = _write_pdf(
        tmp_path / "restricted.pdf",
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        permissions=pymupdf.PDF_PERM_PRINT,
    )

    doc = parse(path)
    assert "Olena Testenko" in doc.full_text
