from pathlib import Path

import pytest

from jobpdf.extraction import ParseError, parse
from jobpdf.extraction.models import RawBlock
from jobpdf.extraction.parser import MAX_FILE_BYTES, _assemble


def _raw(text: str) -> RawBlock:
    return RawBlock(text=text, page=None, bbox=None, font_size=None, is_bold=False)


def test_assemble_offsets_point_into_full_text() -> None:
    doc = _assemble("docx", [_raw("Olena Testenko"), _raw("Experience\nAcme")], True, [])

    assert doc.full_text == "Olena Testenko\n\nExperience\nAcme"
    for block in doc.blocks:
        assert doc.full_text[block.start : block.end] == block.text


def test_assemble_rejects_separator_inside_block() -> None:
    with pytest.raises(ValueError):
        _assemble("docx", [_raw("a\n\nb")], True, [])


def test_assemble_empty_document() -> None:
    doc = _assemble("pdf", [], False, ["w"])

    assert doc.full_text == ""
    assert doc.blocks == []
    assert doc.warnings == ["w"]


def test_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "cv.txt"
    path.write_text("Olena Testenko")

    with pytest.raises(ParseError) as exc:
        parse(path)
    assert exc.value.reason == "unsupported_type"


def test_too_large_is_rejected_before_opening(tmp_path: Path) -> None:
    path = tmp_path / "huge.pdf"
    with path.open("wb") as f:
        f.truncate(MAX_FILE_BYTES + 1)  # sparse: no real 20 MB written

    with pytest.raises(ParseError) as exc:
        parse(path)
    assert exc.value.reason == "too_large"
