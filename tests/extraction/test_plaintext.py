import pytest

from jobpdf.extraction.plaintext import DEFAULT_FONT_SIZE, from_plain_text
from jobpdf.extraction.sections import split_sections

TEXT = (
    "Jane Placeholder\nData Analyst - Example Corp\n\n"
    "EXPERIENCE\nData Analyst\nExample Corp\n\n\n"
    "SKILLS\nPython, SQL\n\n\n\n"
    "EDUCATION\nB.Sc. Statistics, Placeholder University, 2019"
)


@pytest.mark.parametrize("text", [TEXT, "", "\n", "one line", "\n\nlead and trail\n\n\n"])
def test_full_text_is_the_input_exactly(text: str) -> None:
    assert from_plain_text(text).full_text == text


def test_one_block_per_non_blank_line_with_exact_offsets() -> None:
    doc = from_plain_text(TEXT)

    non_blank = [line for line in TEXT.split("\n") if line.strip()]
    assert [b.text for b in doc.blocks] == non_blank
    for block in doc.blocks:
        assert doc.full_text[block.start : block.end] == block.text
        assert (block.font_size, block.is_bold, block.page, block.bbox) == (
            DEFAULT_FONT_SIZE, False, None, None,
        )
    assert doc.has_text_layer is True


def test_whitespace_only_lines_are_not_blocks() -> None:
    doc = from_plain_text("a\n   \nb")

    assert [b.text for b in doc.blocks] == ["a", "b"]
    assert from_plain_text("  \n ").has_text_layer is False


def test_sections_slice_the_original_text() -> None:
    doc = from_plain_text(TEXT)

    sections, _ = split_sections(doc)

    types = [s.types for s in sections]
    assert ["skills"] in types and ["education"] in types
    for section in sections:
        content = doc.full_text[section.content_start : section.end]
        assert all(b.text in content for b in section.blocks)
