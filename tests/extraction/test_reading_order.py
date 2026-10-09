from pathlib import Path

import pytest

from jobpdf.extraction import parse
from jobpdf.extraction.models import RawBlock
from jobpdf.extraction.parse_pdf import find_gutter, order_blocks

FIXTURES = Path(__file__).parent / "fixtures"
PAGE_WIDTH = 595.0

# Synthetic fixture content: the sidebar/left-column skills and the text that
# lives only in the main column (see make_fixtures.py).
COLUMN_CASES = {
    "two_column.pdf": (
        ["Python", "SQL", "Apache Airflow", "Docker", "PostgreSQL"],
        ["Experience", "Data Engineer", "Acme Widgets", "Education"],
    ),
    "left_sidebar.pdf": (
        ["Python", "SQL", "Docker", "Kubernetes", "Terraform"],
        ["Olena Testenko", "Profile", "Platform engineer", "Initech Demo"],
    ),
}


@pytest.mark.parametrize("name", COLUMN_CASES)
def test_sidebar_skills_are_contiguous(name: str) -> None:
    skills, main_column = COLUMN_CASES[name]
    text = parse(FIXTURES / name).full_text

    positions = [text.index(skill) for skill in skills]
    skills_span = text[min(positions) : max(positions) + len(skills[-1])]
    for marker in main_column:
        assert marker not in skills_span, f"{marker!r} interleaved with sidebar skills"


def test_two_column_banner_comes_first_and_columns_stay_whole() -> None:
    doc = parse(FIXTURES / "two_column.pdf")
    text = doc.full_text

    assert doc.blocks[0].text == "Olena Testenko"
    assert text.index("example.com/olena-demo") < text.index("Skills")
    # Whole left column before the right one.
    assert text.index("English (C1)") < text.index("Experience")
    assert text.index("Experience") < text.index("Junior Analyst") < text.index("Education")


def test_sidebar_contact_and_skills_stay_together() -> None:
    text = parse(FIXTURES / "left_sidebar.pdf").full_text

    assert text.index("Contact") < text.index("Kyiv, Ukraine") < text.index("Terraform")
    assert text.index("Terraform") < text.index("Olena Testenko")


def _block(text: str, x0: float, y0: float, x1: float, y1: float) -> RawBlock:
    return RawBlock(text=text, page=0, bbox=(x0, y0, x1, y1), font_size=10.0, is_bold=False)


def test_single_column_has_no_gutter_and_sorts_top_to_bottom() -> None:
    blocks = [
        _block("second", 50, 200, 500, 240),
        _block("first", 50, 100, 500, 140),
        _block("short", 50, 150, 120, 160),
    ]

    assert find_gutter(blocks, PAGE_WIDTH) is None
    assert [b.text for b in order_blocks(blocks, PAGE_WIDTH)] == ["first", "short", "second"]


def test_mid_page_full_width_block_splits_columns_into_bands() -> None:
    blocks = [
        _block("L1", 40, 100, 270, 150),
        _block("R1", 320, 100, 555, 150),
        _block("WIDE", 40, 170, 555, 190),
        _block("L2", 40, 210, 270, 260),
        _block("R2", 320, 210, 555, 260),
    ]

    ordered = [b.text for b in order_blocks(blocks, PAGE_WIDTH)]
    assert ordered == ["L1", "R1", "WIDE", "L2", "R2"]


def test_narrow_block_beside_columns_blocks_the_gutter() -> None:
    # A block sitting across the gap at the same height as column content means
    # the gap is not a real column gutter.
    blocks = [
        _block("left", 40, 100, 200, 140),
        _block("right", 400, 100, 555, 140),
        _block("middle", 180, 110, 420, 130),
    ]

    assert find_gutter(blocks, PAGE_WIDTH) is None


def test_empty_page() -> None:
    assert order_blocks([], PAGE_WIDTH) == []
