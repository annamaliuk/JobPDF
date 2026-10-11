import re

import pytest

from jobpdf.extraction.models import ParsedDocument, RawBlock, Section
from jobpdf.extraction.parser import _assemble
from jobpdf.extraction.prompt import MASK_CHAR, build_user_message, mask_pii


@pytest.mark.parametrize(
    "pii",
    [
        "olena.testenko@example.com",
        "first.last+cv@mail.example.org",
        "+380 67 123 4567",
        "+44 (20) 7946 0958",
        "(067) 123-45-67",
        "+1-202-555-0143",
        "linkedin.com/in/olena-demo",
        "https://www.linkedin.com/in/olena-demo/",
        "github.com/olena-demo",
        "https://example.com/portfolio",
    ],
)
def test_pii_is_masked_with_equal_length(pii: str) -> None:
    text = f"Contact: {pii} | Kyiv"

    masked, count = mask_pii(text)

    assert len(masked) == len(text)
    assert count == 1
    assert pii not in masked
    start = text.index(pii)
    assert masked[start : start + len(pii)] == MASK_CHAR * len(pii)
    assert masked.startswith("Contact: ") and masked.endswith(" | Kyiv")


@pytest.mark.parametrize(
    "safe",
    [
        "2019 - 2021",
        "01.2019 – 03.2021",
        "01.2019 - 03.2021",
        "2015–2018, 2019–2023",
        "2019 - 2021 2021 - 2024",
        "Acme Widgets LLC, 2021-2024",
        "Python 3.11",
        "Python 3.11.4, Java 17, C++20",
        "2020",
        "managed a team of 12 engineers, 250 000 users",
    ],
)
def test_dates_years_and_versions_are_not_masked(safe: str) -> None:
    masked, count = mask_pii(safe)

    assert masked == safe
    assert count == 0


def test_mask_keeps_length_and_other_text() -> None:
    text = "Olena\nolena@example.com\n+380 67 123 4567\nData Engineer, 2021 - 2024"

    masked, count = mask_pii(text)

    assert count == 2
    assert len(masked) == len(text)
    assert masked.endswith("Data Engineer, 2021 - 2024")
    assert [i for i, c in enumerate(text) if c == "\n"] == [
        i for i, c in enumerate(masked) if c == "\n"
    ]


def raw(text: str) -> RawBlock:
    return RawBlock(text=text, page=0, bbox=None, font_size=10.0, is_bold=False)


def make_doc(*texts: str) -> ParsedDocument:
    return _assemble("pdf", [raw(t) for t in texts], True, [])


def section(doc: ParsedDocument, index: int, types, heading=None, conf="high") -> Section:
    block = doc.blocks[index]
    return Section(
        types=types, heading=heading, blocks=[block], content_start=block.start,
        end=block.end, confidence=conf, label_source="keyword",
    )


def test_contact_only_sections_are_excluded_but_tagged_ones_kept() -> None:
    doc = make_doc("olena@example.com", "Python, SQL", "teamwork, olena@example.com")
    sections = [
        section(doc, 0, ["contact"]),
        section(doc, 1, ["skills"], heading="Skills"),
        section(doc, 2, ["skills", "contact"], heading="Soft skills"),
    ]

    message = build_user_message(doc, sections)

    assert message.count("<section ") == 2
    assert 'types="skills,contact"' in message
    assert "olena@example.com" not in message
    assert "teamwork, " + MASK_CHAR * len("olena@example.com") in message


def test_markers_escape_attributes_and_slice_masked_text() -> None:
    doc = make_doc("Experience at <Acme> & Co", "BSc, 2016 - 2020")
    sections = [
        section(doc, 0, ["experience", "education"], heading='Work "&" <Study>', conf="low"),
        section(doc, 1, ["education"]),
    ]

    message = build_user_message(doc, sections)

    assert message.startswith(
        '<section types="experience,education" confidence="low" '
        'heading="Work &quot;&amp;&quot; &lt;Study&gt;">\nExperience at <Acme> & Co\n</section>'
    )
    assert '<section types="education" confidence="high">\nBSc, 2016 - 2020\n</section>' in message
    masked, _ = mask_pii(doc.full_text)
    for s in sections:
        assert f">\n{masked[s.content_start:s.end]}\n</section>" in message


def test_sections_are_emitted_in_document_order() -> None:
    doc = make_doc("First block text", "Second block text", "Third block text")
    sections = [
        section(doc, 2, ["skills"]), section(doc, 0, ["summary"]), section(doc, 1, ["other"])
    ]

    message = build_user_message(doc, sections)

    assert [m for m in re.findall(r"(\w+) block text", message)] == ["First", "Second", "Third"]


def test_no_sections_gives_one_fallback_section_with_whole_masked_text() -> None:
    doc = make_doc("Olena", "olena@example.com", "Python developer")

    message = build_user_message(doc, [])

    masked, _ = mask_pii(doc.full_text)
    assert message == f'<section types="other" confidence="low">\n{masked}\n</section>'


def test_only_contact_sections_also_fall_back_to_whole_text() -> None:
    doc = make_doc("olena@example.com")

    message = build_user_message(doc, [section(doc, 0, ["contact"])])

    assert message.startswith('<section types="other" confidence="low">')
    assert "olena@example.com" not in message
