from pathlib import Path

import pytest

from jobpdf.extraction import ParsedDocument, parse
from jobpdf.extraction.models import RawBlock, Section
from jobpdf.extraction.parser import _assemble
from jobpdf.extraction.sections import (
    NO_HEADINGS_WARNING,
    STYLE_GUARD_MAX,
    STYLE_GUARD_WARNING,
    STYLE_SKIPPED_WARNING,
    is_contact_line,
    split_sections,
)

FIXTURES = Path(__file__).parent / "fixtures"
BODY = 10.0
HEADING = 14.0


def b(text: str, size: float | None = BODY, bold: bool = False) -> RawBlock:
    return RawBlock(text=text, page=0, bbox=None, font_size=size, is_bold=bold)


def h(text: str) -> RawBlock:
    """A level-1 heading: large and bold."""
    return b(text, HEADING, bold=True)


def make_doc(*blocks: RawBlock) -> ParsedDocument:
    # The real assembler, so offsets are computed exactly as in production.
    return _assemble("pdf", list(blocks), True, [])


def split(doc: ParsedDocument) -> tuple[list[Section], list[str]]:
    sections, warnings = split_sections(doc)
    assert_invariants(doc, sections)
    return sections, warnings


def assert_invariants(doc: ParsedDocument, sections: list[Section]) -> None:
    """Every character of every block is in exactly one place, in order.

    Walks full_text left to right: each heading, then each content block, must
    come next, with only whitespace or heading separators in between. A dropped,
    duplicated or reordered block breaks the walk.
    """
    text = doc.full_text
    cursor = 0
    for section in sections:
        if section.blocks:
            assert section.content_start == section.blocks[0].start
            assert section.end == section.blocks[-1].end
        else:
            assert section.content_start == section.end
        if section.heading is not None:
            at = text.index(section.heading, cursor)
            assert text[cursor:at].strip() == ""
            cursor = at + len(section.heading)
        for block in section.blocks:
            assert block.start >= cursor, f"{block.text!r} out of order or duplicated"
            assert text[cursor : block.start].strip(" \n:–—-") == ""
            assert text[block.start : block.end] == block.text
            cursor = block.end
    assert text[cursor:].strip() == ""


def summary(sections: list[Section]) -> list[tuple[list[str], str | None]]:
    return [(s.types, s.heading) for s in sections]


def standard_cv() -> ParsedDocument:
    return make_doc(
        b("Olena Testenko", 20, bold=True),
        b("olena.testenko@example.com | +380 00 000 0000"),
        h("SUMMARY"),
        b("Data engineer with five years of experience building pipelines."),
        h("EXPERIENCE"),
        b("Data Engineer", bold=True),
        b("Acme Widgets LLC, 2021-2024"),
        b("- Built nightly reporting jobs for the finance team."),
        h("EDUCATION"),
        b("BSc Computer Science, Example State University"),
        h("SKILLS"),
        b("Python, SQL, Docker"),
    )


def test_standard_cv() -> None:
    sections, warnings = split(standard_cv())

    assert summary(sections) == [
        (["contact"], None),
        (["summary"], "SUMMARY"),
        (["experience"], "EXPERIENCE"),
        (["education"], "EDUCATION"),
        (["skills"], "SKILLS"),
    ]
    assert all(s.confidence == "high" for s in sections)
    assert [s.label_source for s in sections] == ["pattern"] + ["keyword"] * 4
    assert warnings == []


def test_bold_job_title_in_another_style_is_not_a_heading() -> None:
    sections, _ = split(standard_cv())

    experience = sections[2]
    assert [blk.text for blk in experience.blocks][:2] == [
        "Data Engineer",
        "Acme Widgets LLC, 2021-2024",
    ]


def test_messy_headings() -> None:
    sections, _ = split(
        make_doc(
            h("W O R K E X P E R I E N C E"),
            b("Acme Widgets LLC, 2021-2024"),
            h("01. Education:"),
            b("BSc Computer Science"),
            h("Experiences"),
            b("Globex Testing Co, 2019-2021"),
        )
    )

    assert [s.types for s in sections] == [["experience"], ["education"], ["experience"]]
    assert [s.label_source for s in sections] == ["keyword", "keyword", "fuzzy"]


def test_compound_headings() -> None:
    sections, _ = split(
        make_doc(
            h("Education & Experience"),
            b("BSc Computer Science"),
            h("Skills / Languages"),
            b("Python, English (C1)"),
        )
    )

    assert [s.types for s in sections] == [["education", "experience"], ["skills", "languages"]]


def test_ukrainian_headings() -> None:
    sections, _ = split(
        make_doc(
            h("Досвід роботи"),
            b("ТОВ Приклад, 2021-2024"),
            h("Освіта та курси"),
            b("Бакалавр комп'ютерних наук"),
        )
    )

    assert [s.types for s in sections] == [["experience"], ["education", "certifications"]]


def test_unknown_heading_in_level1_style() -> None:
    sections, _ = split(
        make_doc(
            h("Experience"),
            b("Acme Widgets LLC, 2021-2024"),
            h("My Journey"),
            b("Moved from finance into data engineering."),
            h("Education"),
            b("BSc Computer Science"),
        )
    )

    journey = sections[1]
    assert journey.types == ["other"]
    assert journey.heading == "My Journey"
    assert (journey.label_source, journey.confidence) == ("style", "low")


def test_name_in_heading_style_before_first_keyword_heading_is_not_a_section() -> None:
    sections, _ = split(
        make_doc(
            h("Olena Testenko"),
            b("olena.testenko@example.com"),
            h("Experience"),
            b("Acme Widgets LLC"),
            h("Education"),
            b("BSc Computer Science"),
        )
    )

    assert summary(sections)[0] == (["contact"], None)


def test_smaller_sub_heading_stays_in_section() -> None:
    sections, _ = split(
        make_doc(
            h("SKILLS"),
            b("Python, SQL, Docker"),
            b("Languages", 11, bold=True),
            b("English (C1), Ukrainian (native)"),
            b("Languages: Python, Java"),
            h("EXPERIENCE"),
            b("Acme Widgets LLC"),
        )
    )

    assert [s.types for s in sections] == [["skills"], ["experience"]]
    assert [blk.text for blk in sections[0].blocks] == [
        "Python, SQL, Docker",
        "Languages",
        "English (C1), Ukrainian (native)",
        "Languages: Python, Java",
    ]


def test_style_guard_when_heading_style_is_not_distinctive() -> None:
    job_titles = [h(f"Project Lead {i}") for i in range(STYLE_GUARD_MAX + 3)]
    sections, warnings = split(
        make_doc(h("Experience"), *job_titles, h("Education"), b("BSc Computer Science"))
    )

    assert STYLE_GUARD_WARNING in warnings
    assert [s.types for s in sections] == [["experience"], ["education"]]
    assert len(sections[0].blocks) == len(job_titles)


def test_fewer_than_two_keyword_headings_skips_style_learning() -> None:
    sections, warnings = split(
        make_doc(
            h("Experience"),
            b("Acme Widgets LLC"),
            h("My Journey"),
            b("Moved into data engineering."),
        )
    )

    assert STYLE_SKIPPED_WARNING in warnings
    assert [s.types for s in sections] == [["experience"]]


def test_inline_heading_content_offset_is_exact() -> None:
    doc = make_doc(b("Olena Testenko"), b("Skills: Python, SQL"))
    sections, _ = split(doc)

    skills = sections[-1]
    assert skills.types == ["skills"]
    assert skills.heading == "Skills"
    assert doc.full_text[skills.content_start :].startswith("Python")
    assert skills.blocks[0].text == "Python, SQL"


def test_first_line_heading_in_merged_block() -> None:
    doc = make_doc(
        h("Summary"),
        b("Data engineer."),
        h("Education"),
        b("BSc Computer Science"),
        b("Experience\nAcme Widgets LLC, 2021-2024"),
    )
    sections, _ = split(doc)

    experience = sections[-1]
    assert experience.types == ["experience"]
    assert experience.heading == "Experience"
    assert experience.confidence == "low"
    assert doc.full_text[experience.content_start : experience.end] == (
        "Acme Widgets LLC, 2021-2024"
    )


def test_top_section_with_email_is_contact() -> None:
    sections, _ = split(
        make_doc(b("Olena Testenko"), b("olena@example.com"), h("Skills"), b("Python"))
    )

    assert (sections[0].types, sections[0].label_source) == (["contact"], "pattern")


def test_top_section_without_contacts_is_summary() -> None:
    sections, _ = split(
        make_doc(b("Data engineer, Kyiv, 2015-2019 graduate"), h("Skills"), b("Python"))
    )

    top = sections[0]
    assert (top.types, top.label_source, top.confidence) == (["summary"], "fallback", "low")


def test_no_headings_gives_one_fallback_section() -> None:
    sections, warnings = split(make_doc(b("Some text."), b("More text about work.")))

    assert len(sections) == 1
    assert (sections[0].types, sections[0].label_source) == (["other"], "fallback")
    assert NO_HEADINGS_WARNING in warnings


def test_empty_document() -> None:
    assert split(make_doc()) == ([], [])


def test_empty_section_between_headings() -> None:
    doc = make_doc(h("Skills"), h("Experience"), b("Acme Widgets LLC"))
    sections, _ = split(doc)

    assert sections[0].blocks == []
    assert sections[0].content_start == sections[0].end == doc.blocks[0].end


def test_trailing_unlabelled_contacts_get_their_own_section() -> None:
    sections, _ = split(
        make_doc(
            h("Skills"),
            b("Python, SQL"),
            h("Education"),
            b("BSc Computer Science"),
            b("olena@example.com\n+380 67 123 4567"),
            b("linkedin.com/in/olena-demo"),
        )
    )

    assert summary(sections) == [
        (["skills"], "Skills"),
        (["education"], "Education"),
        (["contact"], None),
    ]
    assert sections[1].end == sections[1].blocks[-1].end
    contact = sections[2]
    assert (contact.label_source, contact.confidence, len(contact.blocks)) == (
        "pattern",
        "high",
        2,
    )


def test_contacts_in_the_middle_tag_the_section() -> None:
    sections, _ = split(
        make_doc(
            h("Skills"),
            b("Python"),
            b("olena@example.com"),
            b("SQL"),
            h("Education"),
            b("BSc"),
        )
    )

    assert sections[0].types == ["skills", "contact"]
    assert len(sections[0].blocks) == 3


def test_paragraph_mentioning_an_email_is_not_a_contact_block() -> None:
    sections, _ = split(
        make_doc(
            h("Summary"),
            b("Engineer.\nReach me at olena@example.com\nfor references."),
            h("Skills"),
            b("Python"),
        )
    )

    assert sections[0].types == ["summary"]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("olena@example.com", True),
        ("+380 67 123 4567", True),
        ("(067) 123-45-67", True),
        ("github.com/olena-demo", True),
        ("Acme Widgets LLC, 2021-2024", False),
        ("2015 - 2019 2019 - 2021", False),
        ("Order 12345", False),
    ],
)
def test_contact_line_detection(line: str, expected: bool) -> None:
    assert is_contact_line(line) is expected


def test_end_to_end_two_column_skills_stay_together() -> None:
    doc = parse(FIXTURES / "two_column.pdf")
    sections, _ = split(doc)

    skills = [s for s in sections if "skills" in s.types]
    assert len(skills) == 1
    skills_text = doc.full_text[skills[0].content_start : skills[0].end]
    for skill in ["Python", "SQL", "Apache Airflow", "Docker", "PostgreSQL"]:
        assert skill in skills_text
    assert sections[0].types == ["contact"]


@pytest.mark.parametrize(
    "name", ["single_column.pdf", "two_column.pdf", "left_sidebar.pdf", "table_cv.docx"]
)
def test_invariants_on_real_fixtures(name: str) -> None:
    split(parse(FIXTURES / name))
