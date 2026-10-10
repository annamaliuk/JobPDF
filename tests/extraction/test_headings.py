import pytest

from jobpdf.extraction.headings import match_heading, normalize_heading, split_parts


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  WORK EXPERIENCE  ", "work experience"),
        ("01. Education:", "education"),
        ("1) Skills —", "skills"),
        ("• Projects", "projects"),
        ("iv. Languages", "languages"),
        ("W O R K E X P E R I E N C E", "workexperience"),  # as JM-8 delivers it
        ("Education & Experience", "education & experience"),
        ("Skills (technical)", "skills technical"),
        ("Interests", "interests"),  # a leading "i" is not a roman numeral here
        ("Про себе:", "про себе"),
    ],
)
def test_normalize_heading(raw: str, expected: str) -> None:
    assert normalize_heading(raw) == expected


def test_split_parts_dedupes_and_keeps_order() -> None:
    assert split_parts("skills / tools, skills | languages") == ["skills", "tools", "languages"]
    assert split_parts("освіта та курси") == ["освіта", "курси"]
    assert split_parts("education and experience") == ["education", "experience"]


@pytest.mark.parametrize(
    ("heading", "types", "source"),
    [
        ("Experience", ["experience"], "keyword"),
        ("W O R K E X P E R I E N C E", ["experience"], "keyword"),
        ("01. Education:", ["education"], "keyword"),
        ("Experiences", ["experience"], "fuzzy"),
        ("Education & Experience", ["education", "experience"], "keyword"),
        ("Skills / Languages", ["skills", "languages"], "keyword"),
        ("Tools & Technologies", ["skills"], "keyword"),
        ("Досвід роботи", ["experience"], "keyword"),
        ("Освіта та курси", ["education", "certifications"], "keyword"),
        ("Education & Volunteering", ["education", "other"], "keyword"),
    ],
)
def test_match_heading(heading: str, types: list[str], source: str) -> None:
    match = match_heading(heading)

    assert match is not None
    assert match.types == types
    assert match.source == source


@pytest.mark.parametrize(
    "text",
    ["Data Engineer", "Acme Widgets LLC", "My Journey", "Tool", "Project Manager", "Olena"],
)
def test_non_headings_do_not_match(text: str) -> None:
    assert match_heading(text) is None
