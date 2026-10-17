from datetime import date, datetime, timezone

import pytest

from jobpdf.report.wording import (
    join_natural,
    plural,
    posted_text,
    required_sentence,
    required_short,
    safe_text,
    similarity_name,
    years_text,
    years_tile,
)


def test_natural_lists() -> None:
    assert join_natural([]) == ""
    assert join_natural(["Python"]) == "Python"
    assert join_natural(["Python", "SQL"]) == "Python and SQL"
    assert join_natural(["Python", "SQL", "Airflow"]) == "Python, SQL and Airflow"


def test_singular_and_plural() -> None:
    assert plural(1, "skill") == "1 skill"
    assert plural(2, "skill") == "2 skills"
    assert plural(0, "skill") == "0 skills"
    assert plural(3, "vacancy", "vacancies") == "3 vacancies"


@pytest.mark.parametrize("months, text, tile", [
    (None, None, "—"),
    (0, "less than a year", "<1 yr"),
    (5, "less than a year", "<1 yr"),
    (6, "about half a year", "½ yr"),
    (12, "about 1 year", "1 yr"),
    (54, "about 4½ years", "4½ yrs"),
    (57, "about 5 years", "5 yrs"),  # 4.75 rounds up to 5
    (56, "about 4½ years", "4½ yrs"),
])
def test_experience_is_rounded_to_half_years(months, text, tile) -> None:
    assert years_text(months) == text
    assert years_tile(months) == tile


def test_posted_is_counted_from_today() -> None:
    today = date(2026, 10, 16)

    def at(day: int) -> datetime:
        return datetime(2026, 10, day, 23, tzinfo=timezone.utc)

    assert posted_text(None, today) is None
    assert posted_text(at(16), today) == "today"
    assert posted_text(at(17), today) == "today"  # a clock skew never reads "-1 days"
    assert posted_text(at(15), today) == "1 day ago"
    assert posted_text(at(4), today) == "12 days ago"


@pytest.mark.parametrize("args, sentence", [
    ((7, 7), "You have all 7 required skills."),
    ((6, 7), "You have 6 of the 7 required skills."),
    ((0, 3), "You have none of the 3 required skills."),
    ((1, 1), "You have the one required skill."),
    ((0, 1), "You don't have the one required skill."),
    ((0, 0), "This listing doesn't state required skills."),
    ((7, 7, 2, 4), "You have all 7 required skills and 2 of the 4 nice-to-have ones."),
    ((5, 6, 2, 2), "You have 5 of the 6 required skills and all 2 nice-to-have ones."),
    ((0, 2, 1, 1), "You have none of the 2 required skills, but the one nice-to-have one."),
])
def test_required_skill_sentences(args, sentence) -> None:
    assert required_sentence(*args) == sentence


def test_required_short() -> None:
    assert required_short(6, 7) == "6 of 7"
    assert required_short(0, 0) == "—"


def test_similarity_levels() -> None:
    assert similarity_name(None, 0.3) is None
    assert similarity_name(0.29, 0.3) == "low"
    assert similarity_name(0.3, 0.3) == "medium"
    assert similarity_name(0.6, 0.3) == "high"


def test_safe_text_removes_contact_data() -> None:
    assert safe_text("Data Engineer") == "Data Engineer"
    assert safe_text("Data Engineer (ann@example.com)") == "Data Engineer"
    assert safe_text("ann@example.com") is None
    assert safe_text("Call +380 67 123 4567 now") == "Call now"
    assert safe_text(None) is None
