import datetime as dt

from jobpdf.extraction.postprocess import (
    build_profile,
    check_pii_leaks,
    check_quotes,
    dedupe_skills,
    total_experience_months,
)
from jobpdf.extraction.schema import ExtractedJob, ExtractedProfile, SkillMention

TODAY = dt.date(2026, 10, 3)


def job(start: str | None, end: str | None, title: str = "Engineer", quote: str = "Engineer"):
    return ExtractedJob(
        title=title, company=None, location=None, start=start, end=end,
        description=None, skills_used=[], quote=quote,
    )


def mention(name: str, found_in: str = "skills_list", kind: str = "hard", quote: str = "q"):
    return SkillMention(name=name, kind=kind, found_in=found_in, quote=quote)


def profile(**kwargs) -> ExtractedProfile:
    base = dict(summary=None, skills=[], experience=[], education=[], languages=[],
                certifications=[])
    return ExtractedProfile(**{**base, **kwargs})


def test_months_inclusive_for_a_single_job() -> None:
    assert total_experience_months([job("2021-01", "2021-12")], TODAY) == (12, [])


def test_overlapping_jobs_are_counted_once() -> None:
    jobs = [job("2020-01", "2020-12"), job("2020-07", "2021-06"), job("2023-01", "2023-03")]

    months, warnings = total_experience_months(jobs, TODAY)

    assert months == 18 + 3
    assert warnings == []


def test_back_to_back_jobs_merge_without_double_counting() -> None:
    assert total_experience_months([job("2020-01", "2020-06"), job("2020-07", "2020-12")], TODAY)[
        0
    ] == 12


def test_year_only_dates_span_january_to_december() -> None:
    assert total_experience_months([job("2019", "2021")], TODAY)[0] == 36


def test_present_uses_injected_today() -> None:
    assert total_experience_months([job("2026-01", "present")], TODAY)[0] == 10
    assert total_experience_months([job("2026-01", "present")], dt.date(2026, 3, 31))[0] == 3


def test_missing_start_or_end_is_skipped_with_warning() -> None:
    jobs = [job(None, "2020", title="Intern"), job("2021", None, title="Analyst"),
            job("2022-01", "2022-02")]

    months, warnings = total_experience_months(jobs, TODAY)

    assert months == 2
    assert any("'Intern'" in w and "no start date" in w for w in warnings)
    assert any("'Analyst'" in w and "no end date" in w for w in warnings)


def test_no_usable_job_gives_none() -> None:
    assert total_experience_months([job(None, None)], TODAY)[0] is None
    assert total_experience_months([], TODAY) == (None, [])


def test_dedupe_merges_spellings_places_and_quotes() -> None:
    skills = dedupe_skills([
        mention("Python", "skills_list", quote="Python, SQL"),
        mention("python ", "experience", quote="pipelines in python"),
        mention("Python", "skills_list", quote="Python, SQL"),
        mention("Teamwork", "summary", kind="soft", quote="teamwork"),
        mention("teamwork", "experience", kind="hard", quote="Teamwork tools"),
    ])

    assert [(s.name, s.found_in, s.quotes, s.kind) for s in skills] == [
        ("Python", ["skills_list", "experience"], ["Python, SQL", "pipelines in python"], "hard"),
        ("Teamwork", ["summary", "experience"], ["teamwork", "Teamwork tools"], "hard"),
    ]


def test_fabricated_quote_warns_but_item_is_kept() -> None:
    masked = "Data Engineer\nAcme Widgets LLC, 2021-2024\nPython,   SQL"
    extracted = profile(
        skills=[mention("Python", quote="Python, SQL"), mention("Rust", quote="Rust expert")],
        experience=[job("2021", "2024", quote="Data Engineer\nAcme Widgets LLC")],
    )

    warnings = check_quotes(extracted, masked)
    built, _ = build_profile(extracted, masked, TODAY)

    assert warnings == ["quote not found: Rust expert"]
    assert [s.name for s in built.skills] == ["Python", "Rust"]


def test_pii_leak_warning_fires() -> None:
    extracted = profile(
        summary="Reach me at olena@example.com",
        experience=[job("2021", "2024", title="Engineer ███")],
    )

    warnings = check_pii_leaks(extracted)

    assert warnings == [
        "possible personal data in output at summary",
        "possible personal data in output at experience[0].title",
    ]


def test_build_profile_fills_derived_fields() -> None:
    extracted = profile(
        skills=[mention("SQL", quote="SQL")],
        experience=[job("2021", "2021", quote="Engineer")],
    )

    built, warnings = build_profile(extracted, "Engineer SQL", TODAY)

    assert built.schema_version == "1.0"
    assert built.total_experience_months == 12
    assert built.skills[0].quotes == ["SQL"]
    assert warnings == []
