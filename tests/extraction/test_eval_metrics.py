from pathlib import Path

import pytest

from jobpdf.extraction import eval_metrics
from jobpdf.extraction.eval_dataturks import load_dataturks
from jobpdf.extraction.eval_metrics import (
    Counts,
    SkillCounts,
    evaluate_resume,
    find_leaks,
    match_strings,
    skill_quote_counts,
    year_counts,
)
from jobpdf.extraction.schema import CandidateProfile, ExtractedEducation, ExtractedJob, Skill

FIXTURE = Path(__file__).parent / "fixtures" / "dataturks_synthetic.jsonl"


def job(title: str, company: str | None, quote: str) -> ExtractedJob:
    return ExtractedJob(title=title, company=company, location=None, start="2020", end="present",
                        description=None, skills_used=[], quote=quote)


def skill(name: str, *quotes: str) -> Skill:
    return Skill(name=name, kind="hard", found_in=["skills_list"], quotes=list(quotes))


def profile(summary: str | None = None) -> CandidateProfile:
    """A hand-built prediction for fixture resume 0."""
    return CandidateProfile(
        summary=summary,
        skills=[
            skill("Python", "Python, SQL, Tableau"),
            skill("SQL", "Python, SQL, Tableau"),
            skill("Excel", "Excel"),  # not in the CV
        ],
        experience=[
            job("Data Analyst", "Example Corp", "Data Analyst\nExample Corp"),
            job("Junior Analyst", "Sample Labs Inc", "Junior Analyst\nSample Labs"),
            job("Intern", None, "Intern"),
        ],
        education=[
            ExtractedEducation(institution="Placeholder University", degree="Diploma in Arts",
                               field=None, start=None, end="2018", quote="B.Sc. Statistics"),
        ],
        languages=[],
        certifications=[],
        total_experience_months=None,
    )


@pytest.fixture(scope="module")
def resume0():
    resumes, _, _ = load_dataturks(FIXTURE)
    return resumes[0]


def test_match_strings_dedupes_and_counts() -> None:
    counts = match_strings(
        gold=["Data Analyst", " data  ANALYST ", "Junior Analyst", ""],
        predicted=["Data Analyst", "Senior Data Engineer", "Junior Analyst", "junior analyst"],
    )

    assert counts == Counts(tp=2, fp=1, fn=0)


def test_match_strings_is_one_to_one() -> None:
    # "acme" is a token subset of both gold strings (score 100), but it can match only one.
    assert match_strings(["Acme Widgets", "Acme"], ["Acme"]) == Counts(tp=1, fp=0, fn=1)


def test_match_strings_takes_best_pairs_first(monkeypatch: pytest.MonkeyPatch) -> None:
    scores = {("a", "p"): 90, ("b", "p"): 100, ("a", "q"): 95, ("b", "q"): 0}
    monkeypatch.setattr(eval_metrics.fuzz, "token_set_ratio", lambda g, p: scores[(g, p)])

    # In-order matching would pair a-p and leave b unmatched; best-first finds b-p and a-q.
    assert match_strings(["a", "b"], ["p", "q"]) == Counts(tp=2, fp=0, fn=0)


def test_match_strings_threshold() -> None:
    assert match_strings(["Data Analyst"], ["Data Engineer"]) == Counts(tp=0, fp=1, fn=1)
    assert match_strings(["Data Analyst"], ["Data Engineer"], threshold=0) == Counts(1, 0, 0)


def test_counts_report() -> None:
    assert Counts(tp=2, fp=1, fn=0).as_report() == {
        "tp": 2, "fp": 1, "fn": 0, "precision": 0.6667, "recall": 1.0, "f1": 0.8,
    }
    assert Counts().as_report()["precision"] is None
    assert Counts(tp=0, fp=0, fn=3).as_report() == {
        "tp": 0, "fp": 0, "fn": 3, "precision": None, "recall": 0.0, "f1": 0.0,
    }


def test_year_counts() -> None:
    assert year_counts({"2018"}, {"2018", "2016"}) == Counts(tp=1, fp=1, fn=0)


def test_skill_quote_counts() -> None:
    text = "SKILLS\nPython, SQL\nWORK\nUsed Python daily"
    spans = [(7, 18), (0, 6)]  # "Python, SQL" and "SKILLS"
    quotes = ["Python, SQL", "Python", "Used Python daily", "Rust", "Python"]

    counts = skill_quote_counts(text, quotes, spans)

    assert counts == SkillCounts(quotes=4, unlocated_quotes=1, quotes_inside=2,
                                 spans=2, spans_covered=1)
    report = counts.as_report()
    assert (report["quote_precision"], report["span_coverage"]) == (0.6667, 0.5)


def test_find_leaks_fires_on_names_contacts_and_mask_chars() -> None:
    output = {
        "summary": "Worked with Jane Placeholder",
        "skills": [{"quote": "ok"}, {"quote": "call +1 555 010 0199"}],
        "masked": "████ analyst",
        "lower": "jane   placeholder",
        "short": "Plain text",
        "months": 12,
    }

    assert find_leaks(output, ["Jane Placeholder", "Pl"]) == 4
    assert find_leaks({"summary": "Data analyst"}, ["Jane Placeholder"]) == 0


def test_evaluate_resume_hand_computed(resume0) -> None:
    metrics = evaluate_resume(resume0, profile())

    assert metrics.fields == {
        # gold {data analyst, junior analyst}; "Intern" is extra
        "designation": Counts(tp=2, fp=1, fn=0),
        # gold {example corp, sample labs}; "Sample Labs Inc" matches by token subset
        "companies": Counts(tp=2, fp=0, fn=0),
        "degree": Counts(tp=0, fp=1, fn=1),
        "college": Counts(tp=1, fp=0, fn=0),
        "graduation_year": Counts(tp=1, fp=0, fn=0),
    }
    assert metrics.skills == SkillCounts(quotes=2, unlocated_quotes=1, quotes_inside=1,
                                         spans=1, spans_covered=1)
    assert metrics.contact_leaks == 0


def test_evaluate_resume_detects_a_leaked_name(resume0) -> None:
    metrics = evaluate_resume(resume0, profile(summary="Jane Placeholder, analyst"))

    assert metrics.contact_leaks == 1


def test_resume_metrics_add_up(resume0) -> None:
    one = evaluate_resume(resume0, profile())

    total = one + one

    assert total.fields["designation"] == Counts(tp=4, fp=2, fn=0)
    assert total.skills.quotes == 4
    report = total.as_report()
    assert report["fields"]["designation"]["f1"] == 0.8
    assert set(report) == {"fields", "skills", "contact_leaks"}
