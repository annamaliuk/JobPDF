"""Generate the synthetic DataTurks-format fixture for the JM-14 evaluation tests.

Every name, company and contact detail is invented. The real DataTurks resumes
must never be used as fixtures (they live only in gitignored data/external/).

The fixture deliberately carries the dataset's quirks: inclusive end offsets,
an exclusive end, whitespace-padded span text, a broken (shifted) span, a
duplicate span, overlapping spans, a label given as a plain string, an
annotation without a label, unlabeled contact details, a null annotation and
offsets shifted by an "&amp;" entity.

Run from the repo root:  uv run python tests/extraction/make_dataturks_fixture.py
Output is deterministic, so re-running it on a clean checkout leaves git clean.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "dataturks_synthetic.jsonl"

RESUME_0 = """\
Jane Placeholder
jane.placeholder@example.com
+1 555 010 0199
Data Analyst at Example Corp

EXPERIENCE
Data Analyst
Example Corp, 2020 - present

Junior Analyst
Sample Labs, 2018 - 2020

SKILLS
Python, SQL, Tableau

EDUCATION
B.Sc. Statistics
Placeholder University, 2018"""

RESUME_1 = """\
John Sample
Backend Developer


WORK HISTORY
Backend Developer - Acme Widgets
2019 - 2023

TECHNICAL SKILLS
Go, PostgreSQL, Docker
Reach me: john.sample@example.com

EDUCATION
M.Sc. Computer Science, Example Institute of Technology, 2019"""

RESUME_2 = """\
Alex Example
Remote
alex@example.com
QA Engineer with 5 years of experience.

EXPERIENCE
QA Engineer, Test House Ltd (2017-2022)
QA Intern, Bugs &amp; Fixes (2016)

SKILLS
Selenium, Jira

EDUCATION
Diploma in Software Testing - Sample College - 2016"""

RESUME_3 = """\
Sam Placeholder
Intern"""


def inclusive(content: str, label: str | list[str], text: str, occurrence: int = 0) -> dict:
    """An annotation in DataTurks' format: inclusive end, exact text."""
    start = -1
    for _ in range(occurrence + 1):
        start = content.index(text, start + 1)
    labels = label if isinstance(label, list) else [label]
    return {"label": labels, "points": [{"start": start, "end": start + len(text) - 1,
                                         "text": text}]}


def records() -> list[dict]:
    c0 = RESUME_0
    ann0 = [
        inclusive(c0, "Name", "Jane Placeholder"),
        inclusive(c0, "Email Address", "jane.placeholder@example.com"),
        inclusive(c0, "Designation", "Data Analyst"),
        inclusive(c0, "Designation", "Data Analyst", occurrence=1),
        inclusive(c0, "Companies worked at", "Example Corp"),
        inclusive(c0, "Companies worked at", "Example Corp", occurrence=1),
        inclusive(c0, "Designation", "Junior Analyst"),
        inclusive(c0, "Companies worked at", "Sample Labs"),
        inclusive(c0, "Skills", "Python, SQL, Tableau"),
        inclusive(c0, "Degree", "B.Sc. Statistics"),
        inclusive(c0, "College Name", "Placeholder University"),
        inclusive(c0, "Graduation Year", "2018", occurrence=1),
    ]

    c1 = RESUME_1
    exclusive_end = inclusive(c1, "Designation", "Backend Developer")
    exclusive_end["points"][0]["end"] += 1  # end points one past the text
    padded = inclusive(c1, "Companies worked at", "Acme Widgets\n")
    padded["points"][0]["text"] = "Acme Widgets "  # window ends in "\n", text in " "
    shifted = inclusive(c1, "Degree", "M.Sc. Computer Science")
    shifted["points"][0]["start"] -= 4  # the dataset's +4 start shift
    shifted["points"][0]["end"] -= 4
    duplicate = inclusive(c1, "Designation", "Backend Developer", occurrence=1)
    unlabeled = inclusive(c1, "Skills", "Docker")
    unlabeled["label"] = []
    ann1 = [
        inclusive(c1, "Name", "John Sample"),
        exclusive_end,
        duplicate,
        duplicate,
        padded,
        inclusive(c1, "Skills", "Go, PostgreSQL, Docker"),
        inclusive(c1, "Skills", "TECHNICAL SKILLS\nGo, PostgreSQL"),  # overlaps the one above
        shifted,
        inclusive(c1, "College Name", "Example Institute of Technology"),
        inclusive(c1, "Graduation Year", "2019", occurrence=1),
        unlabeled,
    ]

    # Like the real dataset: content holds "&amp;", but the annotator's offsets
    # and texts were taken from the unescaped text, so later spans are shifted.
    c2 = RESUME_2.replace("&amp;", "&")
    string_label = inclusive(c2, "Location", "Remote")
    string_label["label"] = "Location"
    ann2 = [
        inclusive(c2, "Name", "Alex Example"),
        string_label,
        inclusive(c2, "Email Address", "alex@example.com"),
        inclusive(c2, "Years of Experience", "5 years"),
        inclusive(c2, "Designation", "QA Engineer", occurrence=1),
        inclusive(c2, "Companies worked at", "Test House Ltd"),
        inclusive(c2, "Designation", "QA Intern"),
        inclusive(c2, "Companies worked at", "Bugs & Fixes"),
        inclusive(c2, "Skills", "Selenium, Jira"),
        inclusive(c2, "Degree", "Diploma in Software Testing"),
        inclusive(c2, "College Name", "Sample College"),
        inclusive(c2, "Graduation Year", "2016", occurrence=1),
        inclusive(c2, "UNKNOWN", "Remote"),
    ]

    return [
        {"content": c0, "annotation": ann0, "extras": None},
        {"content": c1, "annotation": ann1, "extras": None},
        {"content": RESUME_2, "annotation": ann2, "extras": None},
        {"content": RESUME_3, "annotation": None, "extras": None},
    ]


def main() -> None:
    lines = [json.dumps(r, ensure_ascii=False) for r in records()]
    FIXTURE.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"written {FIXTURE}")


if __name__ == "__main__":
    main()
