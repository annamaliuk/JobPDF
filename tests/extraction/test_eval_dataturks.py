import json
from pathlib import Path

import pytest

from jobpdf.extraction.eval_dataturks import (
    MASK_CHAR,
    GoldSpan,
    load_dataturks,
    mask_contacts,
    repair_span,
)

FIXTURE = Path(__file__).parent / "fixtures" / "dataturks_synthetic.jsonl"


@pytest.fixture(scope="module")
def raw() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def loaded():
    return load_dataturks(FIXTURE)


def test_repair_inclusive_exclusive_and_stripped() -> None:
    content = "Role: Data Analyst\nNext"

    assert repair_span(content, 6, 17, "Data Analyst") == (6, 18)  # inclusive end
    assert repair_span(content, 6, 18, "Data Analyst") == (6, 18)  # exclusive end
    # The window "Data Analyst\n" matches "Data Analyst " once whitespace is ignored.
    assert repair_span(content, 6, 18, "Data Analyst ") == (6, 18)
    assert repair_span(content, 6, 17, " Data Analyst") == (6, 18)
    # An exact match wins even if it includes whitespace.
    assert repair_span(content, 5, 17, " Data Analyst") == (5, 18)


@pytest.mark.parametrize(
    ("start", "end", "text"),
    [(2, 13, "Data Analyst"), (-1, 5, "Role:"), (6, 4, "x"), (6, 17, "   "), (20, 40, "Next")],
)
def test_repair_rejects_what_does_not_match(start: int, end: int, text: str) -> None:
    assert repair_span("Role: Data Analyst\nNext", start, end, text) is None


def test_load_counts(loaded) -> None:
    resumes, stats, _ = loaded

    assert [r.index for r in resumes] == [0, 1, 2, 3]
    assert stats.records == 4 and stats.invalid_records == 0
    assert stats.invalid_spans == 1  # the shifted Degree span in resume 1
    assert stats.duplicate_spans == 1
    assert stats.unlabeled_annotations == 1
    assert stats.label_counts["Designation"] == 7
    assert stats.label_counts["Location"] == 1  # given as a plain string label
    assert stats.label_counts["UNKNOWN"] == 1


def test_warnings_name_index_and_offsets_only(loaded) -> None:
    _, _, warnings = loaded

    assert warnings == [
        "resume 1: Degree span [172, 193] does not match its text, excluded from gold",
        "resume 1: annotation without a label at [(124, 129)]",
    ]


def test_invalid_span_is_excluded_and_duplicate_kept_once(loaded) -> None:
    resumes, _, _ = loaded
    spans = resumes[1].spans

    assert not [s for s in spans if s.label == "Degree"]
    designations = [s for s in spans if s.label == "Designation"]
    assert len(designations) == 2 and len(set(designations)) == 2
    # Overlapping Skills spans both stay.
    assert len([s for s in spans if s.label == "Skills"]) == 2


def test_repaired_spans_are_half_open_and_point_at_the_text(loaded, raw) -> None:
    resumes, _, _ = loaded

    for resume, record in zip(resumes, raw, strict=True):
        for span in resume.spans:
            if span.label in ("Name", "Email Address"):
                continue
            assert resume.text[span.start : span.end] == record["content"][span.start : span.end]
            assert resume.text[span.start : span.end].strip() == resume.text[span.start : span.end]


def test_masking_keeps_length_and_hides_contacts(loaded, raw) -> None:
    resumes, _, _ = loaded

    for resume, record in zip(resumes, raw, strict=True):
        assert len(resume.text) == len(record["content"])
        assert "@example.com" not in resume.text
    assert "Jane Placeholder" not in resumes[0].text
    assert "555 010 0199" not in resumes[0].text  # unlabeled phone
    assert "john.sample" not in resumes[1].text  # unlabeled email
    name = next(s for s in resumes[0].spans if s.label == "Name")
    assert resumes[0].text[name.start : name.end] == MASK_CHAR * (name.end - name.start)


def test_contact_strings_stay_in_memory_only(loaded) -> None:
    resumes, _, _ = loaded
    resume = resumes[0]

    assert resume.contact_strings == ["Jane Placeholder", "jane.placeholder@example.com"]
    assert "contact_strings" not in resume.model_dump()
    for shown in (resume.model_dump_json(), repr(resume)):
        assert "Jane Placeholder" not in shown and "jane.placeholder@" not in shown


def test_mask_contacts_ignores_other_labels() -> None:
    content = "Jane Placeholder, Data Analyst"
    spans = [GoldSpan(label="Name", start=0, end=16),
             GoldSpan(label="Designation", start=18, end=30)]

    masked, originals = mask_contacts(content, spans)

    assert masked == MASK_CHAR * 16 + ", Data Analyst"
    assert originals == ["Jane Placeholder"]


def test_invalid_record_is_counted_not_dropped_silently(tmp_path: Path, raw) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(raw[3]) + "\n{not json\n" + json.dumps({"x": 1}) + "\n",
                    encoding="utf-8")

    resumes, stats, warnings = load_dataturks(path)

    assert [r.index for r in resumes] == [0]
    assert (stats.records, stats.invalid_records) == (3, 2)
    assert warnings == [
        "resume 1: record is not valid DataTurks JSON, skipped",
        "resume 2: record is not valid DataTurks JSON, skipped",
    ]
