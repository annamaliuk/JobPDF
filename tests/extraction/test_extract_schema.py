import json

import pytest
from pydantic import ValidationError

from jobpdf.extraction.schema import (
    CandidateProfile,
    ExtractedEducation,
    ExtractedJob,
    ExtractedProfile,
    Skill,
    VacancyProfile,
    tool_schema,
)

JOB = {
    "title": "Data Engineer",
    "company": "Acme Widgets LLC",
    "location": None,
    "start": "2021",
    "end": "present",
    "description": None,
    "skills_used": [],
    "quote": "Data Engineer",
}
# "name" alone is fine (a skill's or certification's name); a person's name is not.
PII_WORDS = ("email", "phone", "url", "linkedin", "github", "address", "website")
PERSON_NAME_FIELDS = {"full_name", "first_name", "last_name", "candidate_name", "person_name"}


def test_tool_schema_is_self_contained() -> None:
    tool = tool_schema()
    text = json.dumps(tool)

    assert tool["name"] == "record_candidate_profile"
    assert "$ref" not in text and "$defs" not in text
    jobs = tool["input_schema"]["properties"]["experience"]["items"]
    assert "title" in jobs["properties"]  # the real field survives title-stripping
    assert jobs["properties"]["quote"]["maxLength"] == 200


@pytest.mark.parametrize("value", ["2021", "2021-01", "2021-12", None])
def test_valid_start_dates(value: str | None) -> None:
    assert ExtractedJob.model_validate({**JOB, "start": value}).start == value


@pytest.mark.parametrize("value", ["2021-13", "2021-1", "21", "2021/05", "present", "May 2021"])
def test_invalid_start_dates(value: str) -> None:
    with pytest.raises(ValidationError):
        ExtractedJob.model_validate({**JOB, "start": value})


@pytest.mark.parametrize("value", ["present", "2024", "2024-06", None])
def test_valid_end_dates(value: str | None) -> None:
    assert ExtractedJob.model_validate({**JOB, "end": value}).end == value


@pytest.mark.parametrize("value", ["Present", "now", "2024-00", "current"])
def test_invalid_end_dates(value: str) -> None:
    with pytest.raises(ValidationError):
        ExtractedJob.model_validate({**JOB, "end": value})


def test_education_end_allows_present_but_start_does_not() -> None:
    base = {"institution": None, "degree": None, "field": None, "start": None, "quote": "BSc"}
    ExtractedEducation.model_validate({**base, "end": "present"})
    with pytest.raises(ValidationError):
        ExtractedEducation.model_validate({**base, "start": "present", "end": None})


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedJob.model_validate({**JOB, "email": "someone@example.com"})


def test_quote_length_is_bounded() -> None:
    with pytest.raises(ValidationError):
        ExtractedJob.model_validate({**JOB, "quote": "x" * 201})
    with pytest.raises(ValidationError):
        ExtractedJob.model_validate({**JOB, "quote": ""})


@pytest.mark.parametrize(
    "model", [ExtractedProfile, CandidateProfile, Skill, VacancyProfile]
)
def test_no_pii_fields_anywhere(model) -> None:
    def keys(node: object) -> set[str]:
        found: set[str] = set()
        if isinstance(node, dict):
            for key, value in node.get("properties", {}).items():
                found.add(key.lower())
                found |= keys(value)
            for value in node.values():
                found |= keys(value)
        elif isinstance(node, list):
            for item in node:
                found |= keys(item)
        return found

    fields = keys(tool_schema(model)["input_schema"])
    assert fields  # the walk found the schema's fields
    assert not {f for f in fields if any(word in f for word in PII_WORDS)}
    assert not fields & PERSON_NAME_FIELDS
