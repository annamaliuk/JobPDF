"""Gold outputs: what the model SHOULD return for the synthetic fixtures (JM-11).

They double as evaluation labels for JM-14, so they are checked against the
real parse -> split_sections -> mask pipeline here.
"""

import datetime as dt
import json
from pathlib import Path

import pytest

from jobpdf.extraction import parse
from jobpdf.extraction.extract import extract_profile
from jobpdf.extraction.llm import FakeToolCaller, ToolCallResult
from jobpdf.extraction.postprocess import check_pii_leaks, check_quotes
from jobpdf.extraction.prompt import build_user_message, mask_pii
from jobpdf.extraction.schema import ExtractedProfile
from jobpdf.extraction.sections import split_sections

FIXTURES = Path(__file__).parent / "fixtures"
GOLD = FIXTURES / "gold"
SOURCES = {
    "single_column": "single_column.pdf",
    "two_column": "two_column.pdf",
    "table_cv": "table_cv.docx",
    "left_sidebar": "left_sidebar.pdf",
}


def load_gold(name: str) -> dict:
    return json.loads((GOLD / f"{name}.json").read_text(encoding="utf-8"))


def test_every_gold_file_has_a_source_fixture() -> None:
    assert sorted(p.stem for p in GOLD.glob("*.json")) == sorted(SOURCES)


@pytest.mark.parametrize("name", SOURCES)
def test_gold_is_valid_and_quoted_from_the_masked_cv(name: str) -> None:
    gold = ExtractedProfile.model_validate(load_gold(name))
    doc = parse(FIXTURES / SOURCES[name])
    masked, _ = mask_pii(doc.full_text)

    assert check_quotes(gold, masked) == []
    assert check_pii_leaks(gold) == []


@pytest.mark.parametrize("name", SOURCES)
def test_pipeline_with_gold_response(name: str, tmp_path: Path) -> None:
    doc = parse(FIXTURES / SOURCES[name])
    sections, _ = split_sections(doc)
    gold = load_gold(name)
    caller = FakeToolCaller([
        ToolCallResult(tool_input=gold, stop_reason="tool_use", input_tokens=1500,
                       output_tokens=600, model="fake-model")
    ])

    result = extract_profile(doc, sections, caller, cache_dir=tmp_path,
                             today=dt.date(2026, 10, 3))

    expected = ExtractedProfile.model_validate(gold)
    profile = result.profile
    assert result.warnings == []
    assert profile.experience == expected.experience
    assert profile.education == expected.education
    assert profile.languages == expected.languages
    assert [s.name for s in profile.skills] == [s.name for s in expected.skills]
    assert profile.total_experience_months is not None
    # What the model saw: masked text, no pure contact section.
    user = caller.calls[0]["user"]
    assert user == build_user_message(doc, sections)
    assert "@example.com" not in user
