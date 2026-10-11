import datetime as dt
from pathlib import Path

import pytest

from jobpdf.extraction import prompt
from jobpdf.extraction.extract import (
    MAX_TOKENS,
    MAX_TOKENS_RETRY,
    ExtractionError,
    extract_profile,
)
from jobpdf.extraction.llm import FakeToolCaller, MissingAPIKeyError, ToolCallResult
from jobpdf.extraction.models import RawBlock
from jobpdf.extraction.parser import _assemble
from jobpdf.extraction.sections import split_sections

TODAY = dt.date(2026, 10, 3)
CV_TEXT = [
    "Experience",
    "Data Engineer\nAcme Widgets LLC, 2021-2024",
    "Skills",
    "Python, SQL",
]
VALID = {
    "summary": None,
    "skills": [
        {"name": "Python", "kind": "hard", "found_in": "skills_list", "quote": "Python, SQL"},
        {"name": "SQL", "kind": "hard", "found_in": "skills_list", "quote": "Python, SQL"},
    ],
    "experience": [
        {
            "title": "Data Engineer", "company": "Acme Widgets LLC", "location": None,
            "start": "2021", "end": "2024", "description": None, "skills_used": [],
            "quote": "Data Engineer\nAcme Widgets LLC",
        }
    ],
    "education": [],
    "languages": [],
    "certifications": [],
}
INVALID = {**VALID, "experience": [{**VALID["experience"][0], "start": "May 2021"}]}


def doc_and_sections():
    raw = [RawBlock(text=t, page=0, bbox=None, font_size=10.0, is_bold=False) for t in CV_TEXT]
    doc = _assemble("pdf", raw, True, [])
    sections, _ = split_sections(doc)
    return doc, sections


def reply(tool_input: dict, stop_reason: str = "tool_use", tokens: tuple = (1000, 200)):
    return ToolCallResult(
        tool_input=tool_input, stop_reason=stop_reason,
        input_tokens=tokens[0], output_tokens=tokens[1], model="fake-model",
    )


def run(caller: FakeToolCaller, tmp_path: Path, **kwargs):
    doc, sections = doc_and_sections()
    return extract_profile(doc, sections, caller, cache_dir=tmp_path, today=TODAY, **kwargs)


def test_happy_path(tmp_path: Path) -> None:
    caller = FakeToolCaller([reply(VALID)])

    result = run(caller, tmp_path)

    assert len(caller.calls) == 1
    call = caller.calls[0]
    assert call["max_tokens"] == MAX_TOKENS
    assert call["system"] == prompt.SYSTEM_PROMPT
    assert call["tool"]["name"] == "record_candidate_profile"
    assert "Acme Widgets LLC" in call["user"]
    profile = result.profile
    assert [s.name for s in profile.skills] == ["Python", "SQL"]
    assert profile.total_experience_months == 48
    assert result.warnings == []
    assert (result.model, result.prompt_version, result.from_cache) == ("fake-model", "1", False)


def test_invalid_then_valid_sends_errors_back(tmp_path: Path) -> None:
    caller = FakeToolCaller([reply(INVALID), reply(VALID)])

    result = run(caller, tmp_path)

    assert len(caller.calls) == 2
    retry_user = caller.calls[1]["user"]
    assert retry_user.startswith(caller.calls[0]["user"])
    assert "experience.0.start" in retry_user
    assert "May 2021" in retry_user  # the previous output is shown back
    assert result.profile.experience[0].start == "2021"


def test_invalid_twice_raises(tmp_path: Path) -> None:
    caller = FakeToolCaller([reply(INVALID), reply({"skills": "nope"})])

    with pytest.raises(ExtractionError) as exc:
        run(caller, tmp_path)
    assert exc.value.errors
    assert any("skills" in line for line in exc.value.errors)


def test_max_tokens_then_success_uses_bigger_budget(tmp_path: Path) -> None:
    caller = FakeToolCaller([reply({}, stop_reason="max_tokens"), reply(VALID)])

    run(caller, tmp_path)

    assert [c["max_tokens"] for c in caller.calls] == [MAX_TOKENS, MAX_TOKENS_RETRY]


def test_max_tokens_twice_raises(tmp_path: Path) -> None:
    caller = FakeToolCaller([reply({}, "max_tokens"), reply({}, "max_tokens")])

    with pytest.raises(ExtractionError, match="8192"):
        run(caller, tmp_path)


def test_token_usage_is_summed_across_calls(tmp_path: Path) -> None:
    caller = FakeToolCaller([
        reply({}, "max_tokens", tokens=(1000, 4096)),
        reply(INVALID, tokens=(1000, 300)),
        reply(VALID, tokens=(1300, 280)),
    ])

    result = run(caller, tmp_path)

    assert (result.input_tokens, result.output_tokens) == (3300, 4676)


def test_cache_hit_and_prompt_version_miss(tmp_path: Path, monkeypatch) -> None:
    first = run(FakeToolCaller([reply(VALID)]), tmp_path)
    no_calls = FakeToolCaller([])

    cached = run(no_calls, tmp_path)

    assert no_calls.calls == []
    assert cached.from_cache is True
    assert cached.profile == first.profile
    assert len(list(tmp_path.glob("*.json"))) == 1

    monkeypatch.setattr(prompt, "PROMPT_VERSION", "2")
    miss = FakeToolCaller([reply(VALID)])
    assert run(miss, tmp_path).from_cache is False
    assert len(miss.calls) == 1


def test_cache_can_be_disabled(tmp_path: Path) -> None:
    run(FakeToolCaller([reply(VALID)]), tmp_path, use_cache=False)

    assert list(tmp_path.iterdir()) == []


def test_no_caller_and_no_key_raises_missing_key(tmp_path: Path) -> None:
    doc, sections = doc_and_sections()

    with pytest.raises(MissingAPIKeyError):
        extract_profile(doc, sections, cache_dir=tmp_path)
