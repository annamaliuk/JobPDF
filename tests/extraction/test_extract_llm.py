import json
from pathlib import Path

import pytest

from jobpdf.extraction import llm, prompt
from jobpdf.extraction.llm import (
    AnthropicToolCaller,
    FakeToolCaller,
    MissingAPIKeyError,
    RecordingToolCaller,
    ReplayMissError,
    ReplayToolCaller,
    ToolCallResult,
    replay_key,
)

TOOL = {"name": "record_candidate_profile", "description": "x", "input_schema": {}}
RESULT = ToolCallResult(
    tool_input={"summary": "Data engineer", "skills": []},
    stop_reason="tool_use",
    input_tokens=1200,
    output_tokens=300,
    model="fake-model",
)


def test_fake_returns_and_raises_in_order_and_records_calls() -> None:
    fake = FakeToolCaller([RESULT, RuntimeError("boom")])

    assert fake.call(system="s", user="u", tool=TOOL, max_tokens=10) is RESULT
    with pytest.raises(RuntimeError, match="boom"):
        fake.call(system="s2", user="u2", tool=TOOL, max_tokens=20)
    assert [c["max_tokens"] for c in fake.calls] == [10, 20]
    assert fake.calls[1]["user"] == "u2"


def test_missing_key_raises_at_construction_with_guidance() -> None:
    with pytest.raises(MissingAPIKeyError) as exc:
        AnthropicToolCaller()
    assert "JM-35" in str(exc.value) and ".env" in str(exc.value)


def test_model_comes_from_argument_env_or_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    assert AnthropicToolCaller(api_key="test-not-a-real-key").model == llm.DEFAULT_MODEL
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-env-model")
    assert AnthropicToolCaller(api_key="test-not-a-real-key").model == "claude-env-model"
    assert AnthropicToolCaller("claude-arg", api_key="test-not-a-real-key").model == "claude-arg"


def test_live_calls_are_blocked_in_tests() -> None:
    caller = AnthropicToolCaller(api_key="test-not-a-real-key")

    with pytest.raises(AssertionError, match="never run in tests"):
        caller.call(system="s", user="u", tool=TOOL, max_tokens=10)


def test_record_then_replay_roundtrip(tmp_path: Path) -> None:
    recorder = RecordingToolCaller(FakeToolCaller([RESULT]), tmp_path)

    recorded = recorder.call(system="sys", user="cv text", tool=TOOL, max_tokens=4096)
    replayed = ReplayToolCaller(tmp_path).call(
        system="sys", user="cv text", tool=TOOL, max_tokens=4096
    )

    assert recorded == replayed == RESULT
    files = list(tmp_path.glob("*.json"))
    key = replay_key(system="sys", user="cv text", tool_name=TOOL["name"])
    assert [f.name for f in files] == [f"{key}.json"]
    assert json.loads(files[0].read_text(encoding="utf-8"))["output_tokens"] == 300


def test_replay_miss_names_the_key(tmp_path: Path) -> None:
    key = replay_key(system="sys", user="other cv", tool_name=TOOL["name"])

    with pytest.raises(ReplayMissError) as exc:
        ReplayToolCaller(tmp_path).call(system="sys", user="other cv", tool=TOOL, max_tokens=10)
    assert key in str(exc.value)
    assert "RecordingToolCaller" in str(exc.value)


def test_replay_key_depends_on_prompt_version(monkeypatch: pytest.MonkeyPatch) -> None:
    before = replay_key(system="s", user="u", tool_name="t")
    monkeypatch.setattr(prompt, "PROMPT_VERSION", "999")
    monkeypatch.setattr(llm, "PROMPT_VERSION", "999")

    assert replay_key(system="s", user="u", tool_name="t") != before
