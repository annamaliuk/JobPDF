"""Gemini caller, provider switch and .env loading (JM-15). No network."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import types

from jobpdf.extraction import llm
from jobpdf.extraction.llm import (
    AnthropicToolCaller,
    GeminiToolCaller,
    LLMResponseError,
    MissingAPIKeyError,
    default_caller,
    load_env_file,
)

# Captured at import, before the autouse fixture replaces it with a refusal.
REAL_CALL = GeminiToolCaller.call
TOOL = {
    "name": "record_candidate_profile",
    "description": "Record the profile.",
    "input_schema": {"type": "object", "properties": {"skills": {"type": "array"}},
                     "required": ["skills"], "additionalProperties": False},
}


class FakeModels:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def generate_content(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        return self.response


def response(calls, finish=types.FinishReason.STOP):
    return SimpleNamespace(
        function_calls=calls,
        candidates=[SimpleNamespace(finish_reason=finish)],
        usage_metadata=SimpleNamespace(prompt_token_count=1200, candidates_token_count=300),
        model_version="gemini-2.5-flash-001",
    )


def caller_with(fake_response) -> tuple[GeminiToolCaller, FakeModels]:
    caller = GeminiToolCaller(api_key="test-key")
    models = FakeModels(fake_response)
    caller._client = SimpleNamespace(models=models)
    return caller, models


def call(caller: GeminiToolCaller, max_tokens: int = 4096):
    return REAL_CALL(caller, system="sys", user="cv text", tool=TOOL, max_tokens=max_tokens)


def test_forces_the_one_function_and_maps_the_result() -> None:
    fn = SimpleNamespace(name=TOOL["name"], args={"skills": []})
    caller, models = caller_with(response([fn]))

    result = call(caller)

    request = models.requests[0]
    config = request["config"]
    assert request["model"] == "gemini-2.5-flash" and request["contents"] == "cv text"
    assert (config.system_instruction, config.temperature, config.max_output_tokens) == (
        "sys", 0, 4096)
    calling = config.tool_config.function_calling_config
    assert calling.mode == types.FunctionCallingConfigMode.ANY
    assert calling.allowed_function_names == [TOOL["name"]]
    declaration = config.tools[0].function_declarations[0]
    assert declaration.parameters_json_schema == TOOL["input_schema"]
    assert config.automatic_function_calling.disable is True
    assert result.tool_input == {"skills": []}
    assert (result.stop_reason, result.input_tokens, result.output_tokens) == ("stop", 1200, 300)
    assert result.model == "gemini-2.5-flash-001"


def test_thinking_is_off_for_flash_unless_configured(monkeypatch) -> None:
    fn = SimpleNamespace(name=TOOL["name"], args={"skills": []})
    caller, models = caller_with(response([fn]))
    call(caller)
    assert models.requests[0]["config"].thinking_config.thinking_budget == 0

    pro = GeminiToolCaller(model="gemini-2.5-pro", api_key="k")
    assert pro.thinking_budget is None
    monkeypatch.setenv("GEMINI_THINKING_BUDGET", "512")
    assert GeminiToolCaller(api_key="k").thinking_budget == 512


def test_min_interval_spaces_calls(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_MIN_INTERVAL_S", "30")
    fn = SimpleNamespace(name=TOOL["name"], args={"skills": []})
    caller, _ = caller_with(response([fn]))
    clock = iter([1000.0, 1000.0, 1010.0, 1030.0])
    sleeps = []
    monkeypatch.setattr(llm.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    call(caller)  # first call: no wait
    call(caller)  # 10 s later: waits the other 20 s
    assert sleeps == [20.0]


def test_truncated_answer_reads_as_max_tokens() -> None:
    caller, _ = caller_with(response(None, finish=types.FinishReason.MAX_TOKENS))
    result = call(caller)
    assert (result.stop_reason, result.tool_input) == ("max_tokens", {})


def test_answer_without_the_function_call_raises() -> None:
    caller, _ = caller_with(response([SimpleNamespace(name="other", args={})]))
    with pytest.raises(LLMResponseError, match="STOP"):
        call(caller)


def test_missing_key_names_the_gemini_variable() -> None:
    with pytest.raises(MissingAPIKeyError, match="GEMINI_API_KEY"):
        GeminiToolCaller()


def test_model_comes_from_argument_then_environment(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "gemini-env-model")
    assert GeminiToolCaller(api_key="k").model == "gemini-env-model"
    assert GeminiToolCaller(model="gemini-2.5-pro", api_key="k").model == "gemini-2.5-pro"


def test_default_caller_follows_llm_provider(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert isinstance(default_caller(), AnthropicToolCaller)
    monkeypatch.setenv("LLM_PROVIDER", " Gemini ")
    assert isinstance(default_caller(model="gemini-2.5-pro"), GeminiToolCaller)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        default_caller()


def test_api_errors_follow_llm_provider(monkeypatch) -> None:
    from google.genai import errors

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert llm.api_errors() == (errors.APIError,)


def test_load_env_file(tmp_path: Path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nGEMINI_API_KEY='file-key'\nLLM_PROVIDER=gemini\nANTHROPIC_API_KEY=\n"
        "JUDGE_MODEL=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("JUDGE_MODEL", "from-env")

    loaded = load_env_file(env)

    assert loaded == ["GEMINI_API_KEY", "LLM_PROVIDER"]
    assert llm.os.environ["GEMINI_API_KEY"] == "file-key"
    assert llm.os.environ["JUDGE_MODEL"] == "from-env"  # the environment wins
    assert "ANTHROPIC_API_KEY" not in llm.os.environ  # empty values are skipped
    assert load_env_file(tmp_path / "missing.env") == []
