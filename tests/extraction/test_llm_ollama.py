"""Ollama caller and its provider switch (JM-15). No network."""

import json

import pytest

from jobpdf.extraction import llm
from jobpdf.extraction.llm import (
    LLMResponseError,
    OllamaError,
    OllamaToolCaller,
    default_caller,
)

TOOL = {
    "name": "record_candidate_profile",
    "description": "Record the profile.",
    "input_schema": {"type": "object", "properties": {"skills": {"type": "array"}},
                     "required": ["skills"], "additionalProperties": False},
}


def caller_with(reply: dict) -> tuple[OllamaToolCaller, list]:
    caller = OllamaToolCaller()
    requests = []

    def fake_post(path, body):
        requests.append((path, body))
        return reply

    caller._post = fake_post
    return caller, requests


def reply(content: str, done_reason: str = "stop") -> dict:
    return {"model": "qwen2.5:7b", "message": {"role": "assistant", "content": content},
            "done_reason": done_reason, "prompt_eval_count": 900, "eval_count": 120}


def call(caller: OllamaToolCaller, max_tokens: int = 4096):
    return caller.call(system="sys", user="cv text", tool=TOOL, max_tokens=max_tokens)


def test_sends_the_schema_as_format_and_maps_the_result() -> None:
    caller, requests = caller_with(reply(json.dumps({"skills": ["Python"]})))

    result = call(caller)

    path, body = requests[0]
    assert path == "/api/chat" and body["model"] == "qwen2.5:7b"
    assert body["format"] == TOOL["input_schema"] and body["stream"] is False
    assert body["options"] == {"temperature": 0, "num_predict": 4096, "num_ctx": 16384}
    system, user = body["messages"]
    assert system["role"] == "system" and system["content"].startswith("sys")
    assert "Record the profile." in system["content"]
    assert user == {"role": "user", "content": "cv text"}
    assert result.tool_input == {"skills": ["Python"]}
    assert (result.stop_reason, result.input_tokens, result.output_tokens) == ("stop", 900, 120)
    assert result.model == "qwen2.5:7b"


def test_truncated_answer_reads_as_max_tokens() -> None:
    caller, _ = caller_with(reply('{"skills": ["Pyth', done_reason="length"))
    result = call(caller)
    assert (result.stop_reason, result.tool_input) == ("max_tokens", {})


def test_unparseable_answer_raises() -> None:
    caller, _ = caller_with(reply("Sure! Here is the profile"))
    with pytest.raises(LLMResponseError, match="unparseable"):
        call(caller)


def test_model_host_and_window_come_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11500")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    caller = OllamaToolCaller()
    assert (caller.model, caller.host, caller.num_ctx) == (
        "llama3.1:8b", "http://127.0.0.1:11500", 8192)
    assert OllamaToolCaller(model="gemma3:4b").model == "gemma3:4b"


def test_answer_length_cap_lowers_num_predict(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MAX_PREDICT", "2500")
    caller, requests = caller_with(reply(json.dumps({"skills": []})))
    call(caller, max_tokens=8192)
    call(caller, max_tokens=1000)
    assert [body["options"]["num_predict"] for _, body in requests] == [2500, 1000]


def test_provider_switch_needs_no_key(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert isinstance(default_caller(model="qwen2.5:7b"), OllamaToolCaller)
    assert llm.api_errors() == (OllamaError,)
