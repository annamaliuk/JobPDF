"""LLM access behind one small interface (JM-11).

Every LLM call in the project goes through ``ToolCaller``: Anthropic (real),
Fake (scripted, for tests), Replay (recorded responses) and Recording (wraps a
real caller to produce replay fixtures). Tests never touch the network.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from jobpdf.extraction.prompt import PROMPT_VERSION

DEFAULT_MODEL = "claude-sonnet-4-6"
# The SDK itself retries connection errors, 408, 409, 429 and 5xx with backoff.
MAX_RETRIES = 3


class MissingAPIKeyError(RuntimeError):
    """No ANTHROPIC_API_KEY: live LLM calls are impossible (keys arrive with JM-35)."""


class LLMResponseError(RuntimeError):
    """The model answered without the forced tool call."""


class ReplayMissError(LookupError):
    """No recorded response for this exact prompt."""


@dataclass(frozen=True)
class ToolCallResult:
    tool_input: dict[str, Any]
    stop_reason: str
    input_tokens: int
    output_tokens: int
    model: str


class ToolCaller(Protocol):
    # The model name is needed before a call (it is part of the extraction cache key).
    model: str

    def call(self, *, system: str, user: str, tool: dict, max_tokens: int) -> ToolCallResult: ...


class AnthropicToolCaller:
    """Forced tool use via the Anthropic SDK.

    Note: forced ``tool_choice`` and ``temperature`` work on claude-sonnet-4-6 (the
    default); newer models such as claude-opus-5-5 / claude-sonnet-5-5 reject them
    with HTTP 400, so an ANTHROPIC_MODEL override to those needs a code change.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise MissingAPIKeyError(
                "ANTHROPIC_API_KEY is not set, so structured extraction cannot call the LLM. "
                "API keys arrive with JM-35: put ANTHROPIC_API_KEY=... into your local .env "
                "(see .env.example) or the environment. Offline, use FakeToolCaller or "
                "ReplayToolCaller."
            )
        import anthropic  # deferred: importing this module must stay cheap and offline

        self.model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
        self._client = anthropic.Anthropic(api_key=key, max_retries=MAX_RETRIES)

    def call(self, *, system: str, user: str, tool: dict, max_tokens: int) -> ToolCallResult:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            temperature=0,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": user}],
        )
        block = next((b for b in response.content if b.type == "tool_use"), None)
        # A truncated answer (max_tokens) may lack the block or carry partial input;
        # return it so the caller can retry with a larger budget.
        if block is None and response.stop_reason != "max_tokens":
            raise LLMResponseError(
                f"Expected a {tool['name']!r} tool call, got none "
                f"(stop_reason={response.stop_reason!r})"
            )
        return ToolCallResult(
            tool_input=dict(block.input) if block is not None else {},
            stop_reason=response.stop_reason or "",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )


@dataclass
class FakeToolCaller:
    """Returns (or raises) scripted responses in order and records every call."""

    responses: list[ToolCallResult | Exception]
    model: str = "fake-model"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def call(self, *, system: str, user: str, tool: dict, max_tokens: int) -> ToolCallResult:
        self.calls.append({"system": system, "user": user, "tool": tool, "max_tokens": max_tokens})
        if not self.responses:
            raise AssertionError("FakeToolCaller ran out of scripted responses")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def replay_key(*, system: str, user: str, tool_name: str) -> str:
    payload = {
        "system": system,
        "user": user,
        "tool_name": tool_name,
        "prompt_version": PROMPT_VERSION,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ReplayToolCaller:
    """Serves responses recorded by RecordingToolCaller; never calls the network."""

    def __init__(self, fixtures_dir: Path, model: str = "replay") -> None:
        self.fixtures_dir = Path(fixtures_dir)
        self.model = model

    def call(self, *, system: str, user: str, tool: dict, max_tokens: int) -> ToolCallResult:
        key = replay_key(system=system, user=user, tool_name=tool["name"])
        path = self.fixtures_dir / f"{key}.json"
        if not path.is_file():
            where = str(self.fixtures_dir)
            raise ReplayMissError(
                f"No recorded response {key}.json in {where}. Record it in the live step: "
                f"RecordingToolCaller(AnthropicToolCaller(), {where!r}) "
                "(needs ANTHROPIC_API_KEY, JM-35)."
            )
        return ToolCallResult(**json.loads(path.read_text(encoding="utf-8")))


class RecordingToolCaller:
    """Delegates to ``inner`` and saves each response in the replay format."""

    def __init__(self, inner: ToolCaller, fixtures_dir: Path) -> None:
        self.inner = inner
        self.fixtures_dir = Path(fixtures_dir)
        self.model = inner.model

    def call(self, *, system: str, user: str, tool: dict, max_tokens: int) -> ToolCallResult:
        result = self.inner.call(system=system, user=user, tool=tool, max_tokens=max_tokens)
        key = replay_key(system=system, user=user, tool_name=tool["name"])
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        (self.fixtures_dir / f"{key}.json").write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result
