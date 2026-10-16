"""LLM access behind one small interface (JM-11).

Every LLM call in the project goes through ``ToolCaller``: Anthropic and Gemini
(real; ``LLM_PROVIDER`` picks one, see ``default_caller``), Fake (scripted, for
tests), Replay (recorded responses), Recording (wraps a real caller to produce
replay fixtures) and ReplayOrRecord (the offline-first mix the evaluation
scripts use). Tests never touch the network.
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
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
# The SDK itself retries connection errors, 408, 409, 429 and 5xx with backoff.
MAX_RETRIES = 3
# Gemini's free tier answers 429 as soon as its per-minute quota is used up, so
# its retries wait longer than the SDK default.
GEMINI_RETRY_ATTEMPTS = 6
GEMINI_RETRY_MAX_DELAY_S = 60.0
PROVIDERS = ("anthropic", "gemini")
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class MissingAPIKeyError(RuntimeError):
    """No API key for the chosen provider: live LLM calls are impossible (JM-35)."""


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


class GeminiToolCaller:
    """Forced function calling via the Google Gen AI SDK (Gemini API).

    ``mode="ANY"`` with one allowed function is Gemini's equivalent of a forced
    ``tool_choice``. The tool's JSON schema is passed as is through
    ``parameters_json_schema``. Free-tier requests may be used by Google to
    improve its products; CV text is masked before it gets here.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise MissingAPIKeyError(
                "GEMINI_API_KEY is not set, so LLM_PROVIDER=gemini cannot call the LLM. "
                "Create a key in Google AI Studio and put GEMINI_API_KEY=... into your "
                "local .env or the environment."
            )
        from google import genai  # deferred like anthropic above
        from google.genai import types

        self._types = types
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        retry = types.HttpRetryOptions(
            attempts=GEMINI_RETRY_ATTEMPTS, max_delay=GEMINI_RETRY_MAX_DELAY_S
        )
        self._client = genai.Client(
            api_key=key, http_options=types.HttpOptions(retry_options=retry)
        )

    def call(self, *, system: str, user: str, tool: dict, max_tokens: int) -> ToolCallResult:
        types = self._types
        declaration = types.FunctionDeclaration(
            name=tool["name"],
            description=tool["description"],
            parameters_json_schema=tool["input_schema"],
        )
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            max_output_tokens=max_tokens,
            tools=[types.Tool(function_declarations=[declaration])],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=[tool["name"]]
                )
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = self._client.models.generate_content(
            model=self.model, contents=user, config=config
        )
        finish = _gemini_finish_reason(response)
        call = next(
            (c for c in response.function_calls or [] if c.name == tool["name"]), None
        )
        # Mapped to the Anthropic spelling, which the extractor's retry checks for.
        stop_reason = "max_tokens" if finish == "MAX_TOKENS" else (finish or "").lower()
        if call is None and stop_reason != "max_tokens":
            raise LLMResponseError(
                f"Expected a {tool['name']!r} function call, got none "
                f"(finish_reason={finish!r})"
            )
        usage = response.usage_metadata
        return ToolCallResult(
            tool_input=dict(call.args or {}) if call is not None else {},
            stop_reason=stop_reason,
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
            model=response.model_version or self.model,
        )


def _gemini_finish_reason(response: Any) -> str | None:
    candidates = response.candidates or []
    reason = candidates[0].finish_reason if candidates else None
    return getattr(reason, "name", None) or (str(reason) if reason else None)


def provider() -> str:
    """The provider named by LLM_PROVIDER (default "anthropic")."""
    name = (os.environ.get("LLM_PROVIDER") or "anthropic").strip().lower()
    if name not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {PROVIDERS}, got {name!r}")
    return name


def default_caller(model: str | None = None) -> ToolCaller:
    """The real caller for LLM_PROVIDER; raises MissingAPIKeyError without its key."""
    if provider() == "gemini":
        return GeminiToolCaller(model=model)
    return AnthropicToolCaller(model=model)


def api_errors() -> tuple[type[Exception], ...]:
    """The SDK errors that mean a live call failed, for the chosen provider."""
    if provider() == "gemini":
        from google.genai import errors

        return (errors.APIError,)
    import anthropic

    return (anthropic.APIError,)


def load_env_file(env_file: Path = ENV_FILE) -> list[str]:
    """Copy KEY=VALUE lines of a .env file into ``os.environ``; returns the names set.

    For the offline scripts only, so a key kept in the repo's gitignored .env
    works without exporting it. A variable already in the environment always
    wins, and empty values are skipped. Values are never logged.
    """
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    loaded = []
    for line in lines:
        if line.lstrip().startswith("#"):
            continue
        key, sep, value = line.strip().partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if sep and key and value and not os.environ.get(key):
            os.environ[key] = value
            loaded.append(key)
    return loaded


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


class ReplayOrRecord:
    """Serve a recorded response if there is one, else ask ``live`` and record it.

    A recorded truncated answer (stop_reason "max_tokens") counts as a miss, so
    a bigger-budget retry really reaches the API and overwrites it.
    """

    def __init__(self, live: ToolCaller, recordings: Path) -> None:
        self._replay = ReplayToolCaller(recordings)
        self._record = RecordingToolCaller(live, recordings)
        self.model = live.model

    def call(self, *, system: str, user: str, tool: dict, max_tokens: int) -> ToolCallResult:
        kwargs = {"system": system, "user": user, "tool": tool, "max_tokens": max_tokens}
        try:
            recorded = self._replay.call(**kwargs)
        except ReplayMissError:
            return self._record.call(**kwargs)
        if recorded.stop_reason == "max_tokens":
            return self._record.call(**kwargs)
        return recorded
