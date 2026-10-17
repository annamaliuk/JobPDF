"""LLM access behind one small interface (JM-11).

Every LLM call in the project goes through ``ToolCaller``: Gemini and Ollama
(real; ``LLM_PROVIDER`` picks one, see ``default_caller``), Fake (scripted, for
tests), Replay (recorded responses), Recording (wraps a real caller to produce
replay fixtures) and ReplayOrRecord (the offline-first mix the evaluation
scripts use). Tests never touch the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from jobpdf.extraction.prompt import PROMPT_VERSION

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
# Gemini's free tier answers 429 as soon as its per-minute quota is used up, so
# its retries wait longer than the SDK default.
GEMINI_RETRY_ATTEMPTS = 6
GEMINI_RETRY_MAX_DELAY_S = 60.0
# Flash models think by default, and thinking tokens count against
# max_output_tokens: long CVs then end in MAX_TOKENS or MALFORMED_FUNCTION_CALL.
# Thinking is off for them unless GEMINI_THINKING_BUDGET says otherwise.
DEFAULT_FLASH_THINKING_BUDGET = 0
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
# Ollama's own default window (4096 tokens) is too small for a CV plus the prompt.
DEFAULT_OLLAMA_NUM_CTX = 16384
# A local model on a laptop CPU can take minutes for one long answer.
OLLAMA_TIMEOUT_S = 3600
PROVIDERS = ("gemini", "ollama")
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class MissingAPIKeyError(RuntimeError):
    """No API key for the chosen provider: live LLM calls are impossible (JM-35)."""


class LLMResponseError(RuntimeError):
    """The model answered without the forced tool call."""


class ReplayMissError(LookupError):
    """No recorded response for this exact prompt."""


class OllamaError(RuntimeError):
    """The local Ollama server could not be reached or refused the request."""


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
                "GEMINI_API_KEY is not set, so the LLM cannot be called. Create a key in "
                "Google AI Studio and put GEMINI_API_KEY=... into your local .env or the "
                "environment (see .env.example). Offline, use FakeToolCaller or "
                "ReplayToolCaller; LLM_PROVIDER=ollama runs a local model without a key."
            )
        from google import genai  # deferred: importing this module must stay cheap and offline
        from google.genai import types

        self._types = types
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        budget = os.environ.get("GEMINI_THINKING_BUDGET")
        self.thinking_budget = (
            int(budget) if budget
            else DEFAULT_FLASH_THINKING_BUDGET if "flash" in self.model else None
        )
        retry = types.HttpRetryOptions(
            attempts=GEMINI_RETRY_ATTEMPTS, max_delay=GEMINI_RETRY_MAX_DELAY_S
        )
        self._client = genai.Client(
            api_key=key, http_options=types.HttpOptions(retry_options=retry)
        )
        # Free-tier models with a small tokens-per-minute cap (Gemma: 16K) fail
        # with 5xx when long calls come back to back; GEMINI_MIN_INTERVAL_S spaces them.
        self.min_interval_s = float(os.environ.get("GEMINI_MIN_INTERVAL_S") or 0)
        self._last_call = 0.0

    def call(self, *, system: str, user: str, tool: dict, max_tokens: int) -> ToolCallResult:
        wait = self._last_call + self.min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()
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
            thinking_config=(
                types.ThinkingConfig(thinking_budget=self.thinking_budget)
                if self.thinking_budget is not None else None
            ),
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
        # Mapped to the spelling the extractor's retry checks for.
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


class OllamaToolCaller:
    """A local model served by Ollama; no key, and nothing leaves the machine.

    Ollama has no forced tool call, so the tool's JSON schema goes in ``format``
    (structured output) and the answer is the tool input itself. The tool's
    description is appended to the system prompt, where a tool call would carry it.
    """

    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        self.model = model or os.environ.get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
        if "://" not in self.host:
            self.host = f"http://{self.host}"
        self.num_ctx = int(os.environ.get("OLLAMA_NUM_CTX") or DEFAULT_OLLAMA_NUM_CTX)
        # Optional ceiling on answer length: on a CPU a looping answer otherwise runs
        # for an hour before max_tokens stops it. A capped answer reads as truncated.
        cap = os.environ.get("OLLAMA_MAX_PREDICT")
        self.max_predict = int(cap) if cap else None

    def call(self, *, system: str, user: str, tool: dict, max_tokens: int) -> ToolCallResult:
        instructions = (
            f"{system}\n\nAnswer with one JSON object: the input of the "
            f"{tool['name']!r} tool. {tool['description']}"
        )
        response = self._post("/api/chat", {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": user},
            ],
            "format": tool["input_schema"],
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": min(max_tokens, self.max_predict or max_tokens),
                "num_ctx": self.num_ctx,
            },
        })
        # Mapped to the spelling the extractor's retry checks for.
        done = response.get("done_reason") or ""
        stop_reason = "max_tokens" if done == "length" else done
        content = (response.get("message") or {}).get("content") or ""
        try:
            tool_input = json.loads(content)
        except json.JSONDecodeError:
            if stop_reason != "max_tokens":
                raise LLMResponseError(
                    f"Expected a JSON {tool['name']!r} input, got unparseable text "
                    f"(done_reason={done!r})"
                ) from None
            tool_input = {}
        if not isinstance(tool_input, dict):
            raise LLMResponseError(f"Expected a JSON object for {tool['name']!r}")
        return ToolCallResult(
            tool_input=tool_input,
            stop_reason=stop_reason,
            input_tokens=response.get("prompt_eval_count") or 0,
            output_tokens=response.get("eval_count") or 0,
            model=response.get("model") or self.model,
        )

    def _post(self, path: str, body: dict) -> dict:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            self.host + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_S) as reply:
                return json.loads(reply.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise OllamaError(f"Ollama answered HTTP {exc.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.host} ({exc}). Is the Ollama app running?"
            ) from None


def provider() -> str:
    """The provider named by LLM_PROVIDER (default "gemini")."""
    name = (os.environ.get("LLM_PROVIDER") or "gemini").strip().lower()
    if name not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {PROVIDERS}, got {name!r}")
    return name


def default_caller(model: str | None = None) -> ToolCaller:
    """The real caller for LLM_PROVIDER; raises MissingAPIKeyError without its key."""
    if provider() == "ollama":
        return OllamaToolCaller(model=model)
    return GeminiToolCaller(model=model)


def api_errors() -> tuple[type[Exception], ...]:
    """The SDK errors that mean a live call failed, for the chosen provider."""
    if provider() == "ollama":
        return (OllamaError,)
    from google.genai import errors

    return (errors.APIError,)


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
                f"RecordingToolCaller(GeminiToolCaller(), {where!r}) "
                "(needs GEMINI_API_KEY)."
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
