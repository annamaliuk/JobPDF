"""Structured CV extraction with one forced tool-use call (JM-11).

parse() -> split_sections() -> extract_profile(): the whole CV (contact data
masked) goes to the LLM in one call; the tool input is validated with Pydantic,
corrected once if invalid, then completed by post-processing. Results are cached
on disk so an unchanged CV + prompt + model is never sent twice.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from jobpdf.extraction import prompt as prompt_module
from jobpdf.extraction.llm import AnthropicToolCaller, ToolCaller, ToolCallResult
from jobpdf.extraction.models import ParsedDocument, Section
from jobpdf.extraction.postprocess import build_profile
from jobpdf.extraction.schema import (
    SCHEMA_VERSION,
    ExtractedProfile,
    ExtractionResult,
    tool_schema,
)

MAX_TOKENS = 4096
MAX_TOKENS_RETRY = 8192
CACHE_DIR = Path("data/cache/extraction")
# The corrective retry lists at most this many validation errors.
MAX_ERROR_LINES = 20


class ExtractionError(RuntimeError):
    """Extraction failed even after its single retry; ``errors`` says why."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        self.errors = errors or []
        detail = "\n  ".join(self.errors)
        super().__init__(f"{message}\n  {detail}" if detail else message)


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0


def extract_profile(
    doc: ParsedDocument,
    sections: list[Section],
    caller: ToolCaller | None = None,
    *,
    use_cache: bool = True,
    cache_dir: Path = CACHE_DIR,
    today: dt.date | None = None,
) -> ExtractionResult:
    """Extract a validated CandidateProfile from a parsed, sectioned CV.

    ``caller=None`` means the real Anthropic API, which raises MissingAPIKeyError
    without ANTHROPIC_API_KEY. Tests pass a FakeToolCaller or ReplayToolCaller.
    """
    caller = caller if caller is not None else AnthropicToolCaller()
    system = prompt_module.SYSTEM_PROMPT
    user = prompt_module.build_user_message(doc, sections)
    tool = tool_schema()

    cache_path = Path(cache_dir) / f"{cache_key(system, user, caller.model)}.json"
    if use_cache and cache_path.is_file():
        cached = ExtractionResult.model_validate_json(cache_path.read_text(encoding="utf-8"))
        return cached.model_copy(update={"from_cache": True})

    usage = _Usage()
    response = _call(caller, system, user, tool, usage)
    try:
        extracted = ExtractedProfile.model_validate(response.tool_input)
    except ValidationError as first_error:
        corrective = user + _correction(response.tool_input, _concise(first_error))
        response = _call(caller, system, corrective, tool, usage)
        try:
            extracted = ExtractedProfile.model_validate(response.tool_input)
        except ValidationError as second_error:
            raise ExtractionError(
                "Model output failed schema validation twice", _concise(second_error)
            ) from second_error

    masked_text, _ = prompt_module.mask_pii(doc.full_text)
    profile, warnings = build_profile(extracted, masked_text, today or dt.date.today())
    result = ExtractionResult(
        profile=profile,
        warnings=warnings,
        model=response.model,
        prompt_version=prompt_module.PROMPT_VERSION,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        from_cache=False,
    )
    if use_cache:
        _write_cache(cache_path, result)
    return result


def cache_key(system: str, user: str, model: str) -> str:
    parts = [system, user, prompt_module.PROMPT_VERSION, SCHEMA_VERSION, model]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def _call(
    caller: ToolCaller, system: str, user: str, tool: dict, usage: _Usage
) -> ToolCallResult:
    """One logical call: retried once with a bigger budget if the output was cut off."""
    for max_tokens in (MAX_TOKENS, MAX_TOKENS_RETRY):
        response = caller.call(system=system, user=user, tool=tool, max_tokens=max_tokens)
        usage.input_tokens += response.input_tokens
        usage.output_tokens += response.output_tokens
        if response.stop_reason != "max_tokens":
            return response
    raise ExtractionError(
        f"Model output exceeded {MAX_TOKENS_RETRY} tokens even after retrying; "
        "the CV may be unusually long."
    )


def _concise(error: ValidationError) -> list[str]:
    lines = []
    for item in error.errors(include_url=False)[:MAX_ERROR_LINES]:
        location = ".".join(str(part) for part in item["loc"]) or "(root)"
        lines.append(f"{location}: {item['msg']}")
    return lines


def _correction(previous: dict, errors: list[str]) -> str:
    return (
        "\n\n<previous_output>\n"
        + json.dumps(previous, ensure_ascii=False, indent=1)
        + "\n</previous_output>\n<validation_errors>\n"
        + "\n".join(f"- {line}" for line in errors)
        + "\n</validation_errors>\n"
        "Your previous record_candidate_profile call did not match the schema. "
        "Call the tool again with the complete, corrected profile."
    )


def _write_cache(path: Path, result: ExtractionResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, path)  # a crash mid-write never leaves a half-written cache entry
