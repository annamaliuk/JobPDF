"""Prompt and output schema for the matcher's LLM tier (JM-18).

The LLM only chooses among candidates the index already found, or rejects them
all; it never invents a concept. One call covers a whole batch of skills.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jobpdf.extraction.prompt import mask_pii
from jobpdf.extraction.schema import tool_schema

# Bump on every change to SYSTEM_PROMPT, the user-message format or the tool:
# cached matches and replay fixtures are keyed on it.
MATCHER_PROMPT_VERSION = "1"
MATCHER_TOOL_NAME = "record_skill_matches"
MAX_CONTEXT_CHARS = 100

SYSTEM_PROMPT = """\
You map raw skill names, taken from CVs and job vacancies, to concepts of a skills
taxonomy, and record your decisions by calling the record_skill_matches tool exactly once.

The user message is JSON. Each item has an item_id, the raw skill as written, and
candidate concepts found by search, each with a concept_id, its label and the
alias that matched.

Rules:
1. For each item, choose the candidate that names the SAME skill as the raw text:
   a synonym, abbreviation, acronym, translation, spelling variant, or a specific
   version of it ("Python 3" -> Python, "GBQ" -> Google BigQuery).
2. If no candidate names the same skill, answer null. A candidate that is only
   related, broader or narrower is not the same skill: answer null.
3. concept_id must be copied exactly from that item's own candidates.
4. Answer every item exactly once, using its item_id.
5. Skills and labels may be in English or Ukrainian; compare meanings, not languages.
6. The optional context only says where the skills come from (e.g. "CV skills
   section"); use it to tell apart skills that share a name.
"""


class _LLMModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMDecision(_LLMModel):
    item_id: int
    concept_id: str | None = Field(
        description="The chosen candidate's concept_id, or null if none is the same skill."
    )


class LLMDecisions(_LLMModel):
    decisions: list[LLMDecision]


@dataclass(frozen=True)
class PromptCandidate:
    concept_id: str
    label: str
    matched_alias: str


@dataclass(frozen=True)
class PromptItem:
    item_id: int
    skill: str
    candidates: list[PromptCandidate]


def matcher_tool() -> dict[str, Any]:
    return {
        "name": MATCHER_TOOL_NAME,
        "description": (
            "Record, for every item, the chosen candidate concept_id or null. "
            "Call it exactly once."
        ),
        "input_schema": tool_schema(LLMDecisions)["input_schema"],
    }


def build_user_message(items: list[PromptItem], context: str | None) -> str:
    """The batch as JSON; skills and context are PII-masked as a defence in depth."""
    payload = {
        "context": mask_pii(context.strip()[:MAX_CONTEXT_CHARS])[0] if context else None,
        "items": [
            {
                "item_id": item.item_id,
                "skill": mask_pii(item.skill)[0],
                "candidates": [
                    {"concept_id": c.concept_id, "label": c.label,
                     "matched_alias": c.matched_alias}
                    for c in item.candidates
                ],
            }
            for item in items
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=1)
