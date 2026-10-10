import json

import pytest
from pydantic import ValidationError

from jobpdf.normalization.match_prompt import (
    MATCHER_TOOL_NAME,
    MAX_CONTEXT_CHARS,
    LLMDecisions,
    PromptCandidate,
    PromptItem,
    build_user_message,
    matcher_tool,
)

ITEM = PromptItem(
    item_id=1, skill="GBQ",
    candidates=[PromptCandidate("jobpdf:skill/google-bigquery", "Google BigQuery", "gbq")],
)


def test_user_message_is_json_with_items_and_candidates() -> None:
    payload = json.loads(build_user_message([ITEM], "CV skills section"))

    assert payload == {
        "context": "CV skills section",
        "items": [{
            "item_id": 1, "skill": "GBQ",
            "candidates": [{"concept_id": "jobpdf:skill/google-bigquery",
                            "label": "Google BigQuery", "matched_alias": "gbq"}],
        }],
    }


def test_personal_data_is_masked_and_context_truncated() -> None:
    item = PromptItem(item_id=1, skill="see github.com/jane-doe", candidates=[])
    message = build_user_message([item], "jane.doe@example.com " + "x" * 200)

    payload = json.loads(message)
    assert "jane" not in message
    assert len(payload["context"]) == MAX_CONTEXT_CHARS


def test_tool_schema_and_strict_validation() -> None:
    tool = matcher_tool()

    assert tool["name"] == MATCHER_TOOL_NAME
    assert "$defs" not in json.dumps(tool)
    parsed = LLMDecisions.model_validate({"decisions": [{"item_id": 1, "concept_id": None}]})
    assert parsed.decisions[0].concept_id is None
    with pytest.raises(ValidationError):
        LLMDecisions.model_validate({"decisions": [{"item_id": 1}]})
