"""Skill match results and the cache contract between JM-18 and JM-19.

This module holds only data types and the cache interface, so JM-19's Postgres
cache can implement ``SkillMatchCache`` without importing matcher logic. Keep it
small and stable: branches can't be merged before consolidation day, so this
file is the whole agreement between the two tickets.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field

# index_meta keys whose change can change a match: the taxonomy, the embedding
# model, the build itself (built_at changes on every rebuild) and the threshold.
VERSIONED_META_KEYS = ("taxonomy", "model", "build", "thresholds")
INDEX_VERSION_CHARS = 16


class MatchTier(str, Enum):
    EXACT = "exact"
    VECTOR = "vector"
    LLM = "llm"
    UNMATCHED = "unmatched"


class MatchCandidate(BaseModel):
    concept_id: str
    preferred_label: str
    matched_alias: str
    distance: float  # cosine distance, 0 = identical; 0.0 for exact alias hits


class SkillMatch(BaseModel):
    """One raw skill string and the concept it was mapped to (or None)."""

    raw: str
    normalized: str
    concept_id: str | None = None
    preferred_label: str | None = None
    tier: MatchTier
    distance: float | None = None
    low_confidence: bool = False
    candidates: list[MatchCandidate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)  # codes only, never skill text


@dataclass(frozen=True)
class MatcherVersions:
    index: str  # index_version(index.meta)
    prompt: str  # MATCHER_PROMPT_VERSION


def index_version(meta: Mapping[str, object]) -> str:
    """Short, stable fingerprint of what the index was built and measured from."""
    payload = {key: meta.get(key) for key in VERSIONED_META_KEYS}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:INDEX_VERSION_CHARS]


def cache_key(normalized: str, versions: MatcherVersions) -> str:
    """``<index version>|<prompt version>|<normalized skill>``.

    ``normalized`` must come from ``normalization.text.normalize()``. The versions
    come first, so a new taxonomy, index or prompt never serves a stale result.
    """
    return f"{versions.index}|{versions.prompt}|{normalized}"


class SkillMatchCache(Protocol):
    def get(self, key: str) -> SkillMatch | None: ...

    def set(self, key: str, match: SkillMatch) -> None: ...


class InMemorySkillMatchCache:
    """Process-local default; JM-19 provides the persistent one."""

    def __init__(self) -> None:
        self._store: dict[str, SkillMatch] = {}

    def get(self, key: str) -> SkillMatch | None:
        match = self._store.get(key)
        return match.model_copy(deep=True) if match is not None else None

    def set(self, key: str, match: SkillMatch) -> None:
        self._store[key] = match.model_copy(deep=True)

    def __len__(self) -> int:
        return len(self._store)
