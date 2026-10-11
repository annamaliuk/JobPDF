"""Map raw skill strings to taxonomy concepts (JM-18).

Tiers per unique normalized string, cheapest and most certain first:
cache -> exact alias -> vector search below the measured threshold -> one
batched LLM call that picks a candidate or rejects them all -> unmatched.
The raw string is always kept; nothing is dropped and nothing raises on bad
input or a missing LLM, database or cache.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import psycopg
from pydantic import ValidationError

from jobpdf.extraction.llm import ToolCaller
from jobpdf.normalization import match_prompt
from jobpdf.normalization.index import Candidate
from jobpdf.normalization.match_cache import (
    InMemorySkillMatchCache,
    MatchCandidate,
    MatcherVersions,
    MatchTier,
    SkillMatch,
    SkillMatchCache,
    cache_key,
    index_version,
)
from jobpdf.normalization.text import normalize

log = logging.getLogger(__name__)

TOP_K = 5  # vector candidates fetched per skill
MAX_LLM_BATCH = 30  # skills per LLM call; a typical CV fits in one
LLM_CANDIDATES = 5  # candidates shown to the LLM per skill
LLM_MAX_TOKENS = 4096

# Warning codes (never skill text: warnings may reach logs).
EMPTY_SKILL = "empty_skill"
AMBIGUOUS_ALIAS = "ambiguous_alias"
NO_VECTOR_THRESHOLD = "no_vector_threshold"
NO_CANDIDATES = "no_candidates"
LLM_UNAVAILABLE = "llm_unavailable"
INDEX_UNAVAILABLE = "index_unavailable"
LLM_CHOICE_NOT_IN_CANDIDATES = "llm_choice_not_in_candidates"
LLM_ITEM_MISSING = "llm_item_missing"
LLM_INVALID_OUTPUT = "llm_invalid_output"
LLM_UNKNOWN_ITEM = "llm_unknown_item"  # logged only: there is no item to attach it to

# A result carrying any of these is not cached, so it is retried next time.
# Everything else is: exact, vector, llm, and an LLM's explicit reject-all.
UNCACHEABLE_WARNINGS = frozenset({
    EMPTY_SKILL, NO_CANDIDATES, LLM_UNAVAILABLE, INDEX_UNAVAILABLE,
    LLM_CHOICE_NOT_IN_CANDIDATES, LLM_ITEM_MISSING, LLM_INVALID_OUTPUT,
})


class SkillIndexLike(Protocol):
    """The part of JM-17's SkillIndex the matcher uses (a fake satisfies it in tests)."""

    @property
    def meta(self) -> Mapping[str, object]: ...

    def lookup_exact(self, raw: str) -> list[Candidate]: ...

    def search(self, raws: list[str], k: int = 10) -> list[list[Candidate]]: ...


@dataclass
class _Pending:
    """One unique normalized skill on its way through the tiers."""

    normalized: str
    raw: str  # first spelling seen; each input position gets its own raw back
    candidates: list[MatchCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    result: SkillMatch | None = None


class SkillMatcher:
    def __init__(
        self,
        index: SkillIndexLike,
        tool_caller: ToolCaller | None = None,
        cache: SkillMatchCache | None = None,
    ) -> None:
        self._index = index
        self._caller = tool_caller
        self._cache: SkillMatchCache = cache if cache is not None else InMemorySkillMatchCache()
        meta = index.meta or {}
        self._versions = MatcherVersions(
            index=index_version(meta), prompt=match_prompt.MATCHER_PROMPT_VERSION
        )
        # Read at runtime from index_meta: it was measured on this exact index.
        self._threshold = auto_accept_distance(meta)
        if self._threshold is None:
            log.warning("%s: index_meta has no thresholds.auto_accept_distance, so the "
                        "vector tier only collects candidates (run scripts/eval_index.py)",
                        NO_VECTOR_THRESHOLD)

    def match(self, raw: str, context: str | None = None) -> SkillMatch:
        return self.match_many([raw], context)[0]

    def match_many(self, raws: Sequence[str], context: str | None = None) -> list[SkillMatch]:
        """Exactly one result per input, in input order; duplicates are matched once.

        ``context`` is a short hint for the LLM such as "CV skills section"; it
        must never carry personal data.
        """
        started = time.perf_counter()
        stats: Counter[str] = Counter()
        normalized = [normalize(raw) for raw in raws]
        pending: dict[str, _Pending] = {}
        for raw, norm in zip(raws, normalized, strict=True):
            if norm and norm not in pending:
                pending[norm] = _Pending(normalized=norm, raw=raw)

        todo = self._from_cache(list(pending.values()), stats)
        to_vector, ambiguous = self._exact(todo, stats)
        undecided = self._vector(to_vector, stats)
        self._llm(ambiguous + undecided, context, stats)

        results: list[SkillMatch] = []
        for raw, norm in zip(raws, normalized, strict=True):
            if not norm:
                results.append(_unmatched(raw, norm, [], [EMPTY_SKILL]))
                continue
            match = pending[norm].result
            if match is None:  # unreachable: every tier path finishes its items
                match = _unmatched(raw, norm, [], [LLM_UNAVAILABLE])
            results.append(match.model_copy(update={"raw": raw}, deep=True))

        log.info(
            "matched %d skills (%d unique): cache=%d exact=%d vector=%d llm=%d unmatched=%d "
            "llm_calls=%d in %.0f ms",
            len(raws), len(pending), stats["cache"], stats[MatchTier.EXACT.value],
            stats[MatchTier.VECTOR.value], stats[MatchTier.LLM.value],
            stats[MatchTier.UNMATCHED.value], stats["llm_calls"],
            (time.perf_counter() - started) * 1000,
        )
        return results

    # --- tiers -------------------------------------------------------------------

    def _from_cache(self, items: list[_Pending], stats: Counter[str]) -> list[_Pending]:
        misses = []
        for item in items:
            try:
                hit = self._cache.get(self._key(item))
            except Exception as exc:  # a broken cache must never break matching
                log.warning("skill match cache get failed (%s); matching without it",
                            type(exc).__name__)
                hit = None
            if hit is None:
                misses.append(item)
            else:
                item.result = hit
                stats["cache"] += 1
        return misses

    def _exact(
        self, items: list[_Pending], stats: Counter[str]
    ) -> tuple[list[_Pending], list[_Pending]]:
        to_vector: list[_Pending] = []
        ambiguous: list[_Pending] = []
        for item in items:
            try:
                hits = self._index.lookup_exact(item.normalized)
            except psycopg.Error as exc:
                log.warning("%s: exact lookup failed (%s)", INDEX_UNAVAILABLE, type(exc).__name__)
                self._finish(item, _unmatched(item.raw, item.normalized, [],
                                              [INDEX_UNAVAILABLE]), stats)
                continue
            if len(hits) == 1:
                self._finish(item, _accepted(item, hits[0], MatchTier.EXACT, [_cand(hits[0])]),
                             stats)
            elif hits:
                # An alias naming several concepts: the LLM decides among exactly these.
                item.candidates = [_cand(c) for c in hits[:LLM_CANDIDATES]]
                item.warnings.append(AMBIGUOUS_ALIAS)
                ambiguous.append(item)
            else:
                to_vector.append(item)
        return to_vector, ambiguous

    def _vector(self, items: list[_Pending], stats: Counter[str]) -> list[_Pending]:
        if not items:
            return []
        try:
            # One call: SkillIndex.search embeds the whole batch at once.
            found = self._index.search([item.normalized for item in items], k=TOP_K)
        except (psycopg.Error, OSError) as exc:
            log.warning("%s: vector search failed (%s)", INDEX_UNAVAILABLE, type(exc).__name__)
            for item in items:
                self._finish(item, _unmatched(item.raw, item.normalized, [],
                                              [INDEX_UNAVAILABLE]), stats)
            return []
        undecided = []
        for item, candidates in zip(items, found, strict=True):
            if not candidates:
                self._finish(item, _unmatched(item.raw, item.normalized, [],
                                              [*item.warnings, NO_CANDIDATES]), stats)
                continue
            top = candidates[0]
            if self._threshold is not None and top.distance <= self._threshold:
                self._finish(item, _accepted(item, top, MatchTier.VECTOR,
                                             [_cand(c) for c in candidates]), stats)
                continue
            item.candidates = [_cand(c) for c in candidates[:LLM_CANDIDATES]]
            if self._threshold is None:
                item.warnings.append(NO_VECTOR_THRESHOLD)
            undecided.append(item)
        return undecided

    def _llm(self, items: list[_Pending], context: str | None, stats: Counter[str]) -> None:
        if not items:
            return
        if self._caller is None:
            self._give_up(items, LLM_UNAVAILABLE, stats)
            return
        for start in range(0, len(items), MAX_LLM_BATCH):
            self._llm_chunk(items[start : start + MAX_LLM_BATCH], context, stats)

    def _llm_chunk(self, chunk: list[_Pending], context: str | None, stats: Counter[str]) -> None:
        by_id = {i: item for i, item in enumerate(chunk, start=1)}
        prompt_items = [
            match_prompt.PromptItem(
                item_id=i,
                skill=item.raw.strip(),
                candidates=[match_prompt.PromptCandidate(c.concept_id, c.preferred_label,
                                                         c.matched_alias)
                            for c in item.candidates],
            )
            for i, item in by_id.items()
        ]
        try:
            response = self._caller.call(  # type: ignore[union-attr]
                system=match_prompt.SYSTEM_PROMPT,
                user=match_prompt.build_user_message(prompt_items, context),
                tool=match_prompt.matcher_tool(),
                max_tokens=LLM_MAX_TOKENS,
            )
        except Exception as exc:  # missing key, API outage, no tool call: degrade, never crash
            log.warning("%s: LLM call failed (%s)", LLM_UNAVAILABLE, type(exc).__name__)
            self._give_up(chunk, LLM_UNAVAILABLE, stats)
            return
        stats["llm_calls"] += 1
        if response.stop_reason == "max_tokens":
            self._give_up(chunk, LLM_INVALID_OUTPUT, stats)
            return
        try:
            decisions = match_prompt.LLMDecisions.model_validate(response.tool_input).decisions
        except ValidationError:
            log.warning("%s: LLM output failed validation", LLM_INVALID_OUTPUT)
            self._give_up(chunk, LLM_INVALID_OUTPUT, stats)
            return

        chosen: dict[int, str | None] = {}
        unknown = 0
        for decision in decisions:
            if decision.item_id not in by_id or decision.item_id in chosen:
                unknown += 1
                continue
            chosen[decision.item_id] = decision.concept_id
        if unknown:
            log.warning("%s: %d LLM decisions ignored", LLM_UNKNOWN_ITEM, unknown)

        for item_id, item in by_id.items():
            if item_id not in chosen:
                self._give_up([item], LLM_ITEM_MISSING, stats)
                continue
            concept_id = chosen[item_id]
            pick = next((c for c in item.candidates if c.concept_id == concept_id), None)
            if concept_id is None:
                # An explicit reject-all: a real answer, so it is cached.
                self._finish(item, _unmatched(item.raw, item.normalized, item.candidates,
                                              item.warnings), stats)
            elif pick is None:
                self._give_up([item], LLM_CHOICE_NOT_IN_CANDIDATES, stats)
            else:
                self._finish(item, SkillMatch(
                    raw=item.raw, normalized=item.normalized, concept_id=pick.concept_id,
                    preferred_label=pick.preferred_label, tier=MatchTier.LLM,
                    distance=pick.distance, candidates=item.candidates,
                    warnings=list(item.warnings),
                ), stats)

    # --- helpers -----------------------------------------------------------------

    def _give_up(self, items: list[_Pending], code: str, stats: Counter[str]) -> None:
        for item in items:
            self._finish(item, _unmatched(item.raw, item.normalized, item.candidates,
                                          [*item.warnings, code]), stats)

    def _finish(self, item: _Pending, match: SkillMatch, stats: Counter[str]) -> None:
        item.result = match
        stats[match.tier.value] += 1
        if UNCACHEABLE_WARNINGS.intersection(match.warnings):
            return
        try:
            self._cache.set(self._key(item), match)
        except Exception as exc:
            log.warning("skill match cache set failed (%s)", type(exc).__name__)

    def _key(self, item: _Pending) -> str:
        return cache_key(item.normalized, self._versions)


def auto_accept_distance(meta: Mapping[str, object]) -> float | None:
    """index_meta['thresholds']['auto_accept_distance'] (cosine distance; accept if <=)."""
    thresholds = meta.get("thresholds")
    value = thresholds.get("auto_accept_distance") if isinstance(thresholds, Mapping) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _cand(candidate: Candidate) -> MatchCandidate:
    return MatchCandidate(
        concept_id=candidate.canonical_id, preferred_label=candidate.canonical_name,
        matched_alias=candidate.matched_alias, distance=candidate.distance,
    )


def _accepted(
    item: _Pending, top: Candidate, tier: MatchTier, candidates: list[MatchCandidate]
) -> SkillMatch:
    return SkillMatch(
        raw=item.raw, normalized=item.normalized, concept_id=top.canonical_id,
        preferred_label=top.canonical_name, tier=tier, distance=top.distance,
        candidates=candidates, warnings=list(item.warnings),
    )


def _unmatched(
    raw: str, normalized: str, candidates: list[MatchCandidate], warnings: list[str]
) -> SkillMatch:
    return SkillMatch(
        raw=raw, normalized=normalized, tier=MatchTier.UNMATCHED, low_confidence=True,
        candidates=list(candidates), warnings=list(warnings),
    )
