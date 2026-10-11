import json
import logging

import psycopg
import pytest

from jobpdf.extraction.llm import FakeToolCaller, MissingAPIKeyError, ToolCallResult
from jobpdf.normalization import match_prompt
from jobpdf.normalization.index import Candidate
from jobpdf.normalization.match_cache import InMemorySkillMatchCache, MatchTier
from jobpdf.normalization.matcher import (
    AMBIGUOUS_ALIAS,
    EMPTY_SKILL,
    INDEX_UNAVAILABLE,
    LLM_CANDIDATES,
    LLM_CHOICE_NOT_IN_CANDIDATES,
    LLM_INVALID_OUTPUT,
    LLM_ITEM_MISSING,
    LLM_UNAVAILABLE,
    MAX_LLM_BATCH,
    NO_CANDIDATES,
    NO_VECTOR_THRESHOLD,
    TOP_K,
    SkillMatcher,
)
from jobpdf.normalization.text import normalize

T = MatchTier
THRESHOLD = 0.12
META = {
    "model": {"name": "intfloat/multilingual-e5-small", "revision": "abc"},
    "build": {"built_at": "2026-10-01T10:00:00+00:00"},
    "taxonomy": {"manifest_sha256": "f00", "esco_version": "v1.2.1"},
    "thresholds": {"auto_accept_distance": THRESHOLD},
}
BIGQUERY = "jobpdf:skill/google-bigquery"
TEAMWORK = "esco:teamwork"


def cand(cid: str, distance: float = 0.0, name: str | None = None, alias: str = "a") -> Candidate:
    return Candidate(canonical_id=cid, canonical_name=name or cid.upper(), matched_alias=alias,
                     distance=distance)


class FakeIndex:
    """Scripted SkillIndex: exact hits and search results keyed by normalized text."""

    def __init__(self, exact=None, vectors=None, meta=None, fail: str | None = None) -> None:
        self.exact = exact or {}
        self.vectors = vectors or {}
        self.meta = META if meta is None else meta
        self.fail = fail
        self.exact_calls: list[str] = []
        self.search_calls: list[tuple[list[str], int]] = []

    def lookup_exact(self, raw: str) -> list[Candidate]:
        self.exact_calls.append(raw)
        if self.fail == "exact":
            raise psycopg.OperationalError("database down")
        return list(self.exact.get(normalize(raw), []))

    def search(self, raws: list[str], k: int = 10) -> list[list[Candidate]]:
        self.search_calls.append((list(raws), k))
        if self.fail == "search":
            raise psycopg.OperationalError("database down")
        return [list(self.vectors.get(raw, []))[:k] for raw in raws]


def decisions(*pairs: tuple[int, str | None], stop_reason: str = "tool_use") -> ToolCallResult:
    return ToolCallResult(
        tool_input={"decisions": [{"item_id": i, "concept_id": c} for i, c in pairs]},
        stop_reason=stop_reason, input_tokens=100, output_tokens=20, model="fake-model",
    )


def sent_items(caller: FakeToolCaller, call: int = 0) -> list[dict]:
    return json.loads(caller.calls[call]["user"])["items"]


# --- exact tier ------------------------------------------------------------------


def test_exact_single_hit() -> None:
    index = FakeIndex(exact={"gbq": [cand(BIGQUERY, name="Google BigQuery", alias="gbq")]})

    (m,) = SkillMatcher(index).match_many(["GBQ"])

    assert (m.raw, m.normalized, m.concept_id, m.preferred_label) == (
        "GBQ", "gbq", BIGQUERY, "Google BigQuery")
    assert (m.tier, m.distance, m.low_confidence, m.warnings) == (T.EXACT, 0.0, False, [])
    assert index.search_calls == []


def test_ukrainian_alias_resolves_exactly() -> None:
    index = FakeIndex(exact={"працювати в команді": [cand(TEAMWORK, name="work in teams")]})

    m = SkillMatcher(index).match("Працювати в команді")

    assert (m.tier, m.concept_id, m.preferred_label) == (T.EXACT, TEAMWORK, "work in teams")


def test_short_skill_with_an_exact_only_alias_is_exact() -> None:
    index = FakeIndex(exact={"r": [cand("esco:r")]})

    assert SkillMatcher(index).match("R").tier is T.EXACT


def test_ambiguous_alias_goes_to_the_llm_with_exactly_its_candidates() -> None:
    ids = [f"esco:extrusion-{i}" for i in range(LLM_CANDIDATES + 2)]
    index = FakeIndex(exact={"guard cold extrusion": [cand(i) for i in ids]})
    caller = FakeToolCaller([decisions((1, ids[1]))])

    m = SkillMatcher(index, caller).match("guard cold extrusion")

    assert (m.tier, m.concept_id, m.warnings) == (T.LLM, ids[1], [AMBIGUOUS_ALIAS])
    assert index.search_calls == []
    (item,) = sent_items(caller)
    assert [c["concept_id"] for c in item["candidates"]] == ids[:LLM_CANDIDATES]


# --- vector tier -----------------------------------------------------------------


@pytest.mark.parametrize("distance", [THRESHOLD - 1e-6, THRESHOLD])
def test_vector_accepts_at_or_below_the_threshold(distance: float) -> None:
    index = FakeIndex(vectors={"big query": [cand(BIGQUERY, distance), cand("esco:x", 0.3)]})

    (m,) = SkillMatcher(index).match_many(["Big Query"])

    assert (m.tier, m.concept_id, m.distance) == (T.VECTOR, BIGQUERY, distance)
    assert [c.concept_id for c in m.candidates] == [BIGQUERY, "esco:x"]
    assert index.search_calls == [(["big query"], TOP_K)]


def test_vector_just_outside_the_threshold_goes_to_the_llm() -> None:
    index = FakeIndex(vectors={"big query": [cand(BIGQUERY, THRESHOLD + 1e-6)]})
    caller = FakeToolCaller([decisions((1, BIGQUERY))])

    m = SkillMatcher(index, caller).match("Big Query")

    assert (m.tier, m.concept_id, m.distance) == (T.LLM, BIGQUERY, THRESHOLD + 1e-6)
    assert len(caller.calls) == 1


def test_threshold_is_read_from_index_meta() -> None:
    meta = {**META, "thresholds": {"auto_accept_distance": 0.5}}
    index = FakeIndex(vectors={"big query": [cand(BIGQUERY, 0.4)]}, meta=meta)

    assert SkillMatcher(index).match("Big Query").tier is T.VECTOR


def test_missing_threshold_never_auto_accepts() -> None:
    meta = {k: v for k, v in META.items() if k != "thresholds"}
    index = FakeIndex(vectors={"big query": [cand(BIGQUERY, 0.0001)]}, meta=meta)

    m = SkillMatcher(index).match("Big Query")

    assert m.tier is T.UNMATCHED
    assert m.warnings == [NO_VECTOR_THRESHOLD, LLM_UNAVAILABLE]
    assert [c.concept_id for c in m.candidates] == [BIGQUERY]


def test_short_skill_without_exact_hit_still_uses_vector_and_llm() -> None:
    index = FakeIndex(vectors={"go": [cand("esco:go-lang", 0.4)]})
    caller = FakeToolCaller([decisions((1, "esco:go-lang"))])

    m = SkillMatcher(index, caller).match("Go")

    assert index.search_calls == [(["go"], TOP_K)]
    assert (m.tier, m.concept_id) == (T.LLM, "esco:go-lang")


def test_no_candidates_at_all_is_unmatched_without_an_llm_call() -> None:
    caller = FakeToolCaller([])

    m = SkillMatcher(FakeIndex(), caller).match("Zxqv")

    assert (m.tier, m.warnings, m.low_confidence) == (T.UNMATCHED, [NO_CANDIDATES], True)
    assert caller.calls == []


# --- LLM tier --------------------------------------------------------------------


def undecided_index(*norms: str) -> FakeIndex:
    """Each skill gets two vector candidates above the threshold."""
    return FakeIndex(vectors={
        n: [cand(f"esco:{n}-a", 0.2), cand(f"esco:{n}-b", 0.25)] for n in norms
    })


def test_llm_pick_reject_and_one_call_for_the_whole_batch() -> None:
    cache = InMemorySkillMatchCache()
    index = undecided_index("alpha", "beta", "gamma")
    caller = FakeToolCaller([decisions((1, "esco:alpha-b"), (2, None), (3, "esco:gamma-a"))])

    alpha, beta, gamma = SkillMatcher(index, caller, cache).match_many(
        ["Alpha", "Beta", "Gamma"], context="CV skills section")

    assert len(caller.calls) == 1
    assert caller.calls[0]["tool"]["name"] == match_prompt.MATCHER_TOOL_NAME
    assert [i["skill"] for i in sent_items(caller)] == ["Alpha", "Beta", "Gamma"]
    assert json.loads(caller.calls[0]["user"])["context"] == "CV skills section"
    assert (alpha.tier, alpha.concept_id, alpha.distance) == (T.LLM, "esco:alpha-b", 0.25)
    assert (beta.tier, beta.concept_id, beta.low_confidence) == (T.UNMATCHED, None, True)
    assert len(beta.candidates) == 2  # kept for the report
    assert gamma.concept_id == "esco:gamma-a"
    assert len(cache) == 3  # the explicit reject-all is cached too


def test_llm_choice_outside_the_candidates_is_a_rejection_with_a_warning() -> None:
    cache = InMemorySkillMatchCache()
    caller = FakeToolCaller([decisions((1, "esco:invented"))])

    m = SkillMatcher(undecided_index("alpha"), caller, cache).match("Alpha")

    assert (m.tier, m.concept_id, m.warnings) == (
        T.UNMATCHED, None, [LLM_CHOICE_NOT_IN_CANDIDATES])
    assert len(cache) == 0


def test_item_missing_from_llm_output_is_unmatched_and_unknown_ids_are_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caller = FakeToolCaller([decisions((1, "esco:alpha-a"), (7, "esco:beta-a"),
                                       (1, "esco:alpha-b"))])

    with caplog.at_level(logging.WARNING):
        alpha, beta = SkillMatcher(undecided_index("alpha", "beta"), caller).match_many(
            ["Alpha", "Beta"])

    assert alpha.concept_id == "esco:alpha-a"  # the first decision for an id wins
    assert (beta.tier, beta.warnings) == (T.UNMATCHED, [LLM_ITEM_MISSING])
    assert "llm_unknown_item: 2 LLM decisions ignored" in caplog.text


@pytest.mark.parametrize(
    "response",
    [
        ToolCallResult(tool_input={"decisions": [{"item_id": "x"}]}, stop_reason="tool_use",
                       input_tokens=1, output_tokens=1, model="fake-model"),
        decisions((1, "esco:alpha-a"), stop_reason="max_tokens"),
    ],
    ids=["invalid_schema", "max_tokens"],
)
def test_invalid_llm_output_leaves_the_chunk_unmatched(response: ToolCallResult) -> None:
    cache = InMemorySkillMatchCache()

    m = SkillMatcher(undecided_index("alpha"), FakeToolCaller([response]), cache).match("Alpha")

    assert (m.tier, m.warnings) == (T.UNMATCHED, [LLM_INVALID_OUTPUT])
    assert len(cache) == 0


def test_batches_above_the_limit_are_chunked() -> None:
    names = [f"skill{i:03d}" for i in range(2 * MAX_LLM_BATCH + 5)]
    sizes = [MAX_LLM_BATCH, MAX_LLM_BATCH, 5]
    caller = FakeToolCaller([decisions(*((i, None) for i in range(1, n + 1))) for n in sizes])

    results = SkillMatcher(undecided_index(*names), caller).match_many(names)

    assert [len(sent_items(caller, c)) for c in range(3)] == sizes
    assert [r.raw for r in results] == names
    assert all(r.tier is T.UNMATCHED and r.warnings == [] for r in results)


@pytest.mark.parametrize(
    "caller",
    [None, FakeToolCaller([MissingAPIKeyError("no key")]),
     FakeToolCaller([RuntimeError("API outage")])],
    ids=["no_caller", "missing_key", "api_error"],
)
def test_llm_unavailable_degrades_and_caches_nothing(caller) -> None:
    cache = InMemorySkillMatchCache()
    index = undecided_index("alpha")
    index.exact["gbq"] = [cand(BIGQUERY)]

    gbq, alpha = SkillMatcher(index, caller, cache).match_many(["GBQ", "Alpha"])

    assert gbq.tier is T.EXACT  # tiers 1-3 still run
    assert (alpha.tier, alpha.warnings, alpha.low_confidence) == (
        T.UNMATCHED, [LLM_UNAVAILABLE], True)
    assert [c.concept_id for c in alpha.candidates] == ["esco:alpha-a", "esco:alpha-b"]
    assert len(cache) == 1  # only the exact hit


# --- cache -----------------------------------------------------------------------


def test_cache_hit_skips_every_tier_and_keeps_the_current_raw() -> None:
    cache = InMemorySkillMatchCache()
    SkillMatcher(FakeIndex(exact={"gbq": [cand(BIGQUERY)]}), cache=cache).match("GBQ")
    index = FakeIndex()
    caller = FakeToolCaller([])

    m = SkillMatcher(index, caller, cache).match("  gbq ")

    assert (m.raw, m.concept_id, m.tier) == ("  gbq ", BIGQUERY, T.EXACT)
    assert index.exact_calls == [] and index.search_calls == [] and caller.calls == []


def test_new_index_version_misses_the_cache() -> None:
    cache = InMemorySkillMatchCache()
    SkillMatcher(FakeIndex(exact={"gbq": [cand(BIGQUERY)]}), cache=cache).match("GBQ")
    rebuilt = FakeIndex(exact={"gbq": [cand(BIGQUERY)]},
                        meta={**META, "build": {"built_at": "2026-10-11T10:00:00+00:00"}})

    SkillMatcher(rebuilt, cache=cache).match("GBQ")

    assert rebuilt.exact_calls == ["gbq"]


def test_new_prompt_version_misses_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = InMemorySkillMatchCache()
    SkillMatcher(FakeIndex(exact={"gbq": [cand(BIGQUERY)]}), cache=cache).match("GBQ")
    monkeypatch.setattr(match_prompt, "MATCHER_PROMPT_VERSION", "2")
    index = FakeIndex(exact={"gbq": [cand(BIGQUERY)]})

    SkillMatcher(index, cache=cache).match("GBQ")

    assert index.exact_calls == ["gbq"]


class BrokenCache:
    def get(self, key):
        raise ConnectionError("cache down")

    def set(self, key, match):
        raise ConnectionError("cache down")


def test_a_broken_cache_does_not_break_matching() -> None:
    m = SkillMatcher(FakeIndex(exact={"gbq": [cand(BIGQUERY)]}), cache=BrokenCache()).match("GBQ")

    assert m.tier is T.EXACT


# --- inputs, ordering, degradation -------------------------------------------------


def test_duplicates_are_matched_once_and_order_and_length_are_kept() -> None:
    index = FakeIndex(exact={"gbq": [cand(BIGQUERY)], "docker": [cand("esco:docker")]})
    raws = ["GBQ", "Docker", "gbq", "", " GBQ ", "Zxqv"]

    results = SkillMatcher(index).match_many(raws)

    assert [r.raw for r in results] == raws
    assert [r.tier for r in results] == [T.EXACT, T.EXACT, T.EXACT, T.UNMATCHED, T.EXACT,
                                         T.UNMATCHED]
    assert index.exact_calls == ["gbq", "docker", "zxqv"]
    assert index.search_calls == [(["zxqv"], TOP_K)]


@pytest.mark.parametrize("raw", ["", "   ", "\n\t"])
def test_empty_and_whitespace_skills(raw: str) -> None:
    index = FakeIndex()

    (m,) = SkillMatcher(index).match_many([raw])

    assert (m.raw, m.tier, m.warnings, m.low_confidence) == (raw, T.UNMATCHED, [EMPTY_SKILL], True)
    assert index.exact_calls == [] and index.search_calls == []


def test_empty_input_list() -> None:
    assert SkillMatcher(FakeIndex()).match_many([]) == []


@pytest.mark.parametrize("fail", ["exact", "search"])
def test_database_error_degrades_to_index_unavailable(fail: str) -> None:
    cache = InMemorySkillMatchCache()

    m = SkillMatcher(FakeIndex(fail=fail), FakeToolCaller([]), cache).match("Python")

    assert (m.tier, m.warnings) == (T.UNMATCHED, [INDEX_UNAVAILABLE])
    assert len(cache) == 0


def test_logs_carry_counts_but_no_skill_text(caplog: pytest.LogCaptureFixture) -> None:
    index = FakeIndex(exact={"gbq": [cand(BIGQUERY)]})

    with caplog.at_level(logging.DEBUG, logger="jobpdf.normalization.matcher"):
        SkillMatcher(index).match_many(["GBQ", "Secretskillxyz"])

    assert "exact=1" in caplog.text and "unmatched=1" in caplog.text
    assert "secretskillxyz" not in caplog.text.lower() and "gbq" not in caplog.text.lower()
