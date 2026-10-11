"""Evaluate the skill index and measure JM-18's auto-accept threshold (JM-17).

Usage (from the repo root, after build_index.py):
    uv run python scripts/eval_index.py
    uv run python scripts/eval_index.py --skip-prefix   # skip the slow 'passage: ' rebuild

Steps: held-out alt-alias recall, CV-phrase recall, distance percentiles, the
auto-accept threshold (stored in index_meta), HNSW vs exact search, and a
"query: " vs "passage: " document-prefix comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from build_index import run as build_table
from psycopg import sql

from jobpdf.normalization.db import database_url
from jobpdf.normalization.embedder import Embedder
from jobpdf.normalization.index import FETCH_FACTOR, SkillIndex, dedupe_candidates
from jobpdf.normalization.index_build import DEFAULT_LANGUAGES, DEFAULT_TABLE
from jobpdf.normalization.index_eval import (
    MIN_PRECISION,
    MIN_QUERIES_BELOW,
    Outcome,
    distance_percentiles,
    recall_at_k,
    select_threshold,
)
from jobpdf.normalization.text import normalize

SEED = 17
HELD_OUT_SIZE = 1000
EXACT_CHECK_SIZE = 200
K = 5
HIGHER_EF_SEARCH = 200
PASSAGE_TABLE = "taxonomy_alias_passage"
CV_PHRASES = Path("tests/normalization/fixtures/cv_skill_phrases.csv")


def sample_held_out(index: SkillIndex) -> list[tuple[str, str]]:
    """(alias, canonical_id) of en alt aliases; sampled independently of row ids,
    so the same sample is drawn from any table built from the same taxonomy."""
    rows = index.connection.execute(
        sql.SQL(
            "SELECT alias, canonical_id FROM {} WHERE alias_kind = 'alt' AND lang = 'en' "
            "AND embedding IS NOT NULL ORDER BY canonical_id, alias"
        ).format(sql.Identifier(DEFAULT_TABLE))
    ).fetchall()
    return random.Random(SEED).sample([(a, c) for a, c in rows], HELD_OUT_SIZE)


def ambiguous_norms(index: SkillIndex) -> set[str]:
    """alias_norms that name more than one concept.

    JM-18 resolves these in its exact-alias tier (and asks the LLM), so they never
    reach vector auto-accept; as held-out queries they'd count as wrong at
    distance 0 and make any threshold impossible.
    """
    rows = index.connection.execute(
        sql.SQL(
            "SELECT alias_norm FROM {} GROUP BY alias_norm HAVING count(DISTINCT canonical_id) > 1"
        ).format(sql.Identifier(DEFAULT_TABLE))
    ).fetchall()
    return {row[0] for row in rows}


def held_out_outcomes(
    index: SkillIndex, queries: list[tuple[str, str]], vectors: np.ndarray
) -> list[Outcome]:
    """Query each alias, ignoring the alias's own row, and see if its concept comes back."""
    outcomes = []
    for (alias, cid), vector in zip(queries, vectors, strict=True):
        rows = index.nearest(vector, K * FETCH_FACTOR + 1)
        others = [c for _, c in rows if not (c.canonical_id == cid and c.matched_alias == alias)]
        ranked = dedupe_candidates(others, K)
        outcomes.append(Outcome(
            cid, tuple(c.canonical_id for c in ranked), ranked[0].distance if ranked else None
        ))
    return outcomes


def cv_outcomes(index: SkillIndex, path: Path) -> list[Outcome]:
    with path.open(encoding="utf-8", newline="") as f:
        pairs = [(r["raw"], r["expected_canonical_id"]) for r in csv.DictReader(f)]
    results = index.search([raw for raw, _ in pairs], k=K)
    return [
        Outcome(expected, tuple(c.canonical_id for c in ranked),
                ranked[0].distance if ranked else None)
        for (_, expected), ranked in zip(pairs, results, strict=True)
    ]


def exact_outcomes(index: SkillIndex, queries, vectors) -> list[Outcome]:
    """Same queries with the HNSW index disabled: brute-force, exact neighbours."""
    with index.connection.transaction():
        index.connection.execute("SET LOCAL enable_indexscan = off")
        return held_out_outcomes(index, queries, vectors)


def set_ef_search(index: SkillIndex, value: int) -> None:
    index.connection.execute(sql.SQL("SET hnsw.ef_search = {}").format(sql.Literal(value)))


def report_set(name: str, outcomes: list[Outcome]) -> None:
    print(f"\n== {name} ({len(outcomes)} queries)")
    print(f"recall@1 {recall_at_k(outcomes, 1):.3f}   recall@5 {recall_at_k(outcomes, K):.3f}")
    for label, values in distance_percentiles(outcomes).items():
        print(f"top-1 distance, {label:9}: {values}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the pgvector skill index.")
    parser.add_argument("--skip-prefix", action="store_true",
                        help="skip the 'passage: ' rebuild and comparison")
    args = parser.parse_args()
    url = database_url()
    if not url:
        raise SystemExit("DATABASE_URL is not set (environment or .env)")

    embedder = Embedder()
    with SkillIndex.connect(url, embedder) as index:
        for warning in index.warnings:
            print(f"WARNING: {warning}")
        build = index.meta.get("build", {})
        size = index.connection.execute(
            "SELECT pg_total_relation_size(%s::regclass)", [DEFAULT_TABLE]).fetchone()[0]
        print(f"index: {build.get('rows_total')} rows, {build.get('rows_embedded')} embedded, "
              f"built in {build.get('duration_s')}s, {size / 1e6:.0f} MB")

        queries = sample_held_out(index)
        vectors = embedder.embed([alias for alias, _ in queries])
        held_out = held_out_outcomes(index, queries, vectors)
        report_set("held-out en alt aliases", held_out)
        cv = cv_outcomes(index, CV_PHRASES)
        report_set("CV phrases", cv)
        misses = [(o, r) for o, r in zip(cv, _raws(CV_PHRASES), strict=True) if not o.top1_correct]
        for outcome, raw in misses:
            print(f"  top-1 miss: {raw!r} -> {outcome.ranked_ids[:1]} "
                  f"(expected rank {_rank(outcome)})")

        ambiguous = ambiguous_norms(index)
        single = [o for o, (alias, _) in zip(held_out, queries, strict=True)
                  if normalize(alias) not in ambiguous]
        excluded = len(held_out) - len(single)
        report_set("held-out, aliases naming a single concept", single)
        threshold = select_threshold(single, MIN_PRECISION, MIN_QUERIES_BELOW)
        print(f"\n== auto-accept threshold (precision >= {MIN_PRECISION}, n >= "
              f"{MIN_QUERIES_BELOW}, held-out set minus {excluded} ambiguous aliases)")
        if threshold is None:
            print("no distance qualifies; JM-18 should not auto-accept vector matches")
        else:
            cv_below = [o for o in cv if o.top1_distance is not None
                        and o.top1_distance <= threshold.auto_accept_distance]
            cv_precision = (sum(o.top1_correct for o in cv_below) / len(cv_below)
                            if cv_below else None)
            print(f"distance <= {threshold.auto_accept_distance}: precision "
                  f"{threshold.precision}, coverage {threshold.coverage} (n={threshold.n})")
            print(f"same threshold on CV phrases: {len(cv_below)}/{len(cv)} below, "
                  f"precision {cv_precision}")
            _store_threshold(index, threshold, excluded)

        subset_q, subset_v = queries[:EXACT_CHECK_SIZE], vectors[:EXACT_CHECK_SIZE]
        hnsw = held_out[:EXACT_CHECK_SIZE]
        exact = exact_outcomes(index, subset_q, subset_v)
        diff = recall_at_k(exact, K) - recall_at_k(hnsw, K)
        print(f"\n== HNSW vs exact ({EXACT_CHECK_SIZE} queries)")
        print(f"recall@5 HNSW {recall_at_k(hnsw, K):.3f}, exact {recall_at_k(exact, K):.3f}, "
              f"HNSW loses {diff * 100:.1f} points")
        if diff > 0.01:
            set_ef_search(index, HIGHER_EF_SEARCH)
            higher = held_out_outcomes(index, subset_q, subset_v)
            print(f"with ef_search={HIGHER_EF_SEARCH}: recall@5 {recall_at_k(higher, K):.3f}")

    if not args.skip_prefix:
        compare_prefix(url, queries, held_out)
    return 0


def compare_prefix(url: str, queries, query_outcomes: list[Outcome]) -> None:
    """Documents embedded with 'passage: ', queries still with 'query: '."""
    print(f"\n== prefix comparison: building {PASSAGE_TABLE} with 'passage: ' documents")
    build_table(DEFAULT_LANGUAGES, "passage: ", PASSAGE_TABLE)
    try:
        query_embedder = Embedder()
        with SkillIndex.connect(url, query_embedder, table=PASSAGE_TABLE) as passage:
            vectors = query_embedder.embed([alias for alias, _ in queries])
            outcomes = held_out_outcomes(passage, queries, vectors)
    finally:
        with SkillIndex.connect(url) as index:
            index.connection.execute(
                sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(PASSAGE_TABLE)))
    print(f"held-out recall@1 / @5: query: {recall_at_k(query_outcomes, 1):.3f} / "
          f"{recall_at_k(query_outcomes, K):.3f}   passage: {recall_at_k(outcomes, 1):.3f} / "
          f"{recall_at_k(outcomes, K):.3f}  ({PASSAGE_TABLE} dropped)")


def _store_threshold(index: SkillIndex, threshold, excluded: int) -> None:
    value = {
        "auto_accept_distance": threshold.auto_accept_distance,
        "precision": threshold.precision,
        "coverage": threshold.coverage,
        "n": threshold.n,
        "measured_on": (
            f"{HELD_OUT_SIZE} held-out en alt aliases, seed {SEED}, minus {excluded} whose "
            "text names several concepts (resolved by JM-18's exact tier, not by vectors)"
        ),
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    index.connection.execute(
        "INSERT INTO index_meta (key, value) VALUES ('thresholds', %s) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        [json.dumps(value)],
    )
    print("stored in index_meta['thresholds']")


def _raws(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as f:
        return [r["raw"] for r in csv.DictReader(f)]


def _rank(outcome: Outcome) -> str:
    return str(outcome.ranked_ids.index(outcome.expected) + 1) if outcome.expected in \
        outcome.ranked_ids else f">{K}"


if __name__ == "__main__":
    sys.exit(main())
