"""Pure helpers for evaluating the skill index (JM-17); no database or model here."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# JM-18 auto-accepts a top-1 vector match only below this measured distance.
MIN_PRECISION = 0.98
MIN_QUERIES_BELOW = 50
PERCENTILES = (10, 50, 90)


@dataclass(frozen=True)
class Outcome:
    """One query's result: expected concept, ranked candidate ids, top-1 distance."""

    expected: str
    ranked_ids: tuple[str, ...]
    top1_distance: float | None  # None when nothing came back

    @property
    def top1_correct(self) -> bool:
        return bool(self.ranked_ids) and self.ranked_ids[0] == self.expected


@dataclass(frozen=True)
class Threshold:
    auto_accept_distance: float
    precision: float
    coverage: float  # share of all queries whose top-1 distance is <= the threshold
    n: int  # queries at or below the threshold


def recall_at_k(outcomes: Sequence[Outcome], k: int) -> float:
    if not outcomes:
        return 0.0
    return sum(o.expected in o.ranked_ids[:k] for o in outcomes) / len(outcomes)


def distance_percentiles(outcomes: Sequence[Outcome]) -> dict[str, dict[str, float] | None]:
    """p10/p50/p90 of top-1 distance, split by whether the top-1 was correct."""
    def summary(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        points = np.percentile(values, PERCENTILES)
        return {f"p{p}": round(float(v), 4) for p, v in zip(PERCENTILES, points, strict=True)}

    scored = [o for o in outcomes if o.top1_distance is not None]
    return {
        "correct": summary([o.top1_distance for o in scored if o.top1_correct]),
        "incorrect": summary([o.top1_distance for o in scored if not o.top1_correct]),
    }


def select_threshold(
    outcomes: Sequence[Outcome],
    min_precision: float = MIN_PRECISION,
    min_n: int = MIN_QUERIES_BELOW,
) -> Threshold | None:
    """Largest top-1 distance d where precision among queries with distance <= d
    is >= min_precision and at least min_n queries fall at or below d.

    None when no d qualifies; JM-18 then never auto-accepts vector matches.
    """
    scored = sorted(
        (o for o in outcomes if o.top1_distance is not None), key=lambda o: o.top1_distance
    )
    best: Threshold | None = None
    correct = 0
    for i, outcome in enumerate(scored, start=1):
        correct += outcome.top1_correct
        # Only cut between distinct distances, so ties are all in or all out.
        if i < len(scored) and scored[i].top1_distance == outcome.top1_distance:
            continue
        precision = correct / i
        if i >= min_n and precision >= min_precision:
            best = Threshold(
                auto_accept_distance=round(float(outcome.top1_distance), 4),
                precision=round(precision, 4),
                coverage=round(i / len(outcomes), 4),
                n=i,
            )
    return best
