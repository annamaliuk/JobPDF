"""Offline comparison of the linear and the learned scorer on feedback (JM-27).

The question this answers: on cards real users rated, does the learned scorer put the
useful ones higher than the hand-tuned formula? Two things keep the answer honest:

- Cross-validation grouped by session. Weights are fitted on some sessions (one
  session = one CV request) and scored on others, so a scorer never sees the cards it
  is judged on.
- Product order. Cards are sorted like ``rank_vacancies`` does: band first, then fit.
  Learned weights only reorder cards inside a band, so that is where any gain shows.

Metrics are NDCG@k and Spearman rank correlation per session (the JM-26 metrics),
averaged over sessions, plus a bootstrap interval on the paired NDCG difference. The
verdict is "inconclusive" unless that interval excludes zero. Prints and returns
numbers only, never rows.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from jobpdf.ranking.learned import (
    DEFAULT_PRIOR_STRENGTH,
    FeedbackRow,
    LearnedLinearScorer,
    NotEnoughData,
    components_of,
    fit_weights,
)
from jobpdf.ranking.score import BAND_ORDER, LinearScorer, Scorer

DEFAULT_K = 10
DEFAULT_FOLDS = 5
BOOTSTRAP_SAMPLES = 2000
MIN_SESSIONS_FOR_VERDICT = 20
TIE_TOLERANCE = 1e-9

LEARNED_BETTER = "learned_better"
LINEAR_BETTER = "linear_better"
INCONCLUSIVE = "inconclusive"


# --- metrics -----------------------------------------------------------------------


def ndcg_at_k(labels_in_rank_order: Sequence[int], k: int = DEFAULT_K) -> float | None:
    """NDCG@k with binary gains; None when there is no useful card to find."""
    if sum(labels_in_rank_order) == 0:
        return None

    def dcg(labels: Sequence[int]) -> float:
        return sum(label / math.log2(i + 2) for i, label in enumerate(labels[:k]))

    return dcg(labels_in_rank_order) / dcg(sorted(labels_in_rank_order, reverse=True))


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman correlation with average ranks for ties; None if either side is constant."""
    if len(a) != len(b) or len(a) < 2:
        return None
    ra, rb = _average_ranks(a), _average_ranks(b)
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0  # tied values share their mean rank
        i = j + 1
    return ranks


# --- one session -------------------------------------------------------------------


def rank_session(rows: Sequence[FeedbackRow], scorer: Scorer) -> list[FeedbackRow]:
    """The order a user would have seen: band, then fit (None last), then vacancy id.

    The real order also uses the posting date before the id; feedback does not carry
    it, and exact score ties are rare, so the id stands in.
    """
    def key(row: FeedbackRow) -> tuple:
        fit = scorer.score(components_of(row))
        band = BAND_ORDER[row.band] if row.band is not None else 0  # no band: one group
        return (band, fit is None, -(fit or 0.0), row.vacancy_id)

    return sorted(rows, key=key)


def evaluate_session(
    rows: Sequence[FeedbackRow], scorer: Scorer, k: int = DEFAULT_K
) -> tuple[float, float | None] | None:
    """(NDCG@k, Spearman) of one session, or None when it has no useful card."""
    ranked = rank_session(rows, scorer)
    labels = [row.label for row in ranked]
    ndcg = ndcg_at_k(labels, k)
    if ndcg is None:
        return None
    positions = [-float(i) for i in range(len(ranked))]  # earlier position = higher
    return ndcg, spearman(positions, [float(label) for label in labels])


# --- cross-validated comparison ----------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    k: int
    folds: int
    n_sessions: int  # sessions scored
    n_skipped_sessions: int  # no useful card, or their fold could not be fitted
    n_skipped_folds: int
    linear_ndcg: float
    learned_ndcg: float
    ndcg_diff: float  # learned minus linear
    ndcg_diff_ci: tuple[float, float]  # 95% bootstrap interval over sessions
    linear_spearman: float | None
    learned_spearman: float | None
    wins: int  # sessions where learned beat linear
    ties: int
    losses: int
    fold_weights: list[dict[str, float]]
    verdict: str


def make_folds(
    rows: Sequence[FeedbackRow], folds: int = DEFAULT_FOLDS, seed: int = 0
) -> list[tuple[list[FeedbackRow], list[FeedbackRow]]]:
    """(train, test) pairs where every session is in exactly one test set."""
    by_session: dict[str, list[FeedbackRow]] = defaultdict(list)
    for row in rows:
        by_session[row.session].append(row)
    sessions = sorted(by_session)
    folds = max(2, min(folds, len(sessions)))
    order = np.random.default_rng(seed).permutation(len(sessions))
    assignment = {sessions[index]: position % folds for position, index in enumerate(order)}
    result = []
    for fold in range(folds):
        train = [r for s in sessions if assignment[s] != fold for r in by_session[s]]
        test = [r for s in sessions if assignment[s] == fold for r in by_session[s]]
        result.append((train, test))
    return result


def compare_scorers(
    rows: Sequence[FeedbackRow],
    *,
    folds: int = DEFAULT_FOLDS,
    k: int = DEFAULT_K,
    seed: int = 0,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    bootstrap: int = BOOTSTRAP_SAMPLES,
) -> Comparison:
    """Cross-validated linear-vs-learned comparison. Raises NotEnoughData if no session
    could be scored."""
    linear = LinearScorer()
    linear_scores: list[tuple[float, float | None]] = []
    learned_scores: list[tuple[float, float | None]] = []
    fold_weights: list[dict[str, float]] = []
    skipped_folds = 0
    all_sessions = {row.session for row in rows}

    fold_pairs = make_folds(rows, folds, seed)
    for train, test in fold_pairs:
        try:
            learned: Scorer = LearnedLinearScorer.from_summary(
                fit_weights(train, prior_strength=prior_strength))
        except NotEnoughData:
            skipped_folds += 1
            continue
        fold_weights.append(dict(learned.weights))  # type: ignore[attr-defined]
        by_session: dict[str, list[FeedbackRow]] = defaultdict(list)
        for row in test:
            by_session[row.session].append(row)
        for session in sorted(by_session):
            base = evaluate_session(by_session[session], linear, k)
            other = evaluate_session(by_session[session], learned, k)
            if base is not None and other is not None:
                linear_scores.append(base)
                learned_scores.append(other)

    if not linear_scores:
        raise NotEnoughData("no session could be scored: too few sessions, or none with a "
                            "useful card, or every training fold was too small to fit")

    base_ndcg = np.array([s[0] for s in linear_scores])
    new_ndcg = np.array([s[0] for s in learned_scores])
    diffs = new_ndcg - base_ndcg
    ci = _bootstrap_ci(diffs, bootstrap, seed)
    n = len(diffs)
    return Comparison(
        k=k, folds=len(fold_pairs), n_sessions=n,
        n_skipped_sessions=len(all_sessions) - n, n_skipped_folds=skipped_folds,
        linear_ndcg=float(base_ndcg.mean()), learned_ndcg=float(new_ndcg.mean()),
        ndcg_diff=float(diffs.mean()), ndcg_diff_ci=ci,
        linear_spearman=_mean_defined([s[1] for s in linear_scores]),
        learned_spearman=_mean_defined([s[1] for s in learned_scores]),
        wins=int((diffs > TIE_TOLERANCE).sum()), ties=int((np.abs(diffs) <= TIE_TOLERANCE).sum()),
        losses=int((diffs < -TIE_TOLERANCE).sum()), fold_weights=fold_weights,
        verdict=verdict(ci, n),
    )


def verdict(ci: tuple[float, float], n_sessions: int) -> str:
    """"learned_better" / "linear_better" only when the interval excludes zero and
    there are enough sessions for that to mean something."""
    if n_sessions < MIN_SESSIONS_FOR_VERDICT:
        return INCONCLUSIVE
    low, high = ci
    if low > 0:
        return LEARNED_BETTER
    if high < 0:
        return LINEAR_BETTER
    return INCONCLUSIVE


def _bootstrap_ci(diffs: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    if len(diffs) < 2:
        mean = float(diffs.mean())
        return mean, mean
    rng = np.random.default_rng(seed)
    means = rng.choice(diffs, size=(samples, len(diffs)), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [v for v in values if v is not None]
    return float(np.mean(defined)) if defined else None


__all__ = [
    "Comparison", "INCONCLUSIVE", "LEARNED_BETTER", "LINEAR_BETTER", "compare_scorers",
    "evaluate_session", "make_folds", "ndcg_at_k", "rank_session", "spearman", "verdict",
]
