"""Compare the linear and the learned scorer on feedback (JM-27).

Cross-validated by session, so each scorer is judged on sessions it was not fitted
on, with cards sorted like the app does (band first, then fit). Prints NDCG@k and
Spearman for both, the paired difference with a 95% bootstrap interval, and a
verdict that is "inconclusive" unless that interval excludes zero. Numbers only.

Usage (from the repo root):
    uv run python scripts/compare_scorers.py --feedback path/to/feedback.jsonl
    uv run python scripts/compare_scorers.py --feedback f.jsonl --folds 10 --k 5 --seed 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jobpdf.ranking.learned import DEFAULT_PRIOR_STRENGTH, NotEnoughData, read_feedback
from jobpdf.ranking.scorer_eval import (
    DEFAULT_FOLDS,
    DEFAULT_K,
    INCONCLUSIVE,
    LEARNED_BETTER,
    MIN_SESSIONS_FOR_VERDICT,
    Comparison,
    compare_scorers,
)

ADVICE = {
    LEARNED_BETTER: "learned weights beat the linear formula; reasonable to switch the flag on",
    "linear_better": "the linear formula beats the learned weights; keep JOBPDF_SCORER=linear",
    INCONCLUSIVE: "no reliable difference; keep JOBPDF_SCORER=linear and collect more feedback",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="NDCG cut-off (default %(default)s)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prior-strength", type=float, default=DEFAULT_PRIOR_STRENGTH)
    args = parser.parse_args(argv)

    try:
        rows = read_feedback(args.feedback)
    except (OSError, ValueError) as exc:
        print(f"cannot read feedback: {exc}" if isinstance(exc, ValueError)
              else f"cannot read feedback: {type(exc).__name__}")
        return 1
    try:
        result = compare_scorers(rows, folds=args.folds, k=args.k, seed=args.seed,
                                 prior_strength=args.prior_strength)
    except NotEnoughData as exc:
        print(f"cannot compare: {exc}")
        return 1

    _print(result)
    return 0


def _print(r: Comparison) -> None:
    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    print(f"sessions scored: {r.n_sessions} ({r.n_skipped_sessions} skipped), "
          f"{r.folds} folds ({r.n_skipped_folds} too small to fit)")
    print(f"{'':10}{'NDCG@' + str(r.k):>10}{'Spearman':>10}")
    print(f"{'linear':10}{fmt(r.linear_ndcg):>10}{fmt(r.linear_spearman):>10}")
    print(f"{'learned':10}{fmt(r.learned_ndcg):>10}{fmt(r.learned_spearman):>10}")
    low, high = r.ndcg_diff_ci
    print(f"NDCG@{r.k} difference (learned - linear): {r.ndcg_diff:+.3f}  "
          f"95% interval [{low:+.3f}, {high:+.3f}]")
    print(f"per session: learned better {r.wins}, equal {r.ties}, worse {r.losses}")
    print("fold weights (skill/semantic): "
          + ", ".join(f"{w['skill']:.2f}/{w['semantic']:.2f}" for w in r.fold_weights))
    if r.n_sessions < MIN_SESSIONS_FOR_VERDICT:
        print(f"note: fewer than {MIN_SESSIONS_FOR_VERDICT} sessions, so no verdict is given")
    print(f"verdict: {r.verdict} -- {ADVICE[r.verdict]}")


if __name__ == "__main__":
    sys.exit(main())
