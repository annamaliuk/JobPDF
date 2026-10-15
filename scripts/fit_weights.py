"""Fit the learned fit-score weights from feedback (JM-27).

Reads "useful / not useful" feedback (JSONL, format in jobpdf/ranking/learned.py),
fits ``w x skill + (1 - w) x semantic`` and writes jobpdf/ranking/learned_weights.json
with what it was fitted on. Nothing changes in the app until JOBPDF_SCORER=learned.
Prints counts and numbers only, never rows.

Usage (from the repo root):
    uv run python scripts/fit_weights.py --feedback path/to/feedback.jsonl
    uv run python scripts/fit_weights.py --feedback f.jsonl --out /tmp/w.json --prior-strength 40

Run scripts/compare_scorers.py on the same file before switching the flag on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jobpdf.ranking.learned import (
    DEFAULT_PRIOR_STRENGTH,
    MIN_ROWS,
    RECOMMENDED_ROWS,
    WEIGHTS_PATH,
    NotEnoughData,
    file_sha256,
    fit_weights,
    read_feedback,
    write_weights,
)
from jobpdf.ranking.score import W_SEMANTIC, W_SKILL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--prior-strength", type=float, default=DEFAULT_PRIOR_STRENGTH,
                        help="rows the 0.7/0.3 prior is worth (default %(default)s)")
    args = parser.parse_args(argv)

    try:
        rows = read_feedback(args.feedback)
    except (OSError, ValueError) as exc:
        print(f"cannot read feedback: {exc}" if isinstance(exc, ValueError)
              else f"cannot read feedback: {type(exc).__name__}")
        return 1
    try:
        summary = fit_weights(rows, prior_strength=args.prior_strength)
    except NotEnoughData as exc:
        print(f"not enough feedback: {exc}")
        return 1

    print(f"rows: {summary.n_rows} used ({summary.n_positive} useful), "
          f"{summary.n_skipped} skipped for a missing score")
    print(f"prior   : skill={W_SKILL} semantic={W_SEMANTIC}  (strength {summary.prior_strength})")
    print(f"learned : skill={summary.w_skill} semantic={summary.w_semantic}")
    print(f"log loss: {summary.log_loss:.4f} learned vs {summary.log_loss_prior:.4f} prior")
    if not summary.informative:
        print("WARNING: useful cards do not rank higher on the score than not-useful ones, "
              "so the weights stay at the prior")
    if summary.below_recommended:
        print(f"WARNING: fewer than {RECOMMENDED_ROWS} rated cards (minimum {MIN_ROWS}); "
              "treat the weights as a hint, not a result")

    stored = write_weights(summary, args.out, dataset_sha256=file_sha256(args.feedback))
    print(f"written to {args.out.name} (calibration {stored.calibration_fingerprint})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
