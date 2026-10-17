"""Evaluate the ranking on JM-25's synthetic pairs (JM-26). Prints numbers only.

Modes: ``perfect`` (ground-truth skill keys), ``raw`` (no taxonomy, the baseline) and
``real`` (JM-20 normalization; needs the built index). The semantic part needs the e5
model in the local cache; with ``--no-semantic`` the run is skills only.

Usage (from the repo root):
    uv run python scripts/eval_ranking.py --mode perfect
    uv run python scripts/eval_ranking.py --mode real --write
    uv run python scripts/eval_ranking.py --mode perfect --calibration new_calibration.json

``--write`` stores the report under its mode in jobpdf/ranking/ranking_eval.json, with
the dataset hashes, versions and date it was measured on.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from jobpdf.normalization.service import INDEX_UNAVAILABLE_VERSION
from jobpdf.ranking.models import RANKING_VERSION
from jobpdf.ranking.ranking_eval import (
    CURRENT,
    DEFAULT_FOLDS,
    DEFAULT_K,
    DEFAULT_SEED,
    MODES,
    build_pairs,
    run_evaluation,
)
from jobpdf.ranking.semantic import CALIBRATION_PATH, load_calibration
from jobpdf.ranking.synthetic import MANIFEST_FILE, load_dataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "synthetic" / "ranking"
RESULTS_PATH = ROOT / "jobpdf" / "ranking" / "ranking_eval.json"


def main(argv: list[str] | None = None, *, embedder=None, normalizer=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=MODES, default="perfect")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--no-semantic", action="store_true", help="skills only, no model")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--write", action="store_true", help=f"store the report in "
                        f"{RESULTS_PATH.relative_to(ROOT).as_posix()}")
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    args = parser.parse_args(argv)

    dataset = load_dataset(args.data)
    calibration = load_calibration(args.calibration)
    if not args.no_semantic and embedder is None:
        from jobpdf.ranking.semantic import default_embedder
        embedder = default_embedder()
    if args.mode == "real" and normalizer is None:
        from jobpdf.normalization.service import get_normalization_service
        normalizer = get_normalization_service()
    versions = getattr(normalizer, "versions", None)
    if args.mode == "real" and versions is not None and versions.index == INDEX_UNAVAILABLE_VERSION:
        print("real mode needs the skill index: set DATABASE_URL, run build_taxonomy.py and "
              "build_index.py, then retry", file=sys.stderr)
        return 2

    pairs = build_pairs(dataset, args.mode, embedder=None if args.no_semantic else embedder,
                        normalizer=normalizer, calibration=calibration)
    report = run_evaluation(pairs, k=args.k, folds=args.folds, seed=args.seed)
    _print(args.mode, report)

    if args.write:
        manifest = json.loads((args.data / MANIFEST_FILE).read_text(encoding="utf-8"))
        report["provenance"] = {
            "dataset": _display_path(args.data),
            "dataset_files": manifest["files"],
            "generator_version": manifest["generator_version"],
            "ranking_version": RANKING_VERSION,
            "embedding_version": calibration.embedding_version if report["has_semantic"]
            else None,
            "calibration": {"floor": calibration.floor, "ceil": calibration.ceil,
                            "dataset_sha256": calibration.dataset_sha256},
            "normalization": (normalizer.versions.model_dump()
                              if args.mode == "real" and normalizer is not None
                              and hasattr(normalizer, "versions") else None),
            "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        stored = (json.loads(args.results.read_text(encoding="utf-8"))
                  if args.results.exists() else {})
        stored[args.mode] = report
        args.results.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
        print(f"\nwritten to {_display_path(args.results)} under '{args.mode}'")
    return 0


def _fmt(stat: dict) -> str:
    if stat["mean"] is None:
        return "   -          "
    lo, hi = stat["ci95"]
    return f"{stat['mean']:.3f} [{lo:.3f}, {hi:.3f}]"


def _print(mode: str, report: dict) -> None:
    print(f"mode: {mode}  candidates: {report['n_candidates']}  pairs: {report['n_pairs']}  "
          f"semantic: {'yes' if report['has_semantic'] else 'no'}  "
          f"oracle Spearman: {report['oracle_spearman']}")
    print(f"\n{'variant':16} {'Spearman [95% CI]':24} {'NDCG@' + str(report['k']):24} "
          f"{'band F1':8} {'vs current (paired)':24}")
    for name, v in report["variants"].items():
        diff = v.get("vs_current")
        vs = f"{diff['mean']:+.3f} [{diff['ci95'][0]:+.3f}, {diff['ci95'][1]:+.3f}]" \
            if diff and diff["mean"] is not None else ""
        f1 = f"{v['band_macro_f1']:.3f}" if "band_macro_f1" in v else "-"
        print(f"{name:16} {_fmt(v['spearman']):24} {_fmt(v['ndcg']):24} {f1:8} {vs}")
    cases = report["variants"][CURRENT]["spearman_by_case"]
    print("\ncurrent, Spearman by case: "
          + ", ".join(f"{c}={m:.3f}" for c, m in cases.items() if m is not None))
    if report["semantic_auc"] is not None:
        print(f"semantic AUC (grade 3 vs grade 0, raw cosine): {report['semantic_auc']:.3f}")
    for name, t in report["tuning"].items():
        print(f"tuning {name}: {t['recommendation']} (most chosen {t['most_chosen']} in "
              f"{t['folds_choosing_it']}/{t['folds']} folds; {t['reason']})")
    if report["noisy"]:
        print("WARNING: the 95% interval on the current Spearman is wider than 0.1; "
              "regenerate JM-25 larger before deciding anything")


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


if __name__ == "__main__":
    sys.exit(main())
