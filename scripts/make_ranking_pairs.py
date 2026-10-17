"""Write the synthetic ranking dataset (JM-25) that JM-26 evaluates against.

Same arguments, same bytes: the output is committed and its manifest records the
seed, the sizes and a sha256 per file. Prints counts only.

Usage (from the repo root):
    uv run python scripts/make_ranking_pairs.py
    uv run python scripts/make_ranking_pairs.py --vacancies-per-family 10 --candidates 40
    uv run python scripts/make_ranking_pairs.py --export-calibration path/to/pairs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from jobpdf.ranking.synthetic import (
    DEFAULT_CANDIDATES,
    DEFAULT_SEED,
    DEFAULT_VACANCIES_PER_FAMILY,
    calibration_pairs,
    generate,
    write_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "synthetic" / "ranking"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--vacancies-per-family", type=int, default=DEFAULT_VACANCIES_PER_FAMILY)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--export-calibration", type=Path, default=None,
                        help="also write grade 3 / grade 0 pairs in calibrate_semantic.py format")
    args = parser.parse_args(argv)

    dataset = generate(args.seed, args.vacancies_per_family, args.candidates)
    write_dataset(dataset, args.out, args.seed, args.vacancies_per_family)
    grades = Counter(p.grade for p in dataset.pairs)
    traps = Counter(t.trap for t in dataset.truth.candidates.values() if t.trap)
    print(f"{len(dataset.vacancies)} vacancies, {len(dataset.candidates)} candidates, "
          f"{len(dataset.pairs)} pairs")
    print("grades: " + ", ".join(f"{g}={grades[g]}" for g in sorted(grades)))
    print("hard cases: " + ", ".join(f"{t}={n}" for t, n in sorted(traps.items())))
    print(f"written to {_display_path(args.out)}")

    if args.export_calibration:
        pairs = calibration_pairs(dataset)
        args.export_calibration.parent.mkdir(parents=True, exist_ok=True)
        args.export_calibration.write_text(
            json.dumps(pairs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        related = sum(p["related"] for p in pairs["pairs"])
        print(f"calibration pairs: {related} related, {len(pairs['pairs']) - related} unrelated, "
              f"written to {_display_path(args.export_calibration)}")
    return 0


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


if __name__ == "__main__":
    sys.exit(main())
