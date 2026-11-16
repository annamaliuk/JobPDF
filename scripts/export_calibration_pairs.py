"""Write JM-25's synthetic pairs in the format scripts/calibrate_semantic.py reads.

Grade 3 pairs count as related and grade 0 pairs as unrelated; grades 1 and 2 are left
out. The candidate side is each candidate's first job text and the vacancy side its title
and duties, built by the same code the semantic score uses. Prints counts only.

Usage (from the repo root):
    uv run python scripts/export_calibration_pairs.py --out calibration_pairs.json
    uv run python scripts/calibrate_semantic.py --pairs calibration_pairs.json --provisional
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jobpdf.ranking.semantic import build_candidate_texts, build_vacancy_texts
from jobpdf.ranking.synthetic import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "synthetic" / "ranking"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    dataset = load_dataset(args.data)
    cv_text = {c.id: texts[0].text for c in dataset.candidates
               if (texts := build_candidate_texts(c.profile)[0])}
    vacancy_text = {v.id: build_vacancy_texts(v)[0] for v in dataset.vacancies}
    pairs = [{"cv_text": cv_text[p.candidate_id], "vacancy_text": vacancy_text[p.vacancy_id],
              "related": p.grade == 3}
             for p in dataset.pairs if p.grade in (0, 3) and p.candidate_id in cv_text]
    args.out.write_text(json.dumps({"pairs": pairs}, ensure_ascii=False), encoding="utf-8")
    related = sum(p["related"] for p in pairs)
    print(f"written {len(pairs)} pairs: {related} related, {len(pairs) - related} unrelated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
