"""Measure the floor and ceiling that rescale JM-23's semantic score.

e5 cosine similarities sit in a narrow band, so raw values are rescaled:
floor = median similarity of unrelated pairs, ceil = median of related pairs.
The result goes to jobpdf/ranking/semantic_calibration.json with its provenance
(model, dataset hash, time). Re-running on the same pairs gives the same values.
JM-26 re-runs it on JM-25's dataset. Prints counts and numbers only, never texts.

Usage (from the repo root; needs the e5 model in the local cache):
    uv run python scripts/calibrate_semantic.py --provisional
    uv run python scripts/calibrate_semantic.py --pairs path/to/pairs.json --out path/to/out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from jobpdf.normalization.embedder import MODEL_NAME, MODEL_REVISION, Embedder
from jobpdf.ranking.semantic import CALIBRATION_PATH, embedding_version

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = ROOT / "tests" / "ranking" / "fixtures" / "semantic_calibration_pairs.json"
DECIMALS = 4


def main(argv: list[str] | None = None, embedder: Embedder | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--out", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--provisional", action="store_true",
                        help="mark the values as provisional (synthetic or small dataset)")
    args = parser.parse_args(argv)

    raw = args.pairs.read_bytes()
    pairs = json.loads(raw.decode("utf-8"))["pairs"]
    if not all(isinstance(p.get("related"), bool) and p.get("cv_text") and p.get("vacancy_text")
               for p in pairs):
        print("every pair needs cv_text, vacancy_text and a boolean 'related'")
        return 1

    embedder = embedder if embedder is not None else Embedder()
    cv = np.asarray(embedder.embed([p["cv_text"] for p in pairs]), dtype=np.float64)
    vacancy = np.asarray(embedder.embed([p["vacancy_text"] for p in pairs]), dtype=np.float64)
    similarities = np.sum(cv * vacancy, axis=1)  # unit vectors: dot product = cosine
    related = [float(s) for s, p in zip(similarities, pairs, strict=True) if p["related"]]
    unrelated = [float(s) for s, p in zip(similarities, pairs, strict=True) if not p["related"]]
    if not related or not unrelated:
        print("need at least one related and one unrelated pair")
        return 1

    floor, ceil = statistics.median(unrelated), statistics.median(related)
    result = {
        "floor": round(floor, DECIMALS),
        "ceil": round(ceil, DECIMALS),
        "provisional": args.provisional,
        "model": f"{MODEL_NAME}@{MODEL_REVISION}",
        "embedding_version": embedding_version(embedder),
        "dataset": _display_path(args.pairs),
        "dataset_sha256": dataset_sha256(raw),
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_related": len(related),
        "n_unrelated": len(unrelated),
        "related_range": _summary(related),
        "unrelated_range": _summary(unrelated),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"pairs: {len(related)} related, {len(unrelated)} unrelated")
    print(f"related   cosine: {_fmt(result['related_range'])}")
    print(f"unrelated cosine: {_fmt(result['unrelated_range'])}")
    print(f"floor={result['floor']} ceil={result['ceil']} provisional={args.provisional}")
    if ceil <= floor:
        print("WARNING: ceil <= floor; the score falls back to uncalibrated until re-measured")
    print(f"written to {_display_path(args.out)}")
    return 0


def dataset_sha256(raw: bytes) -> str:
    """Hash with LF line endings, so a Windows checkout (autocrlf) hashes the same."""
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def _summary(values: list[float]) -> dict[str, float]:
    return {"min": round(min(values), DECIMALS),
            "median": round(statistics.median(values), DECIMALS),
            "max": round(max(values), DECIMALS)}


def _fmt(summary: dict[str, float]) -> str:
    return f"min {summary['min']}  median {summary['median']}  max {summary['max']}"


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


if __name__ == "__main__":
    sys.exit(main())
