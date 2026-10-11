"""Evaluate grounding (JM-12): how many extracted quotes are found, and how.

Grounds the gold extractions of the synthetic CV fixtures and, with
``--replay-dir``, the recorded LLM responses for the same fixtures. No network
calls, no API key. The report holds counts, rates and fuzzy scores only, never
CV text or quotes.

Usage (from the repo root):
    uv run python scripts/eval_grounding.py
    uv run python scripts/eval_grounding.py --replay-dir path/to/recordings

The fuzzy-score distribution is the evidence for tuning FUZZY_MIN_SCORE and
MIN_FUZZY_QUOTE_LEN in jobpdf/extraction/grounding.py; tune there, not here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path

from jobpdf.extraction import parse
from jobpdf.extraction.extract import extract_profile
from jobpdf.extraction.grounding import (
    FUZZY_MIN_SCORE,
    MIN_FUZZY_QUOTE_LEN,
    GroundingResult,
    GroundingStatus,
    ground,
)
from jobpdf.extraction.llm import ReplayMissError, ReplayToolCaller
from jobpdf.extraction.prompt import mask_pii
from jobpdf.extraction.schema import ExtractedProfile
from jobpdf.extraction.sections import split_sections

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "extraction" / "fixtures"
GOLD = FIXTURES / "gold"
SOURCES = {
    "single_column": "single_column.pdf",
    "two_column": "two_column.pdf",
    "table_cv": "table_cv.docx",
    "left_sidebar": "left_sidebar.pdf",
}
DEFAULT_OUT = ROOT / "data" / "eval" / "grounding" / "report.json"
HISTOGRAM_BUCKET = 2  # fuzzy-score points per histogram bucket


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replay-dir", type=Path, help="recorded responses for the fixtures")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    datasets: dict[str, dict[str, GroundingResult]] = {"gold": {}}
    skipped: list[str] = []
    hashed = []
    for name, source in SOURCES.items():
        cv_path, gold_path = FIXTURES / source, GOLD / f"{name}.json"
        hashed += [cv_path, gold_path]
        doc = parse(cv_path)
        sections, _ = split_sections(doc)
        masked, _ = mask_pii(doc.full_text)
        gold = ExtractedProfile.model_validate_json(gold_path.read_text(encoding="utf-8"))
        datasets["gold"][name] = ground(gold, doc, sections, masked)
        if args.replay_dir is None:
            continue
        try:
            extraction = extract_profile(
                doc, sections, ReplayToolCaller(args.replay_dir), use_cache=False
            )
        except ReplayMissError:
            skipped.append(name)
            continue
        datasets.setdefault("replay", {})[name] = ground(
            extraction.profile, doc, sections, masked
        )
    if args.replay_dir is not None:
        datasets.setdefault("replay", {})
        hashed += sorted(args.replay_dir.glob("*.json"))

    report = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset_sha256": dataset_hash(hashed),
        "fuzzy_min_score": FUZZY_MIN_SCORE,
        "min_fuzzy_quote_len": MIN_FUZZY_QUOTE_LEN,
        "datasets": {kind: summarize(results) for kind, results in datasets.items()},
    }
    if args.replay_dir is not None:
        report["datasets"]["replay"]["skipped_without_recording"] = skipped

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(report, args.out)
    return 0


def summarize(results: dict[str, GroundingResult]) -> dict:
    counts = dict.fromkeys((s.value for s in GroundingStatus), 0)
    for result in results.values():
        for status, n in result.counts.items():
            counts[status.value] += n
    total = sum(counts.values())
    groundings = [g for r in results.values() for g in r.groundings]
    scores = sorted(g.score for g in groundings if g.score is not None)
    histogram: dict[str, int] = {}
    for score in scores:
        low = math.floor(score / HISTOGRAM_BUCKET) * HISTOGRAM_BUCKET
        label = f"{low}-{low + HISTOGRAM_BUCKET}"
        histogram[label] = histogram.get(label, 0) + 1
    return {
        "documents": len(results),
        "quotes": total,
        "counts": counts,
        "shares": {s: round(n / total, 4) if total else None for s, n in counts.items()},
        "exact_rate": round(counts["exact"] / total, 4) if total else None,
        "ambiguous": sum(g.ambiguous for g in groundings),
        "fuzzy_scores": scores,
        "fuzzy_score_histogram": histogram,
        "per_document": {
            name: {s.value: n for s, n in r.counts.items()} for name, r in results.items()
        },
    }


def dataset_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    # File names, not absolute paths, so the hash is the same on every machine.
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8") + b"\x00")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def print_summary(report: dict, out: Path) -> None:
    for kind, summary in report["datasets"].items():
        rate = summary["exact_rate"]
        counts = ", ".join(f"{s} {n}" for s, n in summary["counts"].items())
        rate_text = f"{rate:.1%}" if rate is not None else "n/a"
        print(f"{kind}: {summary['documents']} CVs, {summary['quotes']} quotes | {counts} | "
              f"exact rate {rate_text}, ambiguous {summary['ambiguous']}")
        if summary["fuzzy_scores"]:
            print(f"  fuzzy scores: {summary['fuzzy_scores']}")
        if summary.get("skipped_without_recording"):
            print(f"  no recording for: {', '.join(summary['skipped_without_recording'])}")
    print(f"FUZZY_MIN_SCORE={report['fuzzy_min_score']}, "
          f"MIN_FUZZY_QUOTE_LEN={report['min_fuzzy_quote_len']}")
    print(f"Report written to {out}")


if __name__ == "__main__":
    sys.exit(main())
