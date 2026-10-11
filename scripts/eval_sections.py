"""Evaluate section chunking (JM-10) on real CVs against hand-made labels.

Real CVs and labels live only in gitignored data/private/. The labels format is
documented by scripts/section_labels.example.json:

    {"cv1.pdf": [{"heading": "Work Experience", "types": ["experience"]}, ...]}

Usage (from the repo root):
    uv run python scripts/eval_sections.py
    uv run python scripts/eval_sections.py --cvs-dir data/private/cvs \
        --labels data/private/section_labels.json

Output stays in the terminal; headings printed under "other" are the best
candidates for new synonyms in jobpdf/extraction/section_keywords.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from jobpdf.extraction import ParseError, parse
from jobpdf.extraction.headings import normalize_heading
from jobpdf.extraction.models import Section
from jobpdf.extraction.sections import NO_HEADINGS_WARNING, split_sections

SUPPORTED = {".pdf", ".docx"}


@dataclass
class Scores:
    tp: Counter[str] = field(default_factory=Counter)
    fp: Counter[str] = field(default_factory=Counter)
    fn: Counter[str] = field(default_factory=Counter)
    headings_matched: int = 0
    headings_predicted: int = 0
    headings_true: int = 0


def main() -> int:
    args = _parse_args()
    if not args.cvs_dir.is_dir():
        print(f"CV folder not found: {args.cvs_dir} (real CVs belong in data/private/cvs/)")
        return 1
    if not args.labels.is_file():
        print(f"Labels file not found: {args.labels} (see scripts/section_labels.example.json)")
        return 1

    labels: dict[str, list[dict]] = json.loads(args.labels.read_text(encoding="utf-8"))
    files = sorted(p for p in args.cvs_dir.iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"No PDF/DOCX files in {args.cvs_dir}")
        return 1

    scores = Scores()
    sources: Counter[str] = Counter()
    fallback_docs: list[str] = []
    other_headings: list[tuple[str, str, list[str]]] = []

    for path in files:
        try:
            sections, warnings = split_sections(parse(path))
        except ParseError as exc:
            print(f"SKIP {path.name}: {exc.reason} ({exc})")
            continue
        sources.update(s.label_source for s in sections)
        if NO_HEADINGS_WARNING in warnings:
            fallback_docs.append(path.name)
        other_headings += [
            (path.name, s.heading, s.types)
            for s in sections
            if s.heading is not None and "other" in s.types
        ]
        if path.name in labels:
            _score(sections, labels[path.name], scores)

    for name in sorted(set(labels) - {p.name for p in files}):
        print(f"WARN labels for {name} but no such file")

    _report(scores, sources, fallback_docs, other_headings, len(files))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cvs-dir", type=Path, default=Path("data/private/cvs"))
    parser.add_argument("--labels", type=Path, default=Path("data/private/section_labels.json"))
    return parser.parse_args()


def _score(sections: list[Section], truth: list[dict], scores: Scores) -> None:
    """Pair predicted and true headings by normalised text, in order, then compare types."""
    predicted = [s for s in sections if s.heading is not None]
    unused = list(predicted)
    scores.headings_predicted += len(predicted)
    scores.headings_true += len(truth)

    for label in truth:
        key = normalize_heading(label["heading"])
        match = next((s for s in unused if normalize_heading(s.heading or "") == key), None)
        true_types = set(label["types"])
        if match is None:
            scores.fn.update(true_types)
            continue
        unused.remove(match)
        scores.headings_matched += 1
        pred_types = set(match.types)
        scores.tp.update(true_types & pred_types)
        scores.fp.update(pred_types - true_types)
        scores.fn.update(true_types - pred_types)

    for section in unused:  # predicted headings that aren't headings in the labels
        scores.fp.update(section.types)


def _report(
    scores: Scores,
    sources: Counter[str],
    fallback_docs: list[str],
    other_headings: list[tuple[str, str, list[str]]],
    n_files: int,
) -> None:
    print(f"\n== Headings ({n_files} files)")
    print(
        f"detected {scores.headings_matched}/{scores.headings_true} labelled headings "
        f"(recall {_ratio(scores.headings_matched, scores.headings_true)}), "
        f"{scores.headings_matched}/{scores.headings_predicted} predicted were real "
        f"(precision {_ratio(scores.headings_matched, scores.headings_predicted)})"
    )

    print("\n== Per type")
    print(f"{'type':<16}{'precision':>10}{'recall':>8}{'tp':>5}{'fp':>5}{'fn':>5}")
    for section_type in sorted(set(scores.tp) | set(scores.fp) | set(scores.fn)):
        tp, fp, fn = scores.tp[section_type], scores.fp[section_type], scores.fn[section_type]
        print(
            f"{section_type:<16}{_ratio(tp, tp + fp):>10}{_ratio(tp, tp + fn):>8}"
            f"{tp:>5}{fp:>5}{fn:>5}"
        )

    print("\n== Sections by label_source")
    for source, count in sources.most_common():
        print(f"{source:<10}{count:>5}")

    print(f"\n== Fallback documents (no headings found): {len(fallback_docs)}")
    for name in fallback_docs:
        print(f"  {name}")

    print(f"\n== Headings labelled 'other' (synonym candidates): {len(other_headings)}")
    for name, heading, types in other_headings:
        print(f"  {name}: {heading!r} -> {types}")


def _ratio(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.2f}" if denominator else "n/a"


if __name__ == "__main__":
    sys.exit(main())
