"""Human check of requirement extraction on real cached postings (JM-48).

Usage (from the repo root, after a fetch has filled data/cache/):
    uv run python scripts/check_vacancy_requirements.py              # 5 per source
    uv run python scripts/check_vacancy_requirements.py --per-source 10

Reads the fetch cache only (never fetches) and writes a Markdown sheet to
data/eval/vacancy_requirements_check.md (git-ignored): for each posting its link, what the
rules found, and the line each item came from, with a column to mark it right or wrong.
Misses are candidates for new word-list entries in jobpdf/ingestion/requirement_rules.py or
new rows in data/taxonomy/custom_skills.csv.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from jobpdf.ingestion.requirements import Requirements, RulesExtractor
from jobpdf.ingestion.sources.models import SourcePosting
from jobpdf.ingestion.vacancies import load_cached_postings

DEFAULT_SOURCES = ("remotive", "arbeitnow", "themuse")
DEFAULT_OUT = Path("data/eval/vacancy_requirements_check.md")


def _cell(text: str) -> str:
    return " ".join(text.replace("|", "/").split())


def _quote(result: Requirements, key: str) -> str:
    return _cell(result.quotes.get(key, ""))


def render(posting: SourcePosting, result: Requirements) -> list[str]:
    rows = [(f"required: {s}", _quote(result, s)) for s in result.required_skills]
    rows += [(f"nice to have: {s}", _quote(result, s)) for s in result.preferred_skills]
    if result.min_years_experience is not None:
        years = result.min_years_experience
        rows.append((f"min years: {years}", _quote(result, f"years:{years}")))
    if result.education is not None:
        field = f" ({result.education.field})" if result.education.field else ""
        rows.append((f"education: {result.education.level}{field}",
                     _cell(result.education.quote or "")))
    rows += [(f"language: {name}", _quote(result, f"language:{name}"))
             for name in result.languages or []]
    rows += [("responsibility", _cell(r)) for r in result.responsibilities]
    lines = [f"## [{_cell(posting.title)}]({posting.url})", "",
             f"{posting.source} · warnings: {', '.join(result.warnings) or 'none'}", "",
             "| Found | From the line | Right? |", "| --- | --- | --- |"]
    lines += [f"| {_cell(found)} | {quote} | |" for found, quote in rows] or ["| nothing | | |"]
    return [*lines, ""]


def main() -> int:
    parser = argparse.ArgumentParser(description="Requirement extraction, for a human check.")
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--per-source", type=int, default=5)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    extractor = RulesExtractor()
    if not extractor.vocabulary.complete:
        print("WARNING: data/taxonomy/aliases.csv not found; only the custom skills are known. "
              "Run scripts/build_taxonomy.py first for a fair check.")
    sections: list[str] = ["# Vacancy requirement check (JM-48)", ""]
    warnings: Counter[str] = Counter()
    checked = 0
    for source in args.sources:
        cached = load_cached_postings(source)
        if cached is None:
            print(f"{source}: no cached fetch, skipped")
            continue
        postings = [p for p in cached[0] if p.description][: args.per_source]
        for posting in postings:
            result = extractor.extract(posting)
            warnings.update(result.warnings)
            sections += render(posting, result)
            checked += 1
        print(f"{source}: {len(postings)} postings")
    if not checked:
        print("ERROR: no cached postings with a description; run a fetch first.")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {checked} postings to {args.out}; warnings: {dict(warnings) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
