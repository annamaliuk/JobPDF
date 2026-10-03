"""Check taxonomy coverage of the top Remotive software-dev job tags (JM-16).

Usage (from the repo root, after build_taxonomy.py):
    uv run python scripts/taxonomy_coverage.py            # uses the cached response if present
    uv run python scripts/taxonomy_coverage.py --refresh  # refetch (Remotive: max 4 calls/day)

Misses are candidates for new rows in data/taxonomy/custom_skills.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jobpdf.normalization.coverage import (
    DEFAULT_ALIASES,
    DEFAULT_CACHE,
    TOP_N,
    load_alias_norms,
    load_jobs,
    measure,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Taxonomy coverage of Remotive job tags.")
    parser.add_argument("--refresh", action="store_true", help="refetch instead of using cache")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--top", type=int, default=TOP_N)
    args = parser.parse_args()

    if not args.aliases.is_file():
        print(f"ERROR: {args.aliases} not found; run scripts/build_taxonomy.py first.")
        return 1
    if args.refresh or not args.cache.is_file():
        print("Fetching Remotive software-dev jobs (counts toward the 4 calls/day limit)...")
    try:
        payload = load_jobs(args.cache, refresh=args.refresh)
    except OSError as exc:
        print(f"ERROR: could not fetch Remotive jobs: {exc}")
        return 1

    report = measure(payload, load_alias_norms(args.aliases), args.top)
    total = len(report.covered) + len(report.misses)
    print(f"\n{report.jobs} jobs, top {total} tags (source: Remotive, https://remotive.com)")
    print(f"coverage: {report.coverage:.1%} of distinct tags "
          f"({len(report.covered)}/{total}), {report.weighted_coverage:.1%} of tag occurrences")
    print(f"\nmisses ({len(report.misses)}), most frequent first:")
    for tag in report.misses:
        print(f"  {tag.count:4}  {tag.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
