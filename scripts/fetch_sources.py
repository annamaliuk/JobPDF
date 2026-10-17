"""Fetch job postings from The Muse, Arbeitnow, Adzuna and Jooble once, by hand.

Uses the same cache and request guards as the Airflow job, so running it cannot overspend a
source's limit. Keys come from the environment or ``.env``. Prints counts and codes only:
never postings, URLs or keys.

Usage (from the repo root):
    uv run python scripts/fetch_sources.py              # all four
    uv run python scripts/fetch_sources.py adzuna jooble
"""

from __future__ import annotations

import sys

from jobpdf.ingestion.sources.models import FetchError
from jobpdf.ingestion.sources.registry import SOURCE_NAMES, build_adapter
from jobpdf.ingestion.sources.runner import fetch_source


def main(names: list[str]) -> int:
    unknown = [n for n in names if n not in SOURCE_NAMES]
    if unknown:
        print(f"unknown source(s): {', '.join(unknown)}; choose from {', '.join(SOURCE_NAMES)}")
        return 2
    failed = 0
    for name in names or SOURCE_NAMES:
        try:
            result = fetch_source(build_adapter(name))
        except FetchError as exc:
            print(f"{name}: no postings ({exc.code})")
            failed += exc.code != "not_configured"
            continue
        origin = "cache" if result.from_cache else "live"
        print(f"{name}: {len(result.postings)} postings from {origin} "
              f"({result.pages} pages, {result.skipped} skipped, {result.filtered} filtered)"
              + (f"; warnings: {', '.join(result.warnings)}" if result.warnings else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
