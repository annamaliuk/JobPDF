"""Fetch Remotive postings once, by hand (JM-31).

Uses the same cache and request guard as the Airflow job, so running it cannot overspend
Remotive's limit: within 6 hours of the last fetch it makes no request at all. Prints
counts only, never postings.

Usage (from the repo root):
    uv run python scripts/fetch_remotive.py
"""

from __future__ import annotations

import sys

from jobpdf.ingestion.remotive import MAX_CACHE_AGE_S, FetchError, fetch_postings


def main() -> int:
    try:
        result = fetch_postings()
    except FetchError as exc:
        print(f"no postings: {exc.code} (and no cache to fall back on)")
        return 1
    source = "cache" if result.from_cache else "live request"
    when = f"{result.fetched_at:%Y-%m-%d %H:%M} UTC"
    print(f"{len(result.postings)} postings from {source}, fetched {when}")
    print(f"skipped {result.skipped} malformed; feed reported {result.reported_count}")
    if result.stale:
        print("WARNING: could not refresh; this is older data "
              f"(cache is normally reused for {MAX_CACHE_AGE_S // 3600} hours)")
    for code in result.warnings:
        print(f"warning: {code}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
