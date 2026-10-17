"""Store the cached postings as vacancies, once, by hand (JM-33).

Reads what the fetch jobs cached (it never fetches) and writes new, changed and
out-of-date postings to the database; everything else is only marked as seen. Needs
DATABASE_URL, the skill index and the embedding model, as the API does. Prints counts and
codes only.

Usage (from the repo root):
    uv run python scripts/ingest_vacancies.py              # every source
    uv run python scripts/ingest_vacancies.py remotive adzuna
"""

from __future__ import annotations

import sys

from jobpdf.ingestion.vacancies import SOURCES, IngestError, run_ingest


def main(names: list[str]) -> int:
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        print(f"unknown source(s): {', '.join(unknown)}; choose from {', '.join(SOURCES)}")
        return 2
    try:
        rows, problem = run_ingest(names or None)["sources"], None
    except IngestError as exc:
        rows, problem = exc.results, str(exc)
    for row in rows:
        if row["status"] == "no_cache":
            print(f"{row['source']}: nothing cached (run the fetch first)")
        elif row["status"] == "failed":
            print(f"{row['source']}: FAILED ({row['code']})")
        else:
            print(f"{row['source']}: {row['received']} received, {row['new']} new, "
                  f"{row['changed']} changed, {row['stale']} stale, {row['unchanged']} unchanged; "
                  f"stored {row['inserted']} new + {row['updated']} updated, "
                  f"failed {sum(row['failed'].values())}, deferred {row['deferred']}"
                  + ("; ABORTED after repeated failures" if row["status"] == "aborted" else ""))
    if problem:
        print(problem)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
