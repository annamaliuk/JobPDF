"""Embed data/taxonomy/aliases.csv into Postgres + pgvector (JM-17).

Usage (from the repo root, after `docker compose up -d` and build_taxonomy.py):
    uv run python scripts/build_index.py
    uv run python scripts/build_index.py --languages en
--prefix and --table exist only for eval_index.py's prefix comparison; a
non-default --table never writes index_meta.
"""

from __future__ import annotations

import argparse
import sys

from jobpdf.normalization.db import connect, database_url
from jobpdf.normalization.embedder import MODEL_NAME, MODEL_REVISION, QUERY_PREFIX, Embedder
from jobpdf.normalization.index_build import (
    DEFAULT_LANGUAGES,
    DEFAULT_TABLE,
    TAXONOMY_DIR,
    BuildSummary,
    build_index,
    read_rows,
    standard_meta,
)


def run(languages: tuple[str, ...], prefix: str, table: str) -> BuildSummary:
    """Shared with eval_index.py, which builds the 'passage: ' comparison table."""
    url = database_url()
    if not url:
        raise SystemExit("DATABASE_URL is not set (environment or .env); see .env.example")
    if not (TAXONOMY_DIR / "aliases.csv").is_file():
        raise SystemExit("data/taxonomy/aliases.csv missing; run scripts/build_taxonomy.py first")

    rows = read_rows(TAXONOMY_DIR, languages)
    print(f"{len(rows)} aliases ({', '.join(languages)}), "
          f"{sum(not r.exact_only for r in rows)} to embed with prefix {prefix!r} -> {table}")
    meta = None
    if table == DEFAULT_TABLE:
        meta = standard_meta(model_name=MODEL_NAME, revision=MODEL_REVISION, prefix=prefix,
                             languages=languages)
    with connect(url) as conn:
        summary = build_index(conn, rows, Embedder(prefix=prefix), table=table, meta=meta)
    print(f"\n{summary.table}: {summary.rows_total} rows ({summary.rows_embedded} embedded) "
          f"in {summary.duration_s:.0f}s, {summary.table_bytes / 1e6:.0f} MB on disk")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the pgvector skill index.")
    parser.add_argument("--languages", default=",".join(DEFAULT_LANGUAGES),
                        help="comma-separated, default en,uk")
    parser.add_argument("--prefix", default=QUERY_PREFIX)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    args = parser.parse_args()
    languages = tuple(lang.strip() for lang in args.languages.split(",") if lang.strip())
    run(languages, args.prefix, args.table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
