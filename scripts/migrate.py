"""Apply the database migrations in sql/ (JM-38).

Run it after the database is up and before the app or Airflow start (docker compose does this
for you with the ``migrate`` service). It is safe to run any number of times, also at the same
time from several places: applied migrations are skipped and a lock makes runs take turns.

Usage (from the repo root):
    uv run python scripts/migrate.py            # apply what is pending
    uv run python scripts/migrate.py --status   # list every migration and its state
    uv run python scripts/migrate.py --check    # change nothing; exit 1 unless all are applied

Reads DATABASE_URL from the environment or .env. Prints versions, names, counts and codes,
never SQL, addresses or passwords. Exit code: 0 done, 1 failed (or --check found a difference),
2 not configured.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import psycopg

from jobpdf.migrations import APPLIED, SQL_DIR, MigrationError, migrate, status
from jobpdf.normalization.db import database_url

CONNECT_TIMEOUT_S = 10


def _connect(url: str) -> psycopg.Connection:
    return psycopg.connect(url, connect_timeout=CONNECT_TIMEOUT_S)


def main(argv: list[str] | None = None, *,
         connect: Callable[[str], psycopg.Connection] = _connect) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true", help="list the migrations and exit")
    mode.add_argument("--check", action="store_true",
                      help="exit 1 unless every migration is applied")
    parser.add_argument("--dir", type=Path, default=SQL_DIR, help="folder with the .sql files")
    args = parser.parse_args(argv)

    url = database_url()
    if not url:
        print("DATABASE_URL is not set (environment or .env)")
        return 2
    try:
        conn = connect(url)
    except psycopg.Error:
        print("database unreachable")
        return 1
    try:
        with conn:
            if args.status or args.check:
                rows = status(conn, args.dir)
                for row in rows:
                    print(f"{row.version}  {row.name:<28} {row.state}")
                if args.check and any(row.state != APPLIED for row in rows):
                    return 1
                return 0
            result = migrate(conn, args.dir)
    except MigrationError as exc:
        print(f"migration stopped: {exc}")
        return 1
    except psycopg.Error:
        print("database error")
        return 1
    for version in result.applied:
        print(f"applied {version}")
    print(f"{len(result.applied)} applied, {len(result.already)} already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
