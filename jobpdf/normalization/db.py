"""Postgres connection helpers for the skill index (JM-17)."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"
SCHEMA_FILE = REPO_ROOT / "sql" / "001_taxonomy.sql"
# psycopg waits forever by default; on Windows a closed port can time out rather
# than refuse, which made a stopped container look like a hung test run.
CONNECT_TIMEOUT_S = 10


def database_url(env_file: Path = ENV_FILE) -> str | None:
    """DATABASE_URL from the environment, else from the repo's .env file.

    A tiny reader instead of python-dotenv: only KEY=VALUE lines are needed, and
    a real environment variable always wins over the file.
    """
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        key, sep, value = line.strip().partition("=")
        if sep and key.strip() == "DATABASE_URL" and not line.lstrip().startswith("#"):
            return value.strip().strip("'\"") or None
    return None


def connect(url: str, **kwargs: object) -> psycopg.Connection:
    """Open a connection that understands the pgvector ``vector`` type.

    Extra keyword arguments go to ``psycopg.connect`` (tests pass ``options`` to
    pin a throwaway ``search_path``).
    """
    kwargs.setdefault("connect_timeout", CONNECT_TIMEOUT_S)
    conn = psycopg.connect(url, **kwargs)  # type: ignore[arg-type]
    apply_extension(conn)
    register_vector(conn)
    return conn


def apply_extension(conn: psycopg.Connection) -> None:
    # register_vector needs the type to exist, so create it before anything else.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()


def apply_schema(conn: psycopg.Connection, schema_file: Path = SCHEMA_FILE) -> None:
    conn.execute(schema_file.read_text(encoding="utf-8"))
    conn.commit()
