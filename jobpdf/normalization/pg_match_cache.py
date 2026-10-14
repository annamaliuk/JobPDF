"""Postgres skill match cache (JM-19), behind JM-18's ``SkillMatchCache`` interface.

The API and the Airflow job are separate processes, so only a shared store makes
"identical strings get identical IDs" hold across the CV side and the vacancy side.
Keys come from ``match_cache.cache_key()``: index version, prompt version and the
``normalize()``d skill, so a rebuilt index or a changed prompt never serves a stale
match.

A cache is an optimisation, never a dependency. A database outage degrades to
misses (``get``) and no-ops (``set``), and after a connection failure the database
is left alone for ``RETRY_AFTER_S`` seconds. Without that pause every skill of a CV
would wait out the connect timeout, which alone would break the 30 s target.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import psycopg
from pydantic import ValidationError

from jobpdf.normalization.db import CONNECT_TIMEOUT_S, REPO_ROOT, apply_schema, database_url
from jobpdf.normalization.match_cache import (
    InMemorySkillMatchCache,
    MatcherVersions,
    SkillMatch,
    SkillMatchCache,
    cache_key,
)

log = logging.getLogger(__name__)

CACHE_SCHEMA_FILE = REPO_ROOT / "sql" / "002_match_cache.sql"
# After a lost connection, skip the database this long before trying to reconnect.
RETRY_AFTER_S = 30.0

_SELECT = "SELECT match FROM skill_match_cache WHERE cache_key = %s"
_UPSERT = (
    "INSERT INTO skill_match_cache (cache_key, match) VALUES (%s, %s::jsonb) "
    "ON CONFLICT (cache_key) DO UPDATE SET match = EXCLUDED.match, created_at = now()"
)
_PRUNE = "DELETE FROM skill_match_cache WHERE left(cache_key, %s) <> %s RETURNING 1"
_COUNT = "SELECT count(*) FROM skill_match_cache"


class PostgresSkillMatchCache:
    """``SkillMatchCache`` stored in the ``skill_match_cache`` table.

    Raises ``psycopg.Error`` if the database is unreachable at construction, so a
    misconfiguration shows up at startup (``default_match_cache`` turns that into a
    fallback). After construction nothing raises for connection problems.

    One autocommit connection, serialized by a lock, like ``SkillIndex``. Autocommit
    matters: a failed statement must not leave an aborted transaction behind that
    would break every later call. Extra keyword arguments go to ``psycopg.connect``
    (tests pass ``options`` to pin a throwaway ``search_path``).
    """

    def __init__(
        self,
        url: str,
        *,
        retry_after_s: float = RETRY_AFTER_S,
        clock: Callable[[], float] = time.monotonic,
        **connect_kwargs: object,
    ) -> None:
        self._url = url
        self._connect_kwargs = connect_kwargs
        self._retry_after_s = retry_after_s
        self._clock = clock
        self._lock = threading.Lock()
        self._conn: psycopg.Connection | None = None
        self._down_until = 0.0
        self._open()

    # --- SkillMatchCache ----------------------------------------------------------

    def get(self, key: str) -> SkillMatch | None:
        rows = self._execute(_SELECT, [key])
        if not rows:
            return None
        try:
            return SkillMatch.model_validate(rows[0][0])
        except ValidationError:
            # A row written by an older SkillMatch shape: recompute rather than fail.
            log.warning("skill match cache row failed validation; treated as a miss")
            return None

    def set(self, key: str, match: SkillMatch) -> None:
        self._execute(_UPSERT, [key, match.model_dump_json()])

    # --- housekeeping -------------------------------------------------------------

    def prune_other_versions(self, versions: MatcherVersions) -> int:
        """Delete every row not written under ``versions``; returns how many.

        The index version changes on every rebuild, so without this the table keeps
        a dead generation of rows per rebuild. Call it after building the index.
        """
        prefix = cache_key("", versions)  # "<index>|<prompt>|", the one key format
        rows = self._execute(_PRUNE, [len(prefix), prefix])
        return len(rows) if rows else 0

    def __len__(self) -> int:
        rows = self._execute(_COUNT, [])
        return int(rows[0][0]) if rows else 0

    def close(self) -> None:
        with self._lock:
            conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()

    # --- connection handling ------------------------------------------------------

    def _open(self) -> psycopg.Connection:
        kwargs = {"connect_timeout": CONNECT_TIMEOUT_S, **self._connect_kwargs}
        conn = psycopg.connect(self._url, autocommit=True, **kwargs)  # type: ignore[arg-type]
        try:
            apply_schema(conn, CACHE_SCHEMA_FILE)  # idempotent CREATE TABLE IF NOT EXISTS
        except psycopg.Error:
            conn.close()
            raise
        self._conn = conn
        return conn

    def _execute(self, sql: str, params: list[object]) -> list[tuple] | None:
        """Run one statement; None means the cache was unavailable for this call."""
        with self._lock:
            if self._clock() < self._down_until:
                return None
            conn = self._conn
            try:
                if conn is None or _unusable(conn):
                    conn = self._open()
                cursor = conn.execute(sql, params)  # type: ignore[arg-type]
                return cursor.fetchall() if cursor.description else []
            except psycopg.Error as exc:
                # Judge by the connection's state, not the exception class: psycopg
                # reports some per-statement failures (an oversized index key) as
                # OperationalError too, and those must not pause the whole cache.
                if conn is not None and not _unusable(conn):
                    log.warning("skill match cache statement failed (%s)", type(exc).__name__)
                    return None
                # The connection is gone, or never came up. The URL is never logged:
                # it carries the database password.
                log.warning("skill match cache database unavailable (%s); skipping it for "
                            "%.0f s", type(exc).__name__, self._retry_after_s)
                self._down_until = self._clock() + self._retry_after_s
                self._discard_connection()
                return None

    def _discard_connection(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except psycopg.Error:
                pass


def _unusable(conn: psycopg.Connection) -> bool:
    return conn.closed or conn.broken


def default_match_cache() -> SkillMatchCache:
    """The persistent cache when ``DATABASE_URL`` works, else the in-memory one.

    Meant for ``build_normalization_service(cache=default_match_cache())``. Never
    raises: a missing or unreachable database logs a warning and degrades to a
    process-local cache, which is correct but not shared between processes.
    """
    url = database_url()
    if not url:
        log.warning("DATABASE_URL missing; using the in-memory skill match cache")
        return InMemorySkillMatchCache()
    try:
        return PostgresSkillMatchCache(url)
    except psycopg.Error as exc:
        log.warning("skill match cache database unreachable (%s); using the in-memory cache",
                    type(exc).__name__)
        return InMemorySkillMatchCache()
