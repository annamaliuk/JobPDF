"""Versioned SQL migrations (JM-38): one place that creates and changes the database schema.

Until now every module applied its own ``sql/NNN_*.sql`` file ("idempotent, safe to run on
every connect"): the index build, the rate limiter, the ingestion run. That works for
``CREATE TABLE IF NOT EXISTS`` but not for a change to an existing table, and two processes
starting together (the app and an Airflow task on a fresh database) can race on the same
``CREATE TABLE``. This runner applies the same files once, in order, and remembers it.

How it works.

- A migration is a file ``sql/<version>_<name>.sql``: three or more digits, an underscore, and
  a lowercase name. Versions are compared as numbers; gaps are fine (a branch may add 002 after
  003 exists). Any other ``.sql`` file in the folder is an error, so a typo is not silently
  skipped. The existing files are migrations 001, 002, 003 and 004 as they are.
- Applied versions are recorded in ``schema_migrations`` with the file's checksum. Each
  migration runs in its own transaction together with its record: it is applied completely or
  not at all, and a failure leaves the earlier ones in place. (So no ``CREATE INDEX
  CONCURRENTLY`` or other statement that refuses to run in a transaction.)
- A run holds a Postgres advisory lock for its whole length, so concurrent runners take turns
  and each migration is applied exactly once.
- Once applied, a file must not change: a different checksum stops the run
  (``checksum_mismatch``). Change the schema with a new migration. Line endings are ignored in
  the checksum, so a Windows checkout does not look like an edit.
- A recorded version whose file is gone also stops the run (``missing_file``): the code and the
  database no longer agree.
- A pending migration older than an applied one is still applied (merge order is not release
  order); a run reports it by listing the versions it applied.

Privacy. Messages and logs carry versions, names, counts and codes. A database error is
replaced by ``migration_failed`` and the exception's type, never its text.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import psycopg

from jobpdf.normalization.db import REPO_ROOT

log = logging.getLogger(__name__)

SQL_DIR = REPO_ROOT / "sql"
MIGRATION_NAME = re.compile(r"^(\d{3,})_([a-z0-9_]+)\.sql$")
TABLE = "schema_migrations"
ADVISORY_LOCK_KEY = 0x6A6F6270  # "jobp"
LOCK_TIMEOUT_MS = 60_000

# Error codes (MigrationError.code).
NO_DIRECTORY = "no_directory"
BAD_NAME = "bad_name"
DUPLICATE_VERSION = "duplicate_version"
CHECKSUM_MISMATCH = "checksum_mismatch"
MISSING_FILE = "missing_file"
MIGRATION_FAILED = "migration_failed"
LOCK_TIMEOUT = "lock_timeout"

# Statuses (MigrationStatus.state).
APPLIED = "applied"
PENDING = "pending"
CHANGED = "changed"  # applied, but the file is different now
MISSING = "missing"  # applied, but there is no file any more

_CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    version    text PRIMARY KEY,
    name       text NOT NULL,
    checksum   text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
)"""


class MigrationError(Exception):
    """The run stopped. ``code`` is stable; ``version`` is the migration it concerns, if one."""

    def __init__(self, code: str, version: str | None = None) -> None:
        super().__init__(f"{code} {version}" if version else code)
        self.code = code
        self.version = version


@dataclass(frozen=True)
class Migration:
    version: str
    name: str
    path: Path
    checksum: str

    @property
    def label(self) -> str:
        return f"{self.version}_{self.name}"


@dataclass(frozen=True)
class MigrationStatus:
    version: str
    name: str
    state: str


@dataclass(frozen=True)
class MigrationResult:
    applied: tuple[str, ...]  # versions applied by this run, in order
    already: tuple[str, ...]  # versions that were applied before


def checksum(text: str) -> str:
    """SHA-256 of the file's text with line endings normalised to ``\\n``."""
    normal = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normal.encode("utf-8")).hexdigest()


def discover(directory: Path = SQL_DIR) -> list[Migration]:
    """The migration files, oldest first. Raises ``MigrationError`` for a missing folder, a
    ``.sql`` file with a wrong name, or two files with the same version number."""
    directory = Path(directory)
    if not directory.is_dir():
        raise MigrationError(NO_DIRECTORY)
    found: dict[int, Migration] = {}
    for path in sorted(directory.glob("*.sql")):
        match = MIGRATION_NAME.match(path.name)
        if not match:
            raise MigrationError(BAD_NAME, path.name)
        version, name = match.groups()
        number = int(version)
        if number in found:
            raise MigrationError(DUPLICATE_VERSION, version)
        found[number] = Migration(version, name, path, checksum(path.read_text(encoding="utf-8")))
    return [found[number] for number in sorted(found)]


def status(conn: psycopg.Connection, directory: Path = SQL_DIR) -> list[MigrationStatus]:
    """Every known migration with its state, oldest first. Changes nothing: it does not take
    the lock and does not create the ``schema_migrations`` table."""
    files = {m.version: m for m in discover(directory)}
    recorded = _recorded(conn)
    rows = [
        MigrationStatus(m.version, m.name, _state(m, recorded.get(m.version)))
        for m in files.values()
    ]
    rows += [
        MigrationStatus(version, name, MISSING)
        for version, (name, _) in recorded.items() if version not in files
    ]
    return sorted(rows, key=lambda row: int(row.version))


def pending(conn: psycopg.Connection, directory: Path = SQL_DIR) -> list[str]:
    """Versions that ``migrate`` would apply."""
    return [row.version for row in status(conn, directory) if row.state == PENDING]


def migrate(conn: psycopg.Connection, directory: Path = SQL_DIR, *,
            lock_timeout_ms: int = LOCK_TIMEOUT_MS) -> MigrationResult:
    """Apply every pending migration, oldest first, and return what was done.

    Raises ``MigrationError``. Migrations applied before a failure stay applied.
    """
    files = discover(directory)
    _lock(conn, lock_timeout_ms)
    try:
        conn.execute(_CREATE_TABLE)
        conn.commit()
        recorded = _recorded(conn)
        _check(files, recorded)
        applied: list[str] = []
        for migration in files:
            if migration.version in recorded:
                continue
            _apply(conn, migration)
            applied.append(migration.version)
        return MigrationResult(
            applied=tuple(applied), already=tuple(m.version for m in files if m.version in recorded)
        )
    finally:
        _unlock(conn)


# --- internals --------------------------------------------------------------------------


def _state(migration: Migration, recorded: tuple[str, str] | None) -> str:
    if recorded is None:
        return PENDING
    return APPLIED if recorded[1] == migration.checksum else CHANGED


def _recorded(conn: psycopg.Connection) -> dict[str, tuple[str, str]]:
    """version -> (name, checksum) of what the database says it has applied."""
    try:
        rows = conn.execute(f"SELECT version, name, checksum FROM {TABLE}").fetchall()
    except psycopg.errors.UndefinedTable:
        rows = []
    finally:
        conn.rollback()
    return {version: (name, digest) for version, name, digest in rows}


def _check(files: Sequence[Migration], recorded: dict[str, tuple[str, str]]) -> None:
    """Refuse to go on when the files and the record disagree about what was applied."""
    known = {m.version: m for m in files}
    for version in sorted(recorded, key=int):
        migration = known.get(version)
        if migration is None:
            raise MigrationError(MISSING_FILE, version)
        if recorded[version][1] != migration.checksum:
            raise MigrationError(CHECKSUM_MISMATCH, version)


def _apply(conn: psycopg.Connection, migration: Migration) -> None:
    sql = migration.path.read_text(encoding="utf-8")
    try:
        conn.execute(sql)
        conn.execute(
            f"INSERT INTO {TABLE} (version, name, checksum) VALUES (%s, %s, %s)",
            (migration.version, migration.name, migration.checksum),
        )
        conn.commit()
    except psycopg.Error as exc:  # the caller's finally rolls the connection back
        log.error("migration %s failed (%s)", migration.label, type(exc).__name__)
        raise MigrationError(MIGRATION_FAILED, migration.version) from None
    log.info("migration %s applied", migration.label)


def _lock(conn: psycopg.Connection, timeout_ms: int) -> None:
    """Take the session-level advisory lock; wait at most ``timeout_ms`` for it."""
    try:
        conn.execute(f"SET LOCAL lock_timeout = {int(timeout_ms)}")  # ends with the commit below
        conn.execute("SELECT pg_advisory_lock(%s)", (ADVISORY_LOCK_KEY,))
        conn.commit()
    except psycopg.errors.LockNotAvailable:
        conn.rollback()
        raise MigrationError(LOCK_TIMEOUT) from None


def _unlock(conn: psycopg.Connection) -> None:
    try:
        conn.rollback()
        conn.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_KEY,))
        conn.commit()
    except psycopg.Error:
        # e.g. the connection died: the server released the lock with the session
        log.warning("could not release the migration lock; it ends with the connection")
