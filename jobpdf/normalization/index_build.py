"""Build the pgvector skill index from JM-16's taxonomy files (JM-17).

Rows go into a staging table via binary COPY; the staging table is indexed and
then swapped in. Everything runs in one transaction, so a failed or
interrupted build leaves the previous index exactly as it was.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg
from psycopg import sql

from jobpdf.normalization.db import apply_schema
from jobpdf.normalization.embedder import EMBEDDING_DIM, TextEmbedder

DEFAULT_TABLE = "taxonomy_alias"
TAXONOMY_DIR = Path("data/taxonomy")
DEFAULT_LANGUAGES = ("en", "uk")
# Rows embedded and copied per round trip; keeps memory flat and shows progress.
CHUNK_SIZE = 4096
MAINTENANCE_WORK_MEM = "512MB"

# Must match sql/001_taxonomy.sql; a db test compares the two.
_COLUMNS: tuple[tuple[str, str], ...] = (
    ("canonical_id", "text NOT NULL"),
    ("canonical_name", "text NOT NULL"),
    ("alias", "text NOT NULL"),
    ("alias_norm", "text NOT NULL"),
    ("lang", "text NOT NULL"),
    ("alias_kind", "text NOT NULL"),
    ("exact_only", "boolean NOT NULL"),
    ("embedding", f"vector({EMBEDDING_DIM})"),
)
_COPY_TYPES = ["text", "text", "text", "text", "text", "text", "bool", "vector"]


@dataclass(frozen=True)
class AliasRow:
    canonical_id: str
    canonical_name: str
    alias: str
    alias_norm: str
    lang: str
    alias_kind: str
    exact_only: bool


@dataclass(frozen=True)
class BuildSummary:
    table: str
    rows_total: int
    rows_embedded: int
    duration_s: float
    table_bytes: int


def read_rows(
    taxonomy_dir: Path = TAXONOMY_DIR, languages: tuple[str, ...] = DEFAULT_LANGUAGES
) -> list[AliasRow]:
    """aliases.csv rows for the chosen languages, each with its concept's display name."""
    names = canonical_names(taxonomy_dir / "concepts.csv")
    with (taxonomy_dir / "aliases.csv").open(encoding="utf-8", newline="") as f:
        return [
            AliasRow(
                canonical_id=row["canonical_id"],
                canonical_name=names.get(row["canonical_id"], row["canonical_id"]),
                alias=row["alias"],
                alias_norm=row["alias_norm"],
                lang=row["lang"],
                alias_kind=row["alias_kind"],
                exact_only=row["exact_only"] == "true",
            )
            for row in csv.DictReader(f)
            if row["lang"] in languages
        ]


def canonical_names(concepts_csv: Path) -> dict[str, str]:
    """Preferred English name, falling back to Ukrainian, then to the id itself."""
    with concepts_csv.open(encoding="utf-8", newline="") as f:
        return {
            row["canonical_id"]: row["name_en"] or row["name_uk"] or row["canonical_id"]
            for row in csv.DictReader(f)
        }


def file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def build_index(
    conn: psycopg.Connection,
    rows: list[AliasRow],
    embedder: TextEmbedder,
    *,
    table: str = DEFAULT_TABLE,
    meta: dict | None = None,
    progress: Callable[[str], None] = print,
) -> BuildSummary:
    """Embed, load, index and swap in ``table``; write ``meta`` into index_meta.

    ``meta`` is None for experiment tables (the eval's prefix comparison), so
    they can never overwrite the real index's metadata.
    """
    started = time.monotonic()
    apply_schema(conn)
    staging = f"{table}_new"
    with conn.transaction():
        conn.execute(_drop(staging))
        conn.execute(_drop(f"{table}_old"))
        conn.execute(create_table_sql(staging))
        embedded = _copy_rows(conn, staging, rows, embedder, progress)
        progress("building indexes ...")
        conn.execute(sql.SQL("SET LOCAL maintenance_work_mem = {}").format(
            sql.Literal(MAINTENANCE_WORK_MEM)))
        _create_indexes(conn, staging)
        _swap(conn, table)
        if meta is not None:
            _write_meta(conn, meta, rows_total=len(rows), rows_embedded=embedded,
                        duration_s=time.monotonic() - started)
    size = conn.execute(
        "SELECT pg_total_relation_size(%s::regclass)", [table]
    ).fetchone()[0]
    return BuildSummary(table, len(rows), embedded, round(time.monotonic() - started, 1), size)


def create_table_sql(table: str) -> sql.Composed:
    columns = sql.SQL(", ").join(
        sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(decl)) for name, decl in _COLUMNS
    )
    return sql.SQL("CREATE TABLE {} (id bigserial PRIMARY KEY, {})").format(
        sql.Identifier(table), columns
    )


def _drop(table: str) -> sql.Composed:
    return sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table))


def _copy_rows(
    conn: psycopg.Connection,
    table: str,
    rows: list[AliasRow],
    embedder: TextEmbedder,
    progress: Callable[[str], None],
) -> int:
    columns = sql.SQL(", ").join(sql.Identifier(name) for name, _ in _COLUMNS)
    statement = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT BINARY)").format(
        sql.Identifier(table), columns
    )
    embedded = done = 0
    with conn.cursor() as cur, cur.copy(statement) as copy:
        copy.set_types(_COPY_TYPES)
        for chunk in _chunks(rows, CHUNK_SIZE):
            vectors = _embed_chunk(chunk, embedder)
            for row, vector in zip(chunk, vectors, strict=True):
                copy.write_row([
                    row.canonical_id, row.canonical_name, row.alias, row.alias_norm,
                    row.lang, row.alias_kind, row.exact_only, vector,
                ])
            embedded += sum(v is not None for v in vectors)
            done += len(chunk)
            progress(f"  {done}/{len(rows)} rows")
    return embedded


def _embed_chunk(chunk: list[AliasRow], embedder: TextEmbedder) -> list[np.ndarray | None]:
    """Vectors for embeddable rows, None for exact_only ones (stored as NULL)."""
    texts = [row.alias for row in chunk if not row.exact_only]
    vectors = iter(embedder.embed(texts))
    return [None if row.exact_only else next(vectors) for row in chunk]


def _chunks(rows: list[AliasRow], size: int) -> Iterator[list[AliasRow]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _create_indexes(conn: psycopg.Connection, table: str) -> None:
    conn.execute(sql.SQL("CREATE INDEX {} ON {} (alias_norm)").format(
        sql.Identifier(f"{table}_norm_idx"), sql.Identifier(table)))
    conn.execute(sql.SQL("CREATE INDEX {} ON {} USING hnsw (embedding vector_cosine_ops)").format(
        sql.Identifier(f"{table}_hnsw_idx"), sql.Identifier(table)))


def _swap(conn: psycopg.Connection, table: str) -> None:
    """Rename current -> _old, staging -> current (indexes, pkey, sequence too), drop old.

    Renaming everything keeps object names identical across rebuilds, so
    sql/001's ``IF NOT EXISTS`` checks and any monitoring keep working.
    """
    new, old = f"{table}_new", f"{table}_old"
    exists = conn.execute("SELECT to_regclass(%s) IS NOT NULL", [table]).fetchone()[0]
    if exists:
        _rename_all(conn, table, old)
    _rename_all(conn, new, table)
    conn.execute(_drop(old))


def _rename_all(conn: psycopg.Connection, source: str, target: str) -> None:
    ident = sql.Identifier
    conn.execute(sql.SQL("ALTER TABLE {} RENAME TO {}").format(ident(source), ident(target)))
    conn.execute(sql.SQL("ALTER TABLE {} RENAME CONSTRAINT {} TO {}").format(
        ident(target), ident(f"{source}_pkey"), ident(f"{target}_pkey")))
    for suffix in ("norm_idx", "hnsw_idx"):
        conn.execute(sql.SQL("ALTER INDEX IF EXISTS {} RENAME TO {}").format(
            ident(f"{source}_{suffix}"), ident(f"{target}_{suffix}")))
    conn.execute(sql.SQL("ALTER SEQUENCE IF EXISTS {} RENAME TO {}").format(
        ident(f"{source}_id_seq"), ident(f"{target}_id_seq")))


def _write_meta(
    conn: psycopg.Connection, meta: dict, *, rows_total: int, rows_embedded: int,
    duration_s: float,
) -> None:
    values = dict(meta)
    values["build"] = {
        **values.get("build", {}),
        "rows_total": rows_total,
        "rows_embedded": rows_embedded,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(duration_s, 1),
    }
    for key, value in values.items():
        conn.execute(
            "INSERT INTO index_meta (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            [key, json.dumps(value)],
        )
    # A threshold measured on the previous vectors doesn't describe the new ones.
    conn.execute("DELETE FROM index_meta WHERE key = 'thresholds'")


def standard_meta(
    *, model_name: str, revision: str, prefix: str, languages: tuple[str, ...],
    taxonomy_dir: Path = TAXONOMY_DIR,
) -> dict:
    """The index_meta keys written by a normal build (thresholds come from eval)."""
    manifest_path = taxonomy_dir / "manifest.json"
    try:
        esco_version = json.loads(manifest_path.read_text(encoding="utf-8")).get("esco_version")
    except (OSError, ValueError):
        esco_version = None
    return {
        "model": {"name": model_name, "revision": revision, "prefix": prefix,
                  "dim": EMBEDDING_DIM},
        "build": {"languages": list(languages)},
        "taxonomy": {"manifest_sha256": file_sha256(manifest_path),
                     "esco_version": esco_version},
    }
