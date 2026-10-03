"""SkillIndex: candidate concepts for a raw skill string (JM-17).

JM-17 only *finds* candidates (exact alias lookup + vector search). Deciding
which candidate is right is JM-18's job.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import numpy as np
import psycopg
from psycopg import sql

from jobpdf.normalization.db import connect as db_connect
from jobpdf.normalization.embedder import Embedder, TextEmbedder
from jobpdf.normalization.index_build import DEFAULT_TABLE, TAXONOMY_DIR, file_sha256
from jobpdf.normalization.text import normalize

log = logging.getLogger(__name__)

# HNSW returns at most ef_search rows per query, and search() fetches k * FETCH_FACTOR.
EF_SEARCH = 100
# Several aliases of one concept often rank together; over-fetch, then dedup by concept.
FETCH_FACTOR = 3
DEFAULT_MANIFEST = TAXONOMY_DIR / "manifest.json"
TAXONOMY_CHANGED = "taxonomy changed since the index was built; rerun build_index.py"


@dataclass(frozen=True)
class Candidate:
    canonical_id: str
    canonical_name: str
    matched_alias: str
    distance: float  # cosine distance from pgvector's <=>, 0 = identical


def dedupe_candidates(candidates: Iterable[Candidate], k: int | None = None) -> list[Candidate]:
    """One candidate per concept (its closest alias), sorted by distance, top k."""
    best: dict[str, Candidate] = {}
    for candidate in candidates:
        current = best.get(candidate.canonical_id)
        if current is None or candidate.distance < current.distance:
            best[candidate.canonical_id] = candidate
    ranked = sorted(best.values(), key=lambda c: (c.distance, c.canonical_id))
    return ranked if k is None else ranked[:k]


def manifest_warning(meta: dict, manifest_path: Path) -> str | None:
    """A warning if the taxonomy files no longer match what the index was built from."""
    built_from = meta.get("taxonomy", {}).get("manifest_sha256")
    current = file_sha256(manifest_path)
    if current is None:
        return f"{manifest_path} not found; cannot check the index is current ({TAXONOMY_CHANGED})"
    if built_from != current:
        return TAXONOMY_CHANGED
    return None


class SkillIndex:
    """Exact and vector lookups over the taxonomy_alias table."""

    def __init__(
        self,
        conn: psycopg.Connection,
        embedder: TextEmbedder | None = None,
        *,
        table: str = DEFAULT_TABLE,
        manifest_path: Path = DEFAULT_MANIFEST,
    ) -> None:
        self._conn = conn
        self._embedder = embedder
        self._table = sql.Identifier(table)
        self._warnings: list[str] = []
        self._meta = self._load_meta()
        warning = manifest_warning(self._meta, manifest_path) if self._meta else None
        if warning:
            self._warn(warning)

    @classmethod
    def connect(
        cls,
        database_url: str,
        embedder: TextEmbedder | None = None,
        *,
        table: str = DEFAULT_TABLE,
        manifest_path: Path = DEFAULT_MANIFEST,
        **conn_kwargs: object,
    ) -> SkillIndex:
        # Autocommit: lookups are read-only, and session settings must not be
        # rolled back with an aborted transaction.
        conn = db_connect(database_url, autocommit=True, **conn_kwargs)
        conn.execute(sql.SQL("SET hnsw.ef_search = {}").format(sql.Literal(EF_SEARCH)))
        return cls(conn, embedder, table=table, manifest_path=manifest_path)

    @property
    def meta(self) -> dict:
        """index_meta contents: model, build, taxonomy and (once measured) thresholds."""
        return self._meta

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    @property
    def embedder(self) -> TextEmbedder:
        # Loaded on first search only, so exact lookups never load the model.
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def lookup_exact(self, raw: str) -> list[Candidate]:
        """Concepts with an alias equal to normalize(raw); several when ambiguous."""
        norm = normalize(raw)
        if not norm:
            return []
        rows = self._conn.execute(
            sql.SQL(
                "SELECT canonical_id, canonical_name, alias FROM {} WHERE alias_norm = %s "
                "ORDER BY canonical_id, alias"
            ).format(self._table),
            [norm],
        ).fetchall()
        return dedupe_candidates(Candidate(cid, name, alias, 0.0) for cid, name, alias in rows)

    def search(self, raws: list[str], k: int = 10) -> list[list[Candidate]]:
        """Top-k concepts per raw string by cosine distance; [] for blank strings."""
        results: list[list[Candidate]] = [[] for _ in raws]
        wanted = [i for i, raw in enumerate(raws) if raw.strip()]
        if not wanted:
            return results
        vectors = self.embedder.embed([raws[i].strip() for i in wanted])
        for i, vector in zip(wanted, vectors, strict=True):
            rows = self.nearest(vector, k * FETCH_FACTOR)
            results[i] = dedupe_candidates((c for _, c in rows), k)
        return results

    def nearest(self, vector: np.ndarray, limit: int) -> list[tuple[int, Candidate]]:
        """Raw nearest alias rows (row id, candidate), no dedup; used by the eval."""
        rows = self._conn.execute(
            sql.SQL(
                "SELECT id, canonical_id, canonical_name, alias, embedding <=> %s AS distance "
                "FROM {} WHERE embedding IS NOT NULL ORDER BY embedding <=> %s LIMIT %s"
            ).format(self._table),
            [vector, vector, limit],
        ).fetchall()
        return [(row_id, Candidate(cid, name, alias, float(dist)))
                for row_id, cid, name, alias, dist in rows]

    @property
    def connection(self) -> psycopg.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SkillIndex:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _load_meta(self) -> dict:
        exists = self._conn.execute("SELECT to_regclass('index_meta') IS NOT NULL").fetchone()[0]
        if not exists:
            self._warn("index_meta table not found; run build_index.py")
            return {}
        return dict(self._conn.execute("SELECT key, value FROM index_meta").fetchall())

    def _warn(self, message: str) -> None:
        self._warnings.append(message)
        log.warning(message)
