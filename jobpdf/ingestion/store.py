"""The vacancy store (JM-32): one row per posting, written once however often it is fetched.

Identity is ``(source, source_id)``. For each batch of fetched postings the store says which
need processing (extraction, normalization, embedding) and which do not:

- ``new``        no row yet.
- ``changed``    the row's ``content_hash`` differs: the posting's content changed.
- ``stale``      content is the same, but the row was built with other versions of the
                 embedding model, the vacancy text or the normalization than the current ones
                 (the spec's "stale vectors are detected later").
- ``unchanged``  nothing to do except note that the posting was seen again.

``upsert`` is one ``INSERT ... ON CONFLICT DO UPDATE``, so a rerun, a backfill or two
overlapping runs end with one row per posting. An older fetch never overwrites a newer one,
and ``last_seen_at`` never moves back (the guard above and the invariant last_seen <=
fetched_at make that so); it follows the time the SOURCE data was fetched, so
serving a stale cache cannot make postings look fresher.

Nothing is ever deleted here: a posting missing from one feed is not "closed" (Remotive is
24 hours behind and fetches can be partial). Readers choose how recent ``last_seen_at`` must
be (``load(seen_since=...)``).

What the content hash covers: the fields a posting's processing depends on. It leaves out
``posted_at`` on purpose (some sources stamp "updated" times, which would re-embed
everything daily). Logs carry counts and codes, never titles or text.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from jobpdf.ingestion.sources.models import SourcePosting
from jobpdf.ranking.models import VacancyFeatures

log = logging.getLogger(__name__)

SCHEMA_FILE = Path(__file__).resolve().parents[2] / "sql" / "004_vacancies.sql"
HASH_VERSION = "1"  # bump when the hashed fields change: every row is then reprocessed once

NEW = "new"
CHANGED = "changed"
STALE = "stale"
UNCHANGED = "unchanged"
UNAVAILABLE = "unavailable"  # a version that could not be determined (never compared)

@dataclass(frozen=True)
class Versions:
    """What the current code would stamp on a vacancy. A field that is None or
    "unavailable" is unknown now and is not compared (else every row would look stale)."""

    embedding: str | None = None
    vacancy_text: str | None = None
    taxonomy: str | None = None
    index: str | None = None
    matcher_prompt: str | None = None


@dataclass(frozen=True)
class StoredState:
    content_hash: str
    versions: Versions
    fetched_at: datetime


@dataclass
class Plan:
    """How a batch splits up. Each posting is in exactly one list; ``duplicates`` counts
    repeats of the same (source, source_id) within the batch (the first one wins)."""

    new: list[SourcePosting] = field(default_factory=list)
    changed: list[SourcePosting] = field(default_factory=list)
    stale: list[SourcePosting] = field(default_factory=list)
    unchanged: list[SourcePosting] = field(default_factory=list)
    duplicates: int = 0

    @property
    def to_process(self) -> list[SourcePosting]:
        return [*self.new, *self.changed, *self.stale]


@dataclass(frozen=True)
class UpsertResult:
    inserted: bool = False
    updated: bool = False
    skipped_older: bool = False  # a newer fetch is already stored


def content_hash(posting: SourcePosting) -> str:
    """Fingerprint of everything processing depends on. Equal postings always hash equal;
    tag order does not matter."""
    material = {
        "v": HASH_VERSION,
        "title": posting.title, "company": posting.company, "url": posting.url,
        "location": posting.location_raw, "remote": posting.remote,
        "description": posting.description, "excerpt": posting.description_is_excerpt,
        "tags": sorted(posting.tags), "category": posting.category,
        "type": posting.employment_type, "salary": posting.salary_raw,
    }
    text = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_stale(stored: Versions, current: Versions) -> bool:
    """Any known current version that differs from what the row was built with."""
    for name in Versions.__dataclass_fields__:
        now = getattr(current, name)
        if now and now != UNAVAILABLE and getattr(stored, name) != now:
            return True
    return False


def plan_batch(postings: Iterable[SourcePosting],
               stored: dict[tuple[str, str], StoredState], current: Versions) -> Plan:
    """Split a batch into new / changed / stale / unchanged against what is stored. The
    order of each list follows the batch. Repeats of one (source, source_id) are counted
    in ``duplicates`` and ignored."""
    plan = Plan()
    seen: set[tuple[str, str]] = set()
    for posting in postings:
        key = (posting.source, posting.source_id)
        if key in seen:
            plan.duplicates += 1
            continue
        seen.add(key)
        state = stored.get(key)
        if state is None:
            plan.new.append(posting)
        elif state.content_hash != content_hash(posting):
            plan.changed.append(posting)
        elif is_stale(state.versions, current):
            plan.stale.append(posting)
        else:
            plan.unchanged.append(posting)
    return plan


def apply_schema(conn: psycopg.Connection, schema_file: Path = SCHEMA_FILE) -> None:
    conn.execute(schema_file.read_text(encoding="utf-8"))
    conn.commit()


class VacancyStore:
    """Reads and writes ``vacancies`` through a psycopg connection the caller owns.
    Each method commits its own work."""

    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    # -- deciding what to process ------------------------------------------------------

    def states(self, keys: Sequence[tuple[str, str]]) -> dict[tuple[str, str], StoredState]:
        rows = self.conn.execute(
            """SELECT v.source, v.source_id, v.content_hash, v.embedding_version,
                      v.vacancy_text_version, v.taxonomy_version, v.index_version,
                      v.matcher_prompt_version, v.fetched_at
               FROM vacancies v
               JOIN unnest(%s::text[], %s::text[]) AS k(source, source_id)
                 ON v.source = k.source AND v.source_id = k.source_id""",
            ([k[0] for k in keys], [k[1] for k in keys]),
        ).fetchall()
        self.conn.rollback()  # end the read transaction
        return {
            (r[0], r[1]): StoredState(r[2], Versions(*r[3:8]), r[8]) for r in rows
        }

    def classify(self, postings: Iterable[SourcePosting], current: Versions) -> Plan:
        postings = list(postings)
        stored = self.states(sorted({(p.source, p.source_id) for p in postings}))
        return plan_batch(postings, stored, current)

    # -- writing ----------------------------------------------------------------------

    def touch(self, postings: Iterable[SourcePosting], seen_at: datetime) -> int:
        """Note that these (unchanged) postings were seen again. ``last_seen_at`` and
        ``fetched_at`` only move forward. Returns how many rows were updated."""
        keys = [(p.source, p.source_id) for p in postings]
        cursor = self.conn.execute(
            """UPDATE vacancies v
               SET last_seen_at = GREATEST(v.last_seen_at, %(seen)s),
                   fetched_at = GREATEST(v.fetched_at, %(seen)s)
               FROM unnest(%(sources)s::text[], %(ids)s::text[]) AS k(source, source_id)
               WHERE v.source = k.source AND v.source_id = k.source_id
                 AND v.last_seen_at < %(seen)s""",
            {"seen": seen_at, "sources": [k[0] for k in keys], "ids": [k[1] for k in keys]},
        )
        self.conn.commit()
        return cursor.rowcount

    def upsert(self, posting: SourcePosting, features: VacancyFeatures,
               fetched_at: datetime) -> UpsertResult:
        """Insert the vacancy, or replace the row of the same posting. If the stored row
        comes from a newer fetch, nothing changes (``skipped_older``)."""
        if (features.source, features.id) != (posting.source, posting.source_id):
            raise ValueError("features do not belong to this posting")
        row = self.conn.execute(
            """INSERT INTO vacancies AS v (
                   source, source_id, content_hash, features, embedding, embedding_version,
                   vacancy_text_version, taxonomy_version, index_version,
                   matcher_prompt_version, url, posted_at, fetched_at, first_seen_at,
                   last_seen_at)
               VALUES (%(source)s, %(source_id)s, %(hash)s, %(features)s, %(embedding)s,
                       %(embedding_version)s, %(text_version)s, %(taxonomy)s, %(index)s,
                       %(prompt)s, %(url)s, %(posted_at)s, %(fetched)s, %(fetched)s,
                       %(fetched)s)
               ON CONFLICT (source, source_id) DO UPDATE SET
                   content_hash = EXCLUDED.content_hash,
                   features = EXCLUDED.features,
                   embedding = EXCLUDED.embedding,
                   embedding_version = EXCLUDED.embedding_version,
                   vacancy_text_version = EXCLUDED.vacancy_text_version,
                   taxonomy_version = EXCLUDED.taxonomy_version,
                   index_version = EXCLUDED.index_version,
                   matcher_prompt_version = EXCLUDED.matcher_prompt_version,
                   url = EXCLUDED.url,
                   posted_at = EXCLUDED.posted_at,
                   fetched_at = EXCLUDED.fetched_at,
                   last_seen_at = EXCLUDED.last_seen_at,
                   updated_at = now()
               WHERE v.fetched_at <= EXCLUDED.fetched_at
               RETURNING (xmax = 0) AS inserted""",
            {
                "source": posting.source, "source_id": posting.source_id,
                "hash": content_hash(posting),
                "features": Jsonb(features.model_dump(mode="json", exclude={"embedding"})),
                "embedding": features.embedding,
                "embedding_version": features.embedding_version,
                "text_version": features.vacancy_text_version,
                "taxonomy": features.taxonomy_version, "index": features.index_version,
                "prompt": features.matcher_prompt_version,
                "url": posting.url, "posted_at": posting.posted_at, "fetched": fetched_at,
            },
        ).fetchone()
        self.conn.commit()
        if row is None:
            return UpsertResult(skipped_older=True)
        return UpsertResult(inserted=bool(row[0]), updated=not row[0])

    # -- reading ----------------------------------------------------------------------

    def load(self, *, source: str | None = None,
             seen_since: datetime | None = None) -> list[VacancyFeatures]:
        """Stored vacancies, newest first (posted time, then source and id for a stable
        order), optionally one source and only those seen since ``seen_since``."""
        rows = self.conn.execute(
            """SELECT features, embedding FROM vacancies
               WHERE (%(source)s::text IS NULL OR source = %(source)s)
                 AND (%(since)s::timestamptz IS NULL OR last_seen_at >= %(since)s)
               ORDER BY posted_at DESC NULLS LAST, source, source_id""",
            {"source": source, "since": seen_since},
        ).fetchall()
        self.conn.rollback()
        return [
            VacancyFeatures.model_validate({**features, "embedding": embedding})
            for features, embedding in rows
        ]

    def count(self, source: str | None = None) -> int:
        row = self.conn.execute(
            "SELECT count(*) FROM vacancies WHERE (%s::text IS NULL OR source = %s)",
            (source, source),
        ).fetchone()
        self.conn.rollback()
        return int(row[0])
