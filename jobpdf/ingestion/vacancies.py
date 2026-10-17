"""Incremental vacancy ingestion (JM-33): process only what is new, changed or stale.

``ingest`` takes the postings of one source (from the fetch cache), asks the store which of
them need work (``VacancyStore.classify``), and for those only runs

    requirements (JM-48 rules / tags fallback) -> normalization (JM-20) -> embedding (JM-23)

then upserts the result. Everything else (unchanged, deferred, or failed this time) just has
its ``last_seen_at`` moved forward, because it is still in the feed. So a
rerun does nothing, a backfill never duplicates, and when the model, text, extraction or
normalization versions change the affected rows are rebuilt once.

Rules (decided here, adjustable constants):

- At most ``MAX_PROCESSED_PER_RUN`` postings per run, newest postings first; the rest wait
  for the next run (``deferred``). A first load of thousands of postings therefore spreads
  over runs instead of making thousands of LLM calls at once.
- One posting failing never stops the others. It is counted under a stage code
  (``extraction_failed``, ``normalization_failed``, ``embedding_failed``, ``store_failed``),
  left unstored, and tried again next run. After ``MAX_CONSECUTIVE_FAILURES`` in a row the
  run stops (``aborted``): an LLM or database outage should cost five attempts, not hundreds.
- Nothing is deleted. Unmatched skills stay as ``raw:<text>`` requirements (JM-22).
- Ingestion never fetches: it reads the fetch jobs' cache (``load_cached_postings``).

Logs and summaries carry counts and codes only: never titles, descriptions or exception text.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import psycopg
from pydantic import BaseModel, ConfigDict

from jobpdf.ingestion import remotive
from jobpdf.ingestion.requirements import RequirementExtractor, Requirements
from jobpdf.ingestion.sources.models import SourcePosting
from jobpdf.ingestion.sources.registry import SOURCE_NAMES, build_adapter
from jobpdf.ingestion.sources.remotive import to_source_posting
from jobpdf.ingestion.sources.runner import read_cache
from jobpdf.ingestion.store import VacancyStore, Versions
from jobpdf.normalization.matcher import SkillMatch
from jobpdf.ranking.models import SkillRef, VacancyFeatures
from jobpdf.ranking.semantic import (
    VACANCY_TEXT_VERSION,
    TextEmbedder,
    embed_vacancy,
    embedding_version,
)

log = logging.getLogger(__name__)

SOURCES = (remotive.SOURCE, *SOURCE_NAMES)
VACANCY_CONTEXT = "job posting"  # tells the matcher what kind of text the skills came from
MAX_PROCESSED_PER_RUN = 200
MAX_CONSECUTIVE_FAILURES = 5

EXTRACTION_FAILED = "extraction_failed"
NORMALIZATION_FAILED = "normalization_failed"
EMBEDDING_FAILED = "embedding_failed"
STORE_FAILED = "store_failed"
NO_CACHE = "no_cache"


class _Service(Protocol):
    """What ingestion needs from NormalizationService."""

    @property
    def versions(self) -> Any: ...

    def normalize_skill_groups(self, groups: Any, context: str | None = None) -> Any: ...


class IngestError(RuntimeError):
    """One or more sources failed in ``run_ingest`` (the others were still processed).
    ``results`` holds the per-source outcome, exactly as a successful run would return it."""

    def __init__(self, message: str, results: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.results = results


class StageError(Exception):
    """A posting failed in one stage; only the stage code is kept, never the cause's text."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class IngestSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    received: int = 0
    duplicates: int = 0
    new: int = 0
    changed: int = 0
    stale: int = 0
    unchanged: int = 0
    inserted: int = 0
    updated: int = 0
    skipped_older: int = 0
    deferred: int = 0
    failed: dict[str, int] = {}
    warnings: dict[str, int] = {}  # extractor/embedding warning codes -> postings
    aborted: bool = False


def current_versions(normalizer: _Service, embedder: TextEmbedder | None,
                     extractor: RequirementExtractor | None = None) -> Versions:
    """What a vacancy processed now would be stamped with."""
    service = normalizer.versions
    return Versions(
        embedding=embedding_version(embedder), vacancy_text=VACANCY_TEXT_VERSION,
        taxonomy=service.taxonomy, index=service.index, matcher_prompt=service.matcher_prompt,
        extraction=getattr(extractor, "version", None),
    )


def _refs(matches: Sequence[SkillMatch]) -> list[SkillRef]:
    return [ref for ref in (SkillRef.from_match(m) for m in matches) if not ref.is_empty]


def build_features(
    posting: SourcePosting, *, extractor: RequirementExtractor, normalizer: _Service,
    embedder: TextEmbedder | None = None,
) -> tuple[VacancyFeatures, Requirements]:
    """One posting -> the vacancy as ranking reads it, embedded and version-stamped."""
    try:
        requirements = extractor.extract(posting)
    except Exception:
        raise StageError(EXTRACTION_FAILED) from None
    try:
        groups = normalizer.normalize_skill_groups(
            {"required": requirements.required_skills,
             "preferred": requirements.preferred_skills}, VACANCY_CONTEXT)
        versions = normalizer.versions
        features = VacancyFeatures(
            id=posting.source_id, source=posting.source, title=posting.title,
            company=posting.company, url=posting.url, posted_at=posting.posted_at,
            location_raw=posting.location_raw,
            required_skills=_refs(groups["required"]),
            preferred_skills=_refs(groups["preferred"]),
            responsibilities=requirements.responsibilities,
            min_years_experience=requirements.min_years_experience,
            education_requirement=requirements.education, languages=requirements.languages,
            taxonomy_version=versions.taxonomy, index_version=versions.index,
            matcher_prompt_version=versions.matcher_prompt,
            extraction_version=getattr(extractor, "version", None),
        )
    except Exception:
        raise StageError(NORMALIZATION_FAILED) from None
    try:
        return embed_vacancy(features, embedder), requirements
    except Exception:
        raise StageError(EMBEDDING_FAILED) from None


def _newest_first(postings: Sequence[SourcePosting]) -> list[SourcePosting]:
    undated = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(postings, key=lambda p: (-(p.posted_at or undated).timestamp(), p.source_id))


def ingest(
    postings: Sequence[SourcePosting], *, store: VacancyStore, extractor: RequirementExtractor,
    normalizer: _Service, embedder: TextEmbedder | None = None, fetched_at: datetime,
    source: str | None = None, limit: int = MAX_PROCESSED_PER_RUN,
) -> IngestSummary:
    """Store the new, changed and stale postings of a fetch, and note the rest as seen."""
    name = source or (postings[0].source if postings else "")
    plan = store.classify(postings, current_versions(normalizer, embedder, extractor))
    ordered = _newest_first(plan.to_process)
    todo, deferred = ordered[:limit], ordered[limit:]

    inserted = updated = skipped_older = 0
    failed: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    streak = 0
    aborted = False
    not_rebuilt: list[SourcePosting] = []  # failed: the old row (if any) still stands
    for index, posting in enumerate(todo):
        try:
            features, requirements = build_features(
                posting, extractor=extractor, normalizer=normalizer, embedder=embedder)
            try:
                result = store.upsert(posting, features, fetched_at)
            except psycopg.Error:
                store.conn.rollback()
                raise StageError(STORE_FAILED) from None
        except StageError as exc:
            failed[exc.code] += 1
            not_rebuilt.append(posting)
            streak += 1
            if streak >= MAX_CONSECUTIVE_FAILURES:
                aborted = True
                deferred = [*todo[index + 1:], *deferred]
                break
            continue
        streak = 0
        inserted += result.inserted
        updated += result.updated
        skipped_older += result.skipped_older
        warnings.update(set(requirements.warnings))
    store.touch([*plan.unchanged, *not_rebuilt, *deferred], fetched_at)

    summary = IngestSummary(
        source=name, received=len(postings), duplicates=plan.duplicates, new=len(plan.new),
        changed=len(plan.changed), stale=len(plan.stale), unchanged=len(plan.unchanged),
        inserted=inserted, updated=updated, skipped_older=skipped_older,
        deferred=len(deferred), failed=dict(failed), warnings=dict(warnings), aborted=aborted,
    )
    log.info("%s: %d received, %d new, %d changed, %d stale, %d unchanged; stored %d+%d, "
             "failed %s, deferred %d%s", name, summary.received, summary.new, summary.changed,
             summary.stale, summary.unchanged, inserted, updated, dict(failed), len(deferred),
             ", ABORTED" if aborted else "")
    return summary


def load_cached_postings(source: str, *, base: Path | None = None
                         ) -> tuple[list[SourcePosting], datetime] | None:
    """The newest cached fetch of a source as common postings, with when it was fetched; or
    None when nothing is cached. Never makes a request."""
    if source == remotive.SOURCE:
        cached = remotive.read_cache(base=base)
        if cached is None:
            return None
        return [to_source_posting(p) for p in cached.postings], cached.fetched_at
    result = read_cache(build_adapter(source, {}), base=base)
    return None if result is None else (result.postings, result.fetched_at)


def run_ingest(
    sources: Sequence[str] | None = None, *, store: VacancyStore | None = None,
    extractor: RequirementExtractor | None = None, normalizer: _Service | None = None,
    embedder: TextEmbedder | None = None, base: Path | None = None,
    limit: int = MAX_PROCESSED_PER_RUN,
    connect: Callable[[], psycopg.Connection] | None = None,
) -> dict[str, Any]:
    """What the Airflow task runs: ingest each source's cached fetch, one source failing
    never stopping the rest. Returns counts per source; raises ``IngestError`` at the end
    if any source failed (or aborted), so the task shows red after everything was tried."""
    names = list(sources) if sources is not None else list(SOURCES)
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        raise ValueError(f"unknown source(s): {', '.join(unknown)}")

    from jobpdf.ingestion.requirements import RulesExtractor

    extractor = extractor or RulesExtractor()
    owned = None
    results: list[dict[str, Any]] = []
    problems: list[str] = []
    try:
        for name in names:
            try:
                cached = load_cached_postings(name, base=base)
            except Exception as exc:
                log.warning("%s: cache unreadable (%s)", name, type(exc).__name__)
                results.append({"source": name, "status": "failed", "code": "cache_unreadable"})
                problems.append(name)
                continue
            if cached is None:
                results.append({"source": name, "status": NO_CACHE})
                continue
            postings, fetched_at = cached
            try:
                if store is None:
                    owned = _open_store(connect)
                    store = VacancyStore(owned)
                if normalizer is None:
                    normalizer = _default_normalizer()
                summary = ingest(postings, store=store, extractor=extractor,
                                 normalizer=normalizer, embedder=embedder,
                                 fetched_at=fetched_at, source=name, limit=limit)
            except Exception as exc:
                log.warning("%s: ingestion failed (%s)", name, type(exc).__name__)
                if store is not None:
                    _rollback(store)
                results.append({"source": name, "status": "failed", "code": "ingest_failed"})
                problems.append(name)
                continue
            results.append({"status": "aborted" if summary.aborted else "ok",
                            **summary.model_dump()})
            if summary.aborted:
                problems.append(name)
    finally:
        if owned is not None:
            owned.close()
    if problems:
        raise IngestError(f"ingestion problems in: {', '.join(problems)}", results)
    return {"sources": results}


def _rollback(store: VacancyStore) -> None:
    try:
        store.conn.rollback()
    except Exception:  # the connection itself may be gone; the next source reports that
        pass


def _open_store(connect: Callable[[], psycopg.Connection] | None) -> psycopg.Connection:
    from jobpdf.ingestion.store import apply_schema
    from jobpdf.normalization.db import database_url

    if connect is not None:
        conn = connect()
    else:
        url = database_url()
        if not url:
            raise RuntimeError("DATABASE_URL is not set")
        conn = psycopg.connect(url, connect_timeout=10)
    apply_schema(conn)
    return conn


def _default_normalizer() -> _Service:
    from jobpdf.normalization.service import get_normalization_service

    return get_normalization_service()
