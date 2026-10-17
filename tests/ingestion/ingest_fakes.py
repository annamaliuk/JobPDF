"""Fakes for the ingestion tests: normalizer, embedder, extractor, and an in-memory store with
the same behaviour as VacancyStore (decisions come from the real ``plan_batch``)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from jobpdf.ingestion.requirements import Requirements
from jobpdf.ingestion.sources.models import SourcePosting
from jobpdf.ingestion.store import (
    Plan,
    StoredState,
    UpsertResult,
    Versions,
    content_hash,
    plan_batch,
)
from jobpdf.normalization.matcher import MatchTier, SkillMatch
from jobpdf.ranking.models import VacancyFeatures

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
KNOWN = {"python": "c/python", "sql": "c/sql", "dbt": "c/dbt"}


@dataclass
class ServiceVersions:
    taxonomy: str | None = "v1.2.1"
    index: str = "idx1"
    matcher_prompt: str = "p1"


class FakeService:
    """normalize_skill_groups like NormalizationService: known skills get a concept."""

    def __init__(self, versions: ServiceVersions | None = None) -> None:
        self.versions = versions or ServiceVersions()
        self.calls: list[tuple[dict, str | None]] = []
        self.fail = False

    def normalize_skill_groups(self, groups, context=None):
        self.calls.append(({k: list(v) for k, v in groups.items()}, context))
        if self.fail:
            raise RuntimeError("LLM down: SecretTitleXYZ")
        result = {}
        for name, raws in groups.items():
            result[name] = [self._match(raw) for raw in raws]
        return result

    @staticmethod
    def _match(raw: str) -> SkillMatch:
        concept = KNOWN.get(raw.lower())
        return SkillMatch(
            raw=raw, normalized=raw.strip().lower(), concept_id=concept,
            preferred_label=raw if concept else None,
            tier=MatchTier.EXACT if concept else MatchTier.UNMATCHED, warnings=[],
        )


class FakeEmbedder:
    model_name = "fake/e5"
    revision = "rev1"
    prefix = "query: "

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail = False

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("model crashed: SecretTitleXYZ")
        return np.asarray([[float(len(t) % 7), 1.0, 0.5] for t in texts], dtype=np.float32)


class FakeExtractor:
    def __init__(self, fail_for: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.fail_for = fail_for or set()

    def extract(self, posting: SourcePosting) -> Requirements:
        self.calls.append(posting.source_id)
        if posting.source_id in self.fail_for:
            raise ValueError(f"cannot read {posting.title}")
        return Requirements(
            required_skills=list(posting.tags), preferred_skills=["dbt", "Obscure Tool"],
            responsibilities=["Build pipelines"], min_years_experience=2, languages=["English"],
            warnings=["fake_warning"],
        )


class _Conn:
    def __init__(self) -> None:
        self.rollbacks = 0

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


class MemoryStore:
    """VacancyStore's interface over a dict."""

    def __init__(self) -> None:
        self.conn = _Conn()
        self.rows: dict[tuple[str, str], dict] = {}
        self.fail_upsert_for: set[str] = set()
        self.touched: list[tuple[list[str], datetime]] = []
        self.upserts: list[str] = []

    def classify(self, postings, current: Versions) -> Plan:
        postings = list(postings)
        stored = {
            key: StoredState(row["hash"], row["versions"], row["fetched_at"])
            for key, row in self.rows.items()
        }
        return plan_batch(postings, stored, current)

    def upsert(self, posting, features: VacancyFeatures, fetched_at) -> UpsertResult:
        import psycopg

        if posting.source_id in self.fail_upsert_for:
            raise psycopg.OperationalError("db down: SecretTitleXYZ")
        key = (posting.source, posting.source_id)
        row = self.rows.get(key)
        if row is not None and row["fetched_at"] > fetched_at:
            return UpsertResult(skipped_older=True)
        self.upserts.append(posting.source_id)
        self.rows[key] = {
            "hash": content_hash(posting), "features": features, "fetched_at": fetched_at,
            "last_seen": fetched_at,
            "versions": Versions(
                features.embedding_version, features.vacancy_text_version,
                features.taxonomy_version, features.index_version,
                features.matcher_prompt_version, features.extraction_version),
        }
        return UpsertResult(inserted=row is None, updated=row is not None)

    def touch(self, postings, seen_at) -> int:
        keys = [(p.source, p.source_id) for p in postings]
        self.touched.append(([k[1] for k in keys], seen_at))
        n = 0
        for key in keys:
            row = self.rows.get(key)
            if row is not None and row["last_seen"] < seen_at:
                row["last_seen"] = row["fetched_at"] = seen_at
                n += 1
        return n


def posting(i: int = 1, **changes) -> SourcePosting:
    data = {
        "source": "remotive", "source_id": str(i), "url": f"https://remotive.com/{i}",
        "title": f"Data Engineer {i}", "company": "Acme", "location_raw": "Worldwide",
        "remote": True, "description": "<p>Build things</p>", "tags": ["python", "sql"],
        "posted_at": T0,
    }
    return SourcePosting(**{**data, **changes})
