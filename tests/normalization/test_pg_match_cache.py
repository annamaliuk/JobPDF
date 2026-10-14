"""JM-19: the Postgres skill match cache.

Tests marked ``db`` need Postgres through DATABASE_URL and skip without it. Each one
works in a throwaway schema, so the real database is never touched. pgvector is not
needed: the cache table has no vector column.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Callable, Iterator

import psycopg
import pytest

from jobpdf.normalization import pg_match_cache
from jobpdf.normalization.db import database_url
from jobpdf.normalization.index import Candidate
from jobpdf.normalization.match_cache import (
    InMemorySkillMatchCache,
    MatchCandidate,
    MatcherVersions,
    MatchTier,
    SkillMatch,
    cache_key,
)
from jobpdf.normalization.matcher import SkillMatcher
from jobpdf.normalization.pg_match_cache import PostgresSkillMatchCache, default_match_cache
from jobpdf.normalization.text import normalize

BIGQUERY = "jobpdf:skill/google-bigquery"
V1 = MatcherVersions(index="abc", prompt="1")
V2 = MatcherVersions(index="def", prompt="1")
V1_PROMPT_10 = MatcherVersions(index="abc", prompt="10")  # "abc|10|" must not match "abc|1|"
META = {
    "model": {"name": "intfloat/multilingual-e5-small", "revision": "abc"},
    "build": {"built_at": "2026-10-01T10:00:00+00:00"},
    "taxonomy": {"manifest_sha256": "f00", "esco_version": "v1.2.1"},
    "thresholds": {"auto_accept_distance": 0.12},
}


def match() -> SkillMatch:
    return SkillMatch(
        raw="GBQ", normalized="gbq", concept_id=BIGQUERY, preferred_label="Google BigQuery",
        tier=MatchTier.EXACT, distance=0.0,
        candidates=[MatchCandidate(concept_id=BIGQUERY, preferred_label="Google BigQuery",
                                   matched_alias="gbq", distance=0.0)],
    )


# --- fixtures ------------------------------------------------------------------------


@pytest.fixture
def db_url() -> str:
    url = database_url()
    if not url:
        pytest.skip("DATABASE_URL not set (environment or .env)")
    return url


@pytest.fixture
def schema(db_url: str) -> Iterator[str]:
    """A throwaway schema per test."""
    name = f"test_{uuid.uuid4().hex[:12]}"
    try:
        admin = psycopg.connect(db_url, autocommit=True, connect_timeout=10)
    except psycopg.OperationalError as exc:
        pytest.skip(f"database not reachable: {type(exc).__name__}")
    with admin:
        admin.execute(f'CREATE SCHEMA "{name}"')
    try:
        yield name
    finally:
        with psycopg.connect(db_url, autocommit=True, connect_timeout=10) as admin:
            admin.execute(f'DROP SCHEMA "{name}" CASCADE')


@pytest.fixture
def make_cache(db_url: str, schema: str) -> Iterator[Callable[..., PostgresSkillMatchCache]]:
    """Caches pinned to the throwaway schema; each call opens its own connection."""
    made: list[PostgresSkillMatchCache] = []

    def make(**kwargs: object) -> PostgresSkillMatchCache:
        cache = PostgresSkillMatchCache(db_url, options=f"-c search_path={schema}", **kwargs)
        made.append(cache)
        return cache

    yield make
    for cache in made:
        cache.close()


# --- the cache contract ---------------------------------------------------------------


@pytest.mark.db
def test_round_trip_returns_what_was_stored(make_cache) -> None:
    cache = make_cache()
    key = cache_key("gbq", V1)

    assert cache.get(key) is None
    cache.set(key, match())

    assert cache.get(key) == match()


@pytest.mark.db
def test_set_overwrites_an_existing_key(make_cache) -> None:
    cache = make_cache()
    key = cache_key("gbq", V1)

    cache.set(key, match())
    cache.set(key, match().model_copy(update={"low_confidence": True}))

    assert cache.get(key).low_confidence is True
    assert len(cache) == 1


@pytest.mark.db
def test_a_second_connection_sees_the_first_ones_rows(make_cache) -> None:
    # The reason for JM-19: the API and the Airflow job are separate processes.
    writer, reader = make_cache(), make_cache()

    writer.set(cache_key("gbq", V1), match())

    assert reader.get(cache_key("gbq", V1)) == match()


@pytest.mark.db
def test_versions_are_part_of_the_key(make_cache) -> None:
    cache = make_cache()
    cache.set(cache_key("gbq", V1), match())

    assert cache.get(cache_key("gbq", V2)) is None
    assert cache.get(cache_key("gbq", V1)) == match()


@pytest.mark.db
def test_prune_other_versions_keeps_only_the_given_generation(make_cache) -> None:
    cache = make_cache()
    cache.set(cache_key("gbq", V1), match())
    cache.set(cache_key("dbt", V1), match())
    cache.set(cache_key("gbq", V1_PROMPT_10), match())
    cache.set(cache_key("gbq", V2), match())

    assert cache.prune_other_versions(V2) == 3

    assert len(cache) == 1
    assert cache.get(cache_key("gbq", V2)) == match()


@pytest.mark.db
def test_a_row_that_no_longer_validates_is_a_miss(make_cache, db_url: str, schema: str) -> None:
    cache = make_cache()
    with psycopg.connect(db_url, autocommit=True) as admin:
        admin.execute(
            f'INSERT INTO "{schema}".skill_match_cache (cache_key, match) VALUES (%s, %s::jsonb)',
            ["k", '{"unexpected": true}'],
        )

    assert cache.get("k") is None


# --- degradation ----------------------------------------------------------------------


@pytest.mark.db
def test_a_failing_statement_does_not_pause_the_cache(make_cache) -> None:
    # A key too large for the btree index fails as an OperationalError in psycopg, but
    # the connection is fine: only that one write may be lost.
    cache = make_cache(clock=lambda: 0.0)

    cache.set(os.urandom(4000).hex(), match())
    cache.set("ok", match())

    assert cache.get("ok") == match()


@pytest.mark.db
def test_outage_degrades_to_misses_then_recovers(
    make_cache, db_url: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    real_connect = psycopg.connect

    def admin() -> psycopg.Connection:
        return real_connect(db_url, autocommit=True, connect_timeout=10)

    now = [0.0]
    app = f"jm19_{uuid.uuid4().hex[:8]}"
    cache = make_cache(retry_after_s=30.0, clock=lambda: now[0], application_name=app)
    key = cache_key("gbq", V1)
    cache.set(key, match())

    with admin() as conn:  # cut the cache's connection from the server side
        conn.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                     "WHERE application_name = %s", [app])
        for _ in range(100):
            if not conn.execute("SELECT 1 FROM pg_stat_activity WHERE application_name = %s",
                                [app]).fetchall():
                break
            time.sleep(0.05)
        else:
            raise AssertionError("backend did not terminate")

    connects: list[int] = []

    def counting_connect(*args: object, **kwargs: object) -> psycopg.Connection:
        connects.append(1)
        return real_connect(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pg_match_cache.psycopg, "connect", counting_connect)

    with caplog.at_level(logging.WARNING):
        assert cache.get(key) is None  # notices the dead connection; no exception
        cache.set(key, match())  # skipped quietly
        assert cache.get(key) is None
    assert connects == []  # cool-down: the database is left alone
    assert db_url not in caplog.text  # the URL carries the password

    now[0] = 31.0
    assert cache.get(key) == match()  # reconnected, and the earlier row survived
    assert len(connects) == 1


# --- through the matcher --------------------------------------------------------------


class CountingIndex:
    """Index with one exact alias, counting how often the matcher had to ask it."""

    def __init__(self) -> None:
        self.meta = META
        self.exact_calls = 0

    def lookup_exact(self, raw: str) -> list[Candidate]:
        self.exact_calls += 1
        if normalize(raw) != "gbq":
            return []
        return [Candidate(canonical_id=BIGQUERY, canonical_name="Google BigQuery",
                          matched_alias="gbq", distance=0.0)]

    def search(self, raws: list[str], k: int = 10) -> list[list[Candidate]]:
        return [[] for _ in raws]


@pytest.mark.db
def test_a_second_matcher_reuses_the_first_ones_results(make_cache) -> None:
    first_index, second_index = CountingIndex(), CountingIndex()

    (first,) = SkillMatcher(first_index, cache=make_cache()).match_many(["GBQ"])
    # A new matcher with its own connection, as in another process, and another spelling.
    (second,) = SkillMatcher(second_index, cache=make_cache()).match_many([" gbq"])

    assert first_index.exact_calls == 1
    assert second_index.exact_calls == 0  # served from the cache
    assert (second.concept_id, second.tier, second.raw) == (BIGQUERY, MatchTier.EXACT, " gbq")


# --- default_match_cache ----------------------------------------------------------------


def test_default_cache_is_in_memory_without_a_database_url(monkeypatch) -> None:
    monkeypatch.setattr(pg_match_cache, "database_url", lambda: None)

    assert isinstance(default_match_cache(), InMemorySkillMatchCache)


def test_default_cache_falls_back_when_the_database_is_unreachable(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    url = "postgresql://user:secret-pw@127.0.0.1:1/none"  # nothing listens on port 1
    monkeypatch.setattr(pg_match_cache, "database_url", lambda: url)

    with caplog.at_level(logging.WARNING):
        cache = default_match_cache()

    assert isinstance(cache, InMemorySkillMatchCache)
    assert "secret-pw" not in caplog.text


@pytest.mark.db
def test_default_cache_is_postgres_when_the_database_works(
    db_url: str, schema: str, monkeypatch
) -> None:
    monkeypatch.setattr(pg_match_cache, "database_url", lambda: db_url)
    monkeypatch.setenv("PGOPTIONS", f"-c search_path={schema}")  # keep it in the throwaway schema

    cache = default_match_cache()
    try:
        assert isinstance(cache, PostgresSkillMatchCache)
    finally:
        if isinstance(cache, PostgresSkillMatchCache):
            cache.close()
