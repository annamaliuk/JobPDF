import json
import logging
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone

import psycopg
import pytest

from jobpdf.ingestion import remotive, vacancies
from jobpdf.ingestion.requirements import TagsExtractor
from jobpdf.ingestion.sources.arbeitnow import Arbeitnow
from jobpdf.ingestion.sources.runner import fetch_source, read_cache
from jobpdf.ingestion.sources.themuse import TheMuse
from jobpdf.ingestion.store import VacancyStore, apply_schema
from jobpdf.ingestion.vacancies import (
    NO_CACHE,
    SOURCES,
    IngestError,
    load_cached_postings,
    run_ingest,
)
from jobpdf.normalization.db import database_url
from tests.ingestion.ingest_fakes import (
    FakeEmbedder,
    FakeExtractor,
    FakeService,
    MemoryStore,
)
from tests.ingestion.source_fakes import (
    HOUR,
    T0,
    Clock,
    Transport,
    arbeitnow_job,
    arbeitnow_page,
    fast,
    muse_job,
    muse_page,
    no_sleep,
    ok,
)


@pytest.fixture
def base(tmp_path):
    return tmp_path / "cache"


def remotive_body(*ids: int) -> bytes:
    jobs = [{"id": i, "url": f"https://remotive.com/remote-jobs/{i}", "title": f"Dev {i}",
             "company_name": "Acme", "tags": ["python", "sql"], "candidate_required_location":
             "Worldwide", "publication_date": "2026-09-20T10:00:00", "description": "<p>x</p>"}
            for i in ids]
    return json.dumps({"jobs": jobs, "job-count": len(jobs)}).encode()


def cache_remotive(base, *ids: int, clock=None) -> None:
    remotive.fetch_postings(base=base, transport=lambda url, timeout: (200, remotive_body(*ids)),
                            clock=clock or Clock())


def cache_arbeitnow(base, *ids: int, clock=None) -> None:
    page = arbeitnow_page([arbeitnow_job(i) for i in ids])
    fetch_source(Arbeitnow(policy=fast()), base=base, transport=Transport(ok(page)),
                 clock=clock or Clock(), sleep=no_sleep)


def cache_muse(base, *ids: int, clock=None) -> None:
    page = muse_page([muse_job(i) for i in ids])
    fetch_source(TheMuse(policy=fast()), base=base, transport=Transport(ok(page)),
                 clock=clock or Clock(), sleep=no_sleep)


# --- reading the cache, never fetching ---------------------------------------------------


def test_remotive_cache_reads_the_newest_response_of_any_age(base) -> None:
    assert remotive.read_cache(base=base) is None
    clock = Clock()
    cache_remotive(base, 1, clock=clock)
    clock.advance(30 * 24 * HOUR)

    result = remotive.read_cache(base=base)

    assert [p.id for p in result.postings] == [1]
    assert result.from_cache and not result.stale
    assert result.fetched_at.timestamp() == T0


def test_remotive_cache_skips_unusable_files_and_prefers_the_newest(base) -> None:
    clock = Clock()
    cache_remotive(base, 1, clock=clock)
    clock.advance(10 * HOUR)
    cache_remotive(base, 2, 3, clock=clock)
    folder = base / "remotive"
    (folder / "20990101T000000Z.json").write_text("{bad")

    assert [p.id for p in remotive.read_cache(base=base).postings] == [2, 3]


def test_a_source_cache_is_read_without_keys_requests_or_age_limits(base) -> None:
    from jobpdf.ingestion.sources.adzuna import Adzuna

    assert read_cache(Adzuna(app_id=None, app_key=None), base=base) is None
    clock = Clock()
    cache_muse(base, 1, 2, clock=clock)
    clock.advance(90 * 24 * HOUR)

    result = read_cache(TheMuse(), base=base)

    assert [p.source_id for p in result.postings] == ["1001", "1002"]
    assert result.from_cache and not result.stale and result.warnings == []


def test_a_cache_is_readable_even_when_the_key_is_no_longer_set(base) -> None:
    from jobpdf.ingestion.sources.adzuna import Adzuna
    from tests.ingestion.source_fakes import adzuna_job, adzuna_page

    keyed = Adzuna(app_id="a", app_key="b", policy=fast())
    fetch_source(keyed, base=base, transport=Transport(ok(adzuna_page([adzuna_job(1)]))),
                 clock=Clock(), sleep=no_sleep)

    result = read_cache(Adzuna(app_id=None, app_key=None), base=base)

    assert [p.source_id for p in result.postings] == ["481"]


def test_a_partial_cache_says_so_when_read(base) -> None:
    pages = [ok(muse_page([muse_job(i)], page=i, page_count=9)) for i in range(1, 10)]
    fetch_source(TheMuse(policy=fast(max_pages=2)), base=base, transport=Transport(*pages),
                 clock=Clock(), sleep=no_sleep)

    assert read_cache(TheMuse(), base=base).warnings == ["partial"]


def test_cached_postings_come_back_in_the_common_shape_with_their_fetch_time(base) -> None:
    cache_remotive(base, 5)
    cache_arbeitnow(base, 1)

    rem, rem_at = load_cached_postings("remotive", base=base)
    arb, arb_at = load_cached_postings("arbeitnow", base=base)

    assert [(p.source, p.source_id, p.remote) for p in rem] == [("remotive", "5", True)]
    assert rem[0].tags == ["python", "sql"] and rem_at.timestamp() == T0
    assert [(p.source, p.source_id) for p in arb] == [("arbeitnow", "python-dev-1-berlin")]
    assert arb_at.tzinfo is not None


def test_a_source_that_was_never_fetched_has_no_cached_postings(base) -> None:
    for name in SOURCES:
        assert load_cached_postings(name, base=base) is None


def test_reading_the_cache_never_touches_the_network_or_the_ledger(base) -> None:
    cache_remotive(base, 1)
    ledger = (base / "remotive" / "requests.json").read_text()

    load_cached_postings("remotive", base=base)

    assert (base / "remotive" / "requests.json").read_text() == ledger  # conftest blocks sockets


# --- run_ingest (no database: an in-memory store) ----------------------------------------


def deps(**kw):
    return {"store": MemoryStore(), "extractor": FakeExtractor(), "normalizer": FakeService(),
            "embedder": FakeEmbedder(), **kw}


def test_every_cached_source_is_ingested_and_the_rest_are_reported(base) -> None:
    cache_remotive(base, 1, 2)
    cache_arbeitnow(base, 1)
    d = deps()

    out = run_ingest(base=base, **d)

    by_source = {r["source"]: r for r in out["sources"]}
    assert list(by_source) == list(SOURCES)
    assert by_source["remotive"]["status"] == "ok" and by_source["remotive"]["inserted"] == 2
    assert by_source["arbeitnow"]["inserted"] == 1
    assert by_source["themuse"] == {"source": "themuse", "status": NO_CACHE}
    assert by_source["jooble"]["status"] == NO_CACHE and by_source["adzuna"]["status"] == NO_CACHE
    assert len(d["store"].rows) == 3


def test_running_twice_changes_nothing_the_second_time(base) -> None:
    cache_remotive(base, 1, 2)
    d = deps()
    run_ingest(["remotive"], base=base, **d)
    work = len(d["extractor"].calls)

    again = run_ingest(["remotive"], base=base, **d)

    assert len(d["extractor"].calls) == work
    assert again["sources"][0]["unchanged"] == 2 and again["sources"][0]["inserted"] == 0


def test_a_later_fetch_with_a_changed_posting_updates_only_that_one(base) -> None:
    clock = Clock()
    cache_remotive(base, 1, 2, clock=clock)
    d = deps()
    run_ingest(["remotive"], base=base, **d)
    clock.advance(9 * HOUR)
    body = json.loads(remotive_body(1, 2))
    body["jobs"][1]["title"] = "Staff Dev 2"
    remotive.fetch_postings(base=base, transport=lambda u, t: (200, json.dumps(body).encode()),
                            clock=clock)

    out = run_ingest(["remotive"], base=base, **d)["sources"][0]

    assert (out["changed"], out["unchanged"], out["updated"]) == (1, 1, 1)


def test_only_the_named_sources_are_run_and_unknown_names_are_refused(base) -> None:
    cache_remotive(base, 1)
    cache_arbeitnow(base, 1)

    out = run_ingest(["arbeitnow"], base=base, **deps())

    assert [r["source"] for r in out["sources"]] == ["arbeitnow"]
    with pytest.raises(ValueError, match="unknown source"):
        run_ingest(["careerjet"], base=base, **deps())
    with pytest.raises(ValueError):
        run_ingest(["remotive", "nope"], base=base, **deps())


def test_one_source_failing_does_not_stop_the_others_but_fails_the_task(base, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    cache_remotive(base, 1)
    cache_arbeitnow(base, 1)
    d = deps()
    calls = []
    real = d["store"].classify

    def classify(postings, current):
        calls.append(postings[0].source)
        if postings[0].source == "remotive":
            raise RuntimeError("boom SecretTitleXYZ")
        return real(postings, current)

    d["store"].classify = classify

    with pytest.raises(IngestError) as caught:
        run_ingest(["remotive", "arbeitnow"], base=base, **d)

    assert [(r["source"], r["status"]) for r in caught.value.results] == [
        ("remotive", "failed"), ("arbeitnow", "ok")]
    assert calls == ["remotive", "arbeitnow"] and len(d["store"].rows) == 1
    assert "remotive" in str(caught.value) and "SecretTitleXYZ" not in str(caught.value)
    assert "SecretTitleXYZ" not in caplog.text and d["store"].conn.rollbacks == 1


def test_an_aborted_source_fails_the_task_after_the_rest_ran(base) -> None:
    cache_remotive(base, *range(1, 9))
    cache_arbeitnow(base, 1)
    d = deps(normalizer=FakeService())
    d["normalizer"].fail = True

    with pytest.raises(IngestError, match="remotive") as caught:
        run_ingest(["remotive"], base=base, **d)

    (row,) = caught.value.results
    assert row["status"] == "aborted" and row["aborted"] is True and row["inserted"] == 0


def test_a_source_with_no_cache_is_not_a_failure(base) -> None:
    assert run_ingest(["jooble"], base=base, **deps()) == {
        "sources": [{"source": "jooble", "status": "no_cache"}]}


def test_an_unreadable_cache_fails_that_source_only(base, monkeypatch) -> None:
    cache_arbeitnow(base, 1)
    real = vacancies.load_cached_postings

    def load(name, base=None):
        if name == "remotive":
            raise OSError("disk")
        return real(name, base=base)

    monkeypatch.setattr(vacancies, "load_cached_postings", load)
    d = deps()

    with pytest.raises(IngestError) as caught:
        run_ingest(["remotive", "arbeitnow"], base=base, **d)
    assert caught.value.results[0] == {"source": "remotive", "status": "failed",
                                       "code": "cache_unreadable"}
    assert len(d["store"].rows) == 1


def test_the_default_extractor_is_the_tags_fallback(base) -> None:
    cache_remotive(base, 1)
    d = deps()
    del d["extractor"]

    run_ingest(["remotive"], base=base, **d)

    (row,) = d["store"].rows.values()
    assert [r.key for r in row["features"].required_skills] == ["c/python", "c/sql"]
    assert row["features"].responsibilities == []


def test_the_per_run_limit_is_passed_through(base) -> None:
    cache_remotive(base, 1, 2, 3)

    out = run_ingest(["remotive"], base=base, limit=2, **deps())["sources"][0]

    assert out["inserted"] == 2 and out["deferred"] == 1


def test_sources_cover_remotive_and_the_four_new_ones() -> None:
    assert SOURCES == ("remotive", "themuse", "arbeitnow", "adzuna", "jooble")


# --- the real database, end to end ----------------------------------------------------------


@pytest.fixture
def db_url() -> str:
    url = database_url()
    if not url:
        pytest.skip("DATABASE_URL not set (environment or .env)")
    return url


@pytest.fixture
def connect(db_url: str) -> Iterator:
    """A factory for connections pinned to a throwaway schema."""
    name = f"test_{uuid.uuid4().hex[:12]}"
    try:
        admin = psycopg.connect(db_url, autocommit=True, connect_timeout=10)
    except psycopg.OperationalError as exc:
        pytest.skip(f"database not reachable: {type(exc).__name__}")
    with admin:
        admin.execute(f'CREATE SCHEMA "{name}"')
    opened: list[psycopg.Connection] = []

    def make() -> psycopg.Connection:
        conn = psycopg.connect(db_url, options=f"-c search_path={name}", connect_timeout=10)
        opened.append(conn)
        return conn

    yield make
    for conn in opened:
        conn.close()
    with psycopg.connect(db_url, autocommit=True, connect_timeout=10) as admin:
        admin.execute(f'DROP SCHEMA "{name}" CASCADE')


def database_rows(connect) -> list[tuple]:
    conn = connect()
    out = conn.execute("SELECT source, source_id FROM vacancies ORDER BY 1, 2").fetchall()
    conn.rollback()
    return out


@pytest.mark.db
def test_end_to_end_reruns_and_backfills_never_duplicate(base, connect) -> None:
    clock = Clock()
    cache_remotive(base, 1, 2, clock=clock)
    cache_arbeitnow(base, 1, clock=clock)
    kw = {"extractor": FakeExtractor(), "normalizer": FakeService(), "embedder": FakeEmbedder(),
          "connect": connect, "base": base}

    first = run_ingest(["remotive", "arbeitnow"], **kw)
    second = run_ingest(["remotive", "arbeitnow"], **kw)
    clock.advance(9 * HOUR)
    cache_remotive(base, 1, 2, 3, 4, clock=clock)  # a bigger, overlapping fetch (backfill)
    third = run_ingest(["remotive", "arbeitnow"], **kw)

    assert [r["inserted"] for r in first["sources"]] == [2, 1]
    assert [r["inserted"] for r in second["sources"]] == [0, 0]
    assert [r["unchanged"] for r in second["sources"]] == [2, 1]
    assert third["sources"][0]["new"] == 2 and third["sources"][0]["unchanged"] == 2
    assert database_rows(connect) == [
        ("arbeitnow", "python-dev-1-berlin"), ("remotive", "1"), ("remotive", "2"),
        ("remotive", "3"), ("remotive", "4")]
    assert len(kw["extractor"].calls) == 5  # each posting was extracted exactly once


@pytest.mark.db
def test_end_to_end_two_overlapping_runs_leave_one_row_per_posting(base, connect) -> None:
    cache_remotive(base, 1, 2, 3)
    store_a, store_b = VacancyStore(connect()), VacancyStore(connect())
    apply_schema(store_a.conn)
    kw = {"extractor": FakeExtractor(), "normalizer": FakeService(), "embedder": FakeEmbedder()}
    postings, fetched_at = load_cached_postings("remotive", base=base)

    # Both classified before either wrote: both think everything is new.
    plan_a = store_a.classify(postings, vacancies.current_versions(kw["normalizer"],
                                                                   kw["embedder"]))
    plan_b = store_b.classify(postings, vacancies.current_versions(kw["normalizer"],
                                                                   kw["embedder"]))
    assert len(plan_a.new) == len(plan_b.new) == 3
    for store in (store_a, store_b):
        for p in postings:
            features, _ = vacancies.build_features(p, extractor=kw["extractor"],
                                                   normalizer=kw["normalizer"],
                                                   embedder=kw["embedder"])
            store.upsert(p, features, fetched_at)

    assert database_rows(connect) == [("remotive", "1"), ("remotive", "2"), ("remotive", "3")]


@pytest.mark.db
def test_end_to_end_stored_vacancies_are_readable_by_ranking(base, connect) -> None:
    cache_remotive(base, 1)
    run_ingest(["remotive"], extractor=TagsExtractor(), normalizer=FakeService(),
               embedder=FakeEmbedder(), connect=connect, base=base)

    (vacancy,) = VacancyStore(connect()).load()

    assert vacancy.id == "1" and vacancy.url == "https://remotive.com/remote-jobs/1"
    assert [r.key for r in vacancy.required_skills] == ["c/python", "c/sql"]
    assert vacancy.embedding is not None and vacancy.embedding_version == \
        "fake/e5@rev1|query: "
    assert vacancy.posted_at == datetime(2026, 9, 20, 10, tzinfo=timezone.utc)
    assert vacancy.location_raw == "Worldwide"


@pytest.mark.db
def test_end_to_end_a_poison_posting_fails_alone(base, connect) -> None:
    jobs = json.loads(remotive_body(1, 2, 3))
    jobs["jobs"][1]["title"] = "bad\x00title"
    remotive.fetch_postings(base=base, transport=lambda u, t: (200, json.dumps(jobs).encode()),
                            clock=Clock())

    out = run_ingest(["remotive"], extractor=TagsExtractor(), normalizer=FakeService(),
                     embedder=FakeEmbedder(), connect=connect, base=base)["sources"][0]

    assert out["failed"] == {"store_failed": 1} and out["inserted"] == 2
    assert database_rows(connect) == [("remotive", "1"), ("remotive", "3")]


@pytest.mark.db
def test_the_connection_this_function_opens_is_closed(base, connect) -> None:
    cache_remotive(base, 1)
    opened: list = []

    def tracking():
        conn = connect()
        opened.append(conn)
        return conn

    run_ingest(["remotive"], extractor=TagsExtractor(), normalizer=FakeService(),
               embedder=FakeEmbedder(), connect=tracking, base=base)

    assert len(opened) == 1 and opened[0].closed


@pytest.mark.db
def test_a_caller_supplied_store_is_left_open(base, connect) -> None:
    cache_remotive(base, 1)
    conn = connect()
    apply_schema(conn)

    run_ingest(["remotive"], store=VacancyStore(conn), extractor=TagsExtractor(),
               normalizer=FakeService(), embedder=FakeEmbedder(), base=base)

    assert not conn.closed


def test_without_a_database_url_no_connection_is_attempted(base, monkeypatch) -> None:
    import jobpdf.normalization.db as db

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(db, "ENV_FILE", base / "no.env")
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: pytest.fail("connected"))
    cache_remotive(base, 1)

    with pytest.raises(IngestError) as caught:
        run_ingest(["remotive"], extractor=TagsExtractor(), normalizer=FakeService(),
                   embedder=FakeEmbedder(), base=base)

    assert caught.value.results[0]["code"] == "ingest_failed"


def test_the_database_connection_has_a_timeout_and_the_url_is_not_logged(
        base, monkeypatch, caplog) -> None:
    seen = {}

    def fake_connect(url, **kwargs):
        seen.update(url=url, **kwargs)
        raise psycopg.OperationalError("cannot connect to postgresql://u:pw@host/db")

    monkeypatch.setenv("DATABASE_URL", "postgresql://u:pw@host/db")
    monkeypatch.setattr(psycopg, "connect", fake_connect)
    caplog.set_level(logging.DEBUG)
    cache_remotive(base, 1)

    with pytest.raises(IngestError) as caught:
        run_ingest(["remotive"], extractor=TagsExtractor(), normalizer=FakeService(),
                   embedder=FakeEmbedder(), base=base)

    assert seen == {"url": "postgresql://u:pw@host/db", "connect_timeout": 10}
    assert "pw" not in caplog.text and "pw" not in str(caught.value)
