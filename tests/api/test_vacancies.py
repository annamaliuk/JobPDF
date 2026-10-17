"""JM-36: the vacancy provider the API ranks against."""

import logging
import threading
import uuid
from collections.abc import Iterator

import psycopg
import pytest

from jobpdf.api import vacancies as vacancies_module
from jobpdf.api.vacancies import (
    VACANCY_CACHE_TTL_S,
    VACANCY_RETRY_S,
    VacancyProvider,
    postgres_loader,
)
from jobpdf.ingestion.store import VacancyStore, apply_schema
from jobpdf.normalization.db import database_url
from tests.ingestion.test_store import T0, features, posting
from tests.pipeline.fakes import vacancy

SECRET = "postgresql://jobpdf:hunter2@db.internal:5432/jobpdf"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Loader:
    def __init__(self, *results) -> None:
        self.results = list(results)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        result = self.results[min(self.calls, len(self.results)) - 1]
        if isinstance(result, Exception):
            raise result
        return result


A, B = vacancy("a", ["python"]), vacancy("b", ["sql"])


def provider(loader, clock=None, **kwargs) -> VacancyProvider:
    return VacancyProvider(loader, clock=clock or Clock(), **kwargs)


def test_the_first_call_loads_and_later_calls_use_the_copy_until_it_expires() -> None:
    clock, loader = Clock(), Loader([A], [A, B])
    vacancies = provider(loader, clock)

    assert list(vacancies.get()) == [A] and loader.calls == 1
    clock.now += VACANCY_CACHE_TTL_S - 1
    assert list(vacancies.get()) == [A] and loader.calls == 1
    clock.now += 1
    assert list(vacancies.get()) == [A, B] and loader.calls == 2


def test_a_failed_reload_keeps_serving_the_old_list_and_retries_after_the_pause(caplog) -> None:
    clock, loader = Clock(), Loader([A], psycopg.OperationalError(SECRET), [A, B])
    vacancies = provider(loader, clock)
    vacancies.get()
    clock.now += VACANCY_CACHE_TTL_S

    with caplog.at_level(logging.DEBUG):
        assert list(vacancies.get()) == [A] and loader.calls == 2
    clock.now += VACANCY_RETRY_S - 1
    assert list(vacancies.get()) == [A] and loader.calls == 2
    clock.now += 1
    assert list(vacancies.get()) == [A, B] and loader.calls == 3
    assert "OperationalError" in caplog.text and "hunter2" not in caplog.text


def test_with_nothing_loaded_a_failure_gives_an_empty_list_not_an_error() -> None:
    clock, loader = Clock(), Loader(RuntimeError("down"), [A])
    vacancies = provider(loader, clock)

    assert list(vacancies.get()) == []
    clock.now += VACANCY_RETRY_S
    assert list(vacancies.get()) == [A]


def test_concurrent_callers_cause_one_reload() -> None:
    gate = threading.Event()

    class Slow(Loader):
        def __call__(self):
            gate.wait(5)
            return super().__call__()

    loader = Slow([A])
    vacancies = provider(loader)
    results: list = []
    threads = [threading.Thread(target=lambda: results.append(list(vacancies.get())))
               for _ in range(5)]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join(10)

    assert loader.calls == 1 and results == [[A]] * 5


def test_the_provider_keeps_its_own_copy_of_what_the_loader_returned() -> None:
    from_loader = [A]
    vacancies = provider(Loader(from_loader))
    vacancies.get()

    from_loader.append(B)

    assert list(vacancies.get()) == [A]


def test_constants_are_sane() -> None:
    assert VACANCY_CACHE_TTL_S > VACANCY_RETRY_S > 0


# --- the Postgres loader ------------------------------------------------------------------


class _Rows:
    def __init__(self, rows) -> None:
        self.rows = rows

    def fetchall(self):
        return self.rows


class _Conn:
    def __init__(self, rows) -> None:
        self.rows, self.closed = rows, False

    def execute(self, sql, params=None):
        return _Rows(self.rows)

    def rollback(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.closed = True


def test_the_loader_reads_every_stored_vacancy_over_a_short_lived_connection(monkeypatch) -> None:
    conn = _Conn([(A.model_dump(mode="json"), None), (B.model_dump(mode="json"), None)])
    seen = {}

    def connect(url, **kwargs):
        seen.update(url=url, **kwargs)
        return conn

    monkeypatch.setattr(vacancies_module.psycopg, "connect", connect)

    loaded = postgres_loader("postgresql://x")()

    assert [v.id for v in loaded] == ["a", "b"] and conn.closed
    assert seen == {"url": "postgresql://x", "connect_timeout": 5}


@pytest.fixture
def schema_url(monkeypatch) -> Iterator[str]:
    url = database_url()
    if not url:
        pytest.skip("DATABASE_URL not set (environment or .env)")
    name = f"test_{uuid.uuid4().hex[:12]}"
    try:
        admin = psycopg.connect(url, autocommit=True, connect_timeout=10)
    except psycopg.OperationalError as exc:
        pytest.skip(f"database not reachable: {type(exc).__name__}")
    with admin:
        admin.execute(f'CREATE SCHEMA "{name}"')
    monkeypatch.setenv("PGOPTIONS", f"-c search_path={name}")  # libpq reads this
    yield url
    with psycopg.connect(url, autocommit=True, connect_timeout=10) as admin:
        admin.execute(f'DROP SCHEMA "{name}" CASCADE')


@pytest.mark.db
def test_the_loader_returns_what_the_store_wrote(schema_url) -> None:
    with psycopg.connect(schema_url) as conn:
        apply_schema(conn)
        post = posting(1)
        VacancyStore(conn).upsert(post, features(post), T0)

    loaded = postgres_loader(schema_url)()

    assert len(loaded) == 1 and loaded[0].source == "remotive"


@pytest.mark.db
def test_the_loader_on_a_database_without_the_table_raises_so_the_provider_degrades(
        schema_url) -> None:
    assert list(provider(postgres_loader(schema_url)).get()) == []
