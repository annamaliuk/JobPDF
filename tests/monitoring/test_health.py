import json
import logging
import time
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg
import pytest

import jobpdf.normalization.db as normalization_db
from jobpdf.ingestion.store import VacancyStore, apply_schema
from jobpdf.monitoring import health as health_module
from jobpdf.monitoring.health import (
    FETCH_STALE_AFTER_S,
    HTTP_STATUS,
    VACANCY_STALE_AFTER_S,
    build_health,
    http_status,
    worst,
)
from jobpdf.normalization.db import database_url
from tests.ingestion.ingest_fakes import posting
from tests.ingestion.test_store import features

NOW = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
SECRET = "sk-ant-SECRET-0123456789"
GOOD_STATUS = {
    "index": "ok", "taxonomy_version": "v1.2.1", "index_version": "idx1",
    "matcher_prompt_version": "p1", "llm": "enabled", "llm_model": "claude-x", "cache": "postgres",
}


class FakeNormalizer:
    def __init__(self, **changes) -> None:
        self.data = {**GOOD_STATUS, **changes}

    def status(self):
        return self.data


class FakeKeys:
    def __init__(self, state: str = "set") -> None:
        self.state = state

    def status(self) -> str:
        return self.state


class _Result:
    def __init__(self, rows) -> None:
        self.rows = rows

    def fetchall(self):
        return self.rows


class FakeConn:
    """Answers SELECT 1 and the vacancy statistics query."""

    def __init__(self, rows=(), *, fail_select=False, missing_table=False) -> None:
        self.rows, self.fail_select, self.missing_table = list(rows), fail_select, missing_table
        self.closed = False
        self.rollbacks = 0

    def execute(self, sql, params=None):
        if sql.strip() == "SELECT 1":
            if self.fail_select:
                raise psycopg.OperationalError("server closed the connection: " + SECRET)
            return _Result([(1,)])
        if self.missing_table:
            raise psycopg.errors.UndefinedTable("relation vacancies does not exist")
        return _Result(self.rows)

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


def stats_row(source="remotive", count=3, seen_ago_s=3600, posted_ago_s=7200):
    return (source, count, NOW - timedelta(seconds=seen_ago_s),
            NOW - timedelta(seconds=posted_ago_s))


def run(*, conn=None, normalizer=None, keys=None, env=None, base=None, **kwargs):
    conn = conn if conn is not None else FakeConn([stats_row()])
    env = {"ANTHROPIC_API_KEY": SECRET} if env is None else env
    if base is None:
        base = Path("/nonexistent-cache")
    return build_health(connect=lambda: conn, normalizer=normalizer, api_keys=keys,
                        environ=env, cache_base=base, clock=lambda: NOW.timestamp(), **kwargs)


def write_cache(base: Path, source: str, age_s: int, requests: list[float] | None = None):
    folder = base / source
    folder.mkdir(parents=True, exist_ok=True)
    stamp = (NOW - timedelta(seconds=age_s)).strftime("%Y%m%dT%H%M%SZ")
    (folder / f"{stamp}.json").write_text("{}")
    if requests is not None:
        (folder / "requests.json").write_text(json.dumps(requests))


@pytest.fixture
def fresh(tmp_path):
    write_cache(tmp_path, "remotive", 3600)
    return tmp_path


# --- overall status -----------------------------------------------------------------------


def test_everything_in_order_is_ok_and_answers_200(fresh) -> None:
    report = run(normalizer=FakeNormalizer(), keys=FakeKeys(), base=fresh)

    assert report.status == "ok" and report.http_status == 200 and http_status(report) == 200
    assert set(report.checks) == {"database", "vacancies", "normalization", "llm", "api_keys",
                                  "fetch"}
    assert all(c.status == "ok" and c.code is None for c in report.checks.values())
    assert report.checked_at == NOW and report.duration_ms >= 0


def test_the_overall_status_is_the_worst_check() -> None:
    assert worst([]) == "ok" and worst(["ok", "ok"]) == "ok"
    assert worst(["ok", "degraded"]) == "degraded"
    assert worst(["degraded", "down", "ok"]) == "down"
    assert HTTP_STATUS == {"ok": 200, "degraded": 200, "down": 503}


def test_a_dependency_that_is_not_passed_is_not_checked(fresh) -> None:
    report = run(base=fresh)

    assert set(report.checks) == {"database", "vacancies", "llm", "fetch"}


# --- database ---------------------------------------------------------------------------


def test_an_unreachable_database_is_down_and_the_service_answers_503() -> None:
    def refuse():
        raise psycopg.OperationalError("could not connect to postgresql://u:" + SECRET + "@h/db")

    report = build_health(connect=refuse, environ={"ANTHROPIC_API_KEY": "x"},
                          cache_base=Path("/none"), clock=lambda: NOW.timestamp())

    assert report.status == "down" and report.http_status == 503
    assert report.checks["database"].status == "down"
    assert report.checks["database"].code == "unreachable"
    assert report.checks["vacancies"].status == "degraded"
    assert report.checks["vacancies"].code == "database_down"
    assert SECRET not in report.model_dump_json()


def test_a_connection_that_dies_on_the_first_query_is_unreachable_and_closed() -> None:
    conn = FakeConn(fail_select=True)

    report = run(conn=conn)

    assert report.checks["database"].code == "unreachable" and conn.closed
    assert SECRET not in report.model_dump_json()


def test_an_unexpected_error_opening_the_connection_is_check_failed_without_its_text(
        caplog) -> None:
    caplog.set_level(logging.DEBUG)

    def explode():
        raise RuntimeError("boom " + SECRET)

    report = build_health(connect=explode, environ={}, cache_base=Path("/none"),
                          clock=lambda: NOW.timestamp())

    assert report.checks["database"].status == "down"
    assert report.checks["database"].code == "check_failed"
    assert SECRET not in report.model_dump_json() and SECRET not in caplog.text
    assert "RuntimeError" in caplog.text


def test_without_a_database_url_the_database_check_says_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(normalization_db, "ENV_FILE", Path("/nonexistent.env"))

    report = build_health(environ={}, cache_base=Path("/none"), clock=lambda: NOW.timestamp())

    assert report.checks["database"].code == "not_configured"
    assert report.status == "down"


def test_the_default_connection_has_a_short_timeout(monkeypatch) -> None:
    seen = {}

    def fake_connect(url, **kwargs):
        seen.update(url=url, **kwargs)
        return FakeConn([stats_row()])

    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    monkeypatch.setattr(psycopg, "connect", fake_connect)

    report = build_health(environ={"ANTHROPIC_API_KEY": "x"}, cache_base=Path("/none"),
                          clock=lambda: NOW.timestamp())

    assert seen == {"url": "postgresql://u:p@h/db", "connect_timeout": 3}
    assert report.checks["database"].status == "ok"


def test_the_connection_is_always_closed_afterwards() -> None:
    conn = FakeConn([stats_row()])

    run(conn=conn)

    assert conn.closed


# --- vacancies ---------------------------------------------------------------------------


def test_vacancy_counts_and_ages_are_reported() -> None:
    rows = [stats_row("remotive", 5, seen_ago_s=600, posted_ago_s=86400),
            stats_row("adzuna", 2, seen_ago_s=7200, posted_ago_s=3600)]

    check = run(conn=FakeConn(rows)).checks["vacancies"]

    assert check.status == "ok" and check.data == {
        "total": 7, "by_source": {"remotive": 5, "adzuna": 2},
        "newest_seen_age_s": 600, "newest_posting_age_s": 3600}


def test_no_vacancies_is_degraded() -> None:
    check = run(conn=FakeConn([])).checks["vacancies"]

    assert (check.status, check.code) == ("degraded", "no_vacancies")
    assert check.data["total"] == 0 and check.data["newest_seen_age_s"] is None


def test_vacancies_not_seen_for_a_day_are_stale() -> None:
    fresh_edge = run(conn=FakeConn([stats_row(seen_ago_s=VACANCY_STALE_AFTER_S)]))
    stale = run(conn=FakeConn([stats_row(seen_ago_s=VACANCY_STALE_AFTER_S + 1)]))

    assert fresh_edge.checks["vacancies"].status == "ok"
    assert (stale.checks["vacancies"].status, stale.checks["vacancies"].code) == (
        "degraded", "stale_vacancies")


def test_staleness_follows_the_newest_source_not_the_oldest() -> None:
    rows = [stats_row("remotive", 1, seen_ago_s=VACANCY_STALE_AFTER_S * 5),
            stats_row("adzuna", 1, seen_ago_s=60)]

    assert run(conn=FakeConn(rows)).checks["vacancies"].status == "ok"


def test_a_missing_table_is_degraded_and_the_transaction_is_rolled_back() -> None:
    conn = FakeConn(missing_table=True)

    check = run(conn=conn).checks["vacancies"]

    assert (check.status, check.code) == ("degraded", "table_missing") and conn.rollbacks == 2


def test_a_successful_query_also_ends_its_transaction() -> None:
    conn = FakeConn([stats_row()])

    run(conn=conn)

    assert conn.rollbacks == 2  # after the connection probe and after the statistics query


def test_one_source_without_posting_dates_does_not_break_the_others() -> None:
    rows = [("remotive", 2, NOW - timedelta(seconds=10), None),
            stats_row("adzuna", 1, posted_ago_s=500)]

    check = run(conn=FakeConn(rows)).checks["vacancies"]

    assert check.status == "ok" and check.data["newest_posting_age_s"] == 500


def test_vacancies_without_posting_dates_are_reported_with_none() -> None:
    row = ("remotive", 1, NOW - timedelta(seconds=10), None)

    data = run(conn=FakeConn([row])).checks["vacancies"].data

    assert data["newest_posting_age_s"] is None and data["total"] == 1


# --- normalization and the keys -----------------------------------------------------------


def test_normalization_status_is_passed_through_without_non_text_values() -> None:
    normalizer = FakeNormalizer(extra=3, flag=True)

    check = run(normalizer=normalizer).checks["normalization"]

    assert check.status == "ok" and check.data["index_version"] == "idx1"
    assert "extra" not in check.data and "flag" not in check.data


def test_an_unreachable_skill_index_is_degraded() -> None:
    check = run(normalizer=FakeNormalizer(index="unavailable")).checks["normalization"]

    assert (check.status, check.code) == ("degraded", "index_unavailable")


def test_a_disabled_llm_matching_tier_is_degraded() -> None:
    check = run(normalizer=FakeNormalizer(llm="disabled")).checks["normalization"]

    assert (check.status, check.code) == ("degraded", "llm_tier_disabled")


def test_an_unavailable_index_is_reported_before_a_disabled_llm() -> None:
    check = run(normalizer=FakeNormalizer(index="unavailable", llm="disabled")
                ).checks["normalization"]

    assert check.code == "index_unavailable"


def test_a_failing_status_call_is_check_failed_not_a_crash(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    normalizer = FakeNormalizer()
    normalizer.status = lambda: 1 / 0

    report = run(normalizer=normalizer)

    assert report.checks["normalization"].code == "check_failed"
    assert report.checks["normalization"].status == "degraded"
    assert report.versions["taxonomy"] is None and report.versions["ranking"] == "1"
    assert "ZeroDivisionError" in caplog.text and "division" not in caplog.text


def test_the_llm_key_is_reported_as_set_or_missing_never_its_value() -> None:
    present = run(env={"ANTHROPIC_API_KEY": SECRET})
    absent = run(env={})
    blank = run(env={"ANTHROPIC_API_KEY": "   "})

    assert present.checks["llm"].status == "ok" and present.checks["llm"].data == {"key": "set"}
    for report in (absent, blank):
        assert (report.checks["llm"].status, report.checks["llm"].code) == ("down", "key_missing")
        assert report.checks["llm"].data == {"key": "missing"} and report.http_status == 503
    assert SECRET not in present.model_dump_json()


def test_api_keys_are_reported_as_set_or_missing() -> None:
    ok = run(keys=FakeKeys("set")).checks["api_keys"]
    missing = run(keys=FakeKeys("missing")).checks["api_keys"]

    assert (ok.status, ok.data) == ("ok", {"keys": "set"})
    assert (missing.status, missing.code, missing.data) == (
        "degraded", "no_api_keys", {"keys": "missing"})


def test_an_unknown_api_key_state_is_not_treated_as_set() -> None:
    check = run(keys=FakeKeys("unknown")).checks["api_keys"]

    assert (check.status, check.code) == ("degraded", "no_api_keys")


def test_with_the_real_api_keys_class() -> None:
    from jobpdf.api.access import ApiKeys, hash_key

    assert run(keys=ApiKeys({"web": hash_key("k")})).checks["api_keys"].status == "ok"
    assert run(keys=ApiKeys({})).checks["api_keys"].code == "no_api_keys"


# --- fetch freshness ----------------------------------------------------------------------


def test_fetch_ages_and_request_counts_are_reported(tmp_path) -> None:
    write_cache(tmp_path, "remotive", 7200,
                requests=[NOW.timestamp() - 100, NOW.timestamp() - 90000])

    check = run(base=tmp_path).checks["fetch"]

    remotive = check.data["sources"]["remotive"]
    assert check.status == "ok" and remotive["age_s"] == 7200
    assert remotive["requests_24h"] == 1 and remotive["counted"] is True
    assert set(check.data["sources"]) == {"remotive", "themuse", "arbeitnow", "adzuna",
                                          "jooble"}


def test_a_missing_remotive_cache_is_degraded(tmp_path) -> None:
    check = run(base=tmp_path).checks["fetch"]

    assert (check.status, check.code) == ("degraded", "no_cache")
    assert check.data["sources"]["remotive"]["age_s"] is None


def test_a_remotive_cache_older_than_a_day_is_stale(tmp_path) -> None:
    write_cache(tmp_path, "remotive", FETCH_STALE_AFTER_S)
    assert run(base=tmp_path).checks["fetch"].status == "ok"

    write_cache(tmp_path / "x", "remotive", FETCH_STALE_AFTER_S + 5)
    stale = run(base=tmp_path / "x").checks["fetch"]

    assert (stale.status, stale.code) == ("degraded", "stale_fetch")


def test_the_newest_cache_file_decides_the_age(tmp_path) -> None:
    write_cache(tmp_path, "remotive", 90000)
    write_cache(tmp_path, "remotive", 600)
    (tmp_path / "remotive" / "notes.json").write_text("{}")  # not a cache stamp: ignored

    assert run(base=tmp_path).checks["fetch"].data["sources"]["remotive"]["age_s"] == 600


def test_a_source_that_was_never_fetched_does_not_count_against_health(tmp_path) -> None:
    write_cache(tmp_path, "remotive", 60)

    check = run(base=tmp_path).checks["fetch"]

    assert check.status == "ok"
    assert check.data["sources"]["adzuna"] == {"age_s": None, "requests_24h": None,
                                               "counted": False}
    assert check.data["sources"]["arbeitnow"]["counted"] is False
    assert check.data["sources"]["remotive"]["counted"] is True


def test_a_source_that_was_fetched_and_then_went_quiet_is_degraded(tmp_path) -> None:
    write_cache(tmp_path, "remotive", 60)
    write_cache(tmp_path, "jooble", FETCH_STALE_AFTER_S + 60)

    check = run(base=tmp_path).checks["fetch"]

    assert (check.status, check.code) == ("degraded", "stale_fetch")
    assert check.data["sources"]["jooble"]["counted"] is True


def test_a_source_that_is_fetched_regularly_counts_and_is_fine(tmp_path) -> None:
    write_cache(tmp_path, "remotive", 60)
    write_cache(tmp_path, "themuse", 3600)

    check = run(base=tmp_path).checks["fetch"]

    assert check.status == "ok" and check.data["sources"]["themuse"]["counted"] is True


def test_the_first_fetch_problem_decides_the_code(tmp_path) -> None:
    write_cache(tmp_path, "themuse", FETCH_STALE_AFTER_S + 60)  # remotive has no cache at all

    check = run(base=tmp_path).checks["fetch"]

    assert (check.status, check.code) == ("degraded", "no_cache")


def test_a_request_exactly_a_day_old_no_longer_counts(tmp_path) -> None:
    now = NOW.timestamp()
    write_cache(tmp_path, "remotive", 60, requests=[now - 86400, now - 86399, now - 1])

    assert run(base=tmp_path).checks["fetch"].data["sources"]["remotive"]["requests_24h"] == 2


def test_an_unreadable_ledger_gives_no_request_count(tmp_path) -> None:
    write_cache(tmp_path, "remotive", 60)
    (tmp_path / "remotive" / "requests.json").write_text("garbage")

    assert run(base=tmp_path).checks["fetch"].data["sources"]["remotive"]["requests_24h"] is None


# --- versions and secrets ------------------------------------------------------------------


def test_versions_come_from_the_code_and_the_normalization_service() -> None:
    versions = run(normalizer=FakeNormalizer()).versions

    assert versions["ranking"] == "1" and versions["response_schema"] == "1"
    assert versions["embedding"].startswith("intfloat/multilingual-e5-small@")
    assert (versions["taxonomy"], versions["index"], versions["matcher_prompt"]) == (
        "v1.2.1", "idx1", "p1")
    assert "taxonomy" not in run().versions


def test_the_normalization_status_is_asked_for_once_because_it_probes_the_database() -> None:
    class Counting(FakeNormalizer):
        calls = 0

        def status(self):
            Counting.calls += 1
            return super().status()

    run(normalizer=Counting())

    assert Counting.calls == 1


def test_no_secret_or_address_appears_anywhere_in_the_report(tmp_path, monkeypatch) -> None:
    write_cache(tmp_path, "remotive", 60)
    env = {"ANTHROPIC_API_KEY": SECRET, "JOOBLE_API_KEY": SECRET + "-j",
           "ADZUNA_APP_ID": SECRET + "-a", "ADZUNA_APP_KEY": SECRET + "-b"}
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:" + SECRET + "@dbhost/db")

    report = run(base=tmp_path, env=env, normalizer=FakeNormalizer(), keys=FakeKeys())

    text = report.model_dump_json()
    assert "SECRET" not in text and "dbhost" not in text and "postgresql" not in text
    assert "https://" not in text and "http://" not in text


def test_every_check_records_how_long_it_took(fresh) -> None:
    report = run(normalizer=FakeNormalizer(), keys=FakeKeys(), base=fresh)

    assert all(isinstance(c.latency_ms, int) and c.latency_ms >= 0 for c in report.checks.values())


def test_a_slow_database_check_shows_its_latency() -> None:
    def slow() -> FakeConn:
        time.sleep(0.03)
        return FakeConn([stats_row()])

    report = build_health(connect=slow, environ={"ANTHROPIC_API_KEY": "x"},
                          cache_base=Path("/none"), clock=lambda: NOW.timestamp())

    assert report.checks["database"].latency_ms >= 25


def test_module_constants_are_sane() -> None:
    assert health_module.DB_CONNECT_TIMEOUT_S == 3
    assert FETCH_STALE_AFTER_S == VACANCY_STALE_AFTER_S == 24 * 3600


# --- the real database ---------------------------------------------------------------------


@pytest.fixture
def real_conn() -> Iterator:
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

    def make() -> psycopg.Connection:
        return psycopg.connect(url, options=f"-c search_path={name}", connect_timeout=10)

    yield make
    with psycopg.connect(url, autocommit=True, connect_timeout=10) as admin:
        admin.execute(f'DROP SCHEMA "{name}" CASCADE')


def real_run(make, **kwargs):
    return build_health(connect=make, environ={"ANTHROPIC_API_KEY": "x"},
                        cache_base=Path("/none"), clock=lambda: NOW.timestamp(), **kwargs)


@pytest.mark.db
def test_real_database_without_the_table_is_degraded_not_down(real_conn) -> None:
    report = real_run(real_conn)

    assert report.checks["database"].status == "ok"
    assert report.checks["vacancies"].code == "table_missing"


@pytest.mark.db
def test_real_database_with_an_empty_table(real_conn) -> None:
    conn = real_conn()
    apply_schema(conn)
    conn.close()

    check = real_run(real_conn).checks["vacancies"]

    assert check.code == "no_vacancies" and check.data["total"] == 0


@pytest.mark.db
def test_real_database_counts_and_ages_stored_vacancies(real_conn) -> None:
    conn = real_conn()
    apply_schema(conn)
    store = VacancyStore(conn)
    seen = NOW - timedelta(hours=2)
    for i, source in ((1, "remotive"), (2, "remotive"), (3, "adzuna")):
        p = posting(i, source=source, posted_at=NOW - timedelta(days=i))
        store.upsert(p, features(p), seen)
    conn.close()

    check = real_run(real_conn).checks["vacancies"]

    assert check.status == "ok" and check.data == {
        "total": 3, "by_source": {"adzuna": 1, "remotive": 2},
        "newest_seen_age_s": 7200, "newest_posting_age_s": 86400}


@pytest.mark.db
def test_real_database_stale_vacancies(real_conn) -> None:
    conn = real_conn()
    apply_schema(conn)
    p = posting(1)
    VacancyStore(conn).upsert(p, features(p), NOW - timedelta(days=3))
    conn.close()

    assert real_run(real_conn).checks["vacancies"].code == "stale_vacancies"


# --- the script ------------------------------------------------------------------------------


def _script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "health_check_script", Path(__file__).parents[2] / "scripts" / "health_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_prints_the_report_and_exits_zero_unless_down(capsys, fresh) -> None:
    script = _script()

    ok = script.main([], builder=lambda **kw: run(base=fresh))
    printed = json.loads(capsys.readouterr().out)
    degraded = script.main([], builder=lambda **kw: run(conn=FakeConn([])))
    capsys.readouterr()
    down = script.main([], builder=lambda **kw: run(env={}))

    assert (ok, degraded, down) == (0, 0, 1)
    assert printed["status"] == "ok" and printed["checks"]["database"]["status"] == "ok"


def test_the_script_refuses_unknown_arguments(capsys) -> None:
    assert _script().main(["--nope"], builder=lambda **kw: pytest.fail("built")) == 2
    assert "usage" in capsys.readouterr().out


def test_the_full_flag_passes_the_normalization_service(monkeypatch, fresh, capsys) -> None:
    import jobpdf.normalization.service as service

    marker = FakeNormalizer()
    monkeypatch.setattr(service, "get_normalization_service", lambda: marker)
    seen = {}

    def builder(**kwargs):
        seen.update(kwargs)
        return run(base=fresh)

    _script().main(["--full"], builder=builder)

    assert seen == {"normalizer": marker}
