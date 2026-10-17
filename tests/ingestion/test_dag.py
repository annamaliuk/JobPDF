"""The Airflow DAG file (JM-31). Needs Airflow installed; skipped otherwise.

Airflow is not a project dependency yet (JM-36 adds the Airflow service), so this runs where
it is available and the DAG is otherwise checked by loading it into a real DagBag.
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("AIRFLOW_HOME", tempfile.mkdtemp(prefix="airflow-test-"))
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")

import pytest  # noqa: E402

pytest.importorskip("airflow")

from airflow.models import DagBag  # noqa: E402
from croniter import croniter  # noqa: E402

from jobpdf.ingestion import remotive  # noqa: E402

DAGS = Path(__file__).parents[2] / "dags"


@pytest.fixture(scope="module")
def bag() -> DagBag:
    return DagBag(str(DAGS), include_examples=False)


@pytest.fixture(scope="module")
def dag(bag):
    return bag.dags["remotive_fetch"]


def test_the_dag_file_loads_without_errors(bag) -> None:
    assert bag.import_errors == {}
    assert {"remotive_fetch", "job_sources_fetch", "vacancy_ingest"} <= set(bag.dags)


def test_it_is_one_fetch_task_that_never_catches_up_or_overlaps(dag) -> None:
    assert [t.task_id for t in dag.tasks] == ["fetch"]
    assert dag.catchup is False and dag.max_active_runs == 1


def test_a_failed_run_is_not_retried_because_a_retry_costs_a_request(dag) -> None:
    task = dag.get_task("fetch")

    assert task.retries == 0 and task.execution_timeout == timedelta(minutes=5)


def test_the_schedule_stays_under_remotives_daily_limit_and_always_refreshes(dag) -> None:
    schedule = croniter(dag.schedule_interval, datetime(2026, 10, 1, tzinfo=timezone.utc))
    runs = [schedule.get_next(datetime) for _ in range(30)]
    gaps = [(b - a).total_seconds() for a, b in zip(runs, runs[1:], strict=False)]

    per_day = 86400 / min(gaps)
    assert per_day < remotive.MAX_REQUESTS_PER_DAY  # one request spare for a manual fetch
    assert min(gaps) > remotive.SCHEDULED_MAX_AGE_S  # every run finds the cache old enough
    assert min(gaps) > remotive.MIN_INTERVAL_S


def test_the_task_runs_the_fetch_and_returns_its_summary(dag, monkeypatch) -> None:
    monkeypatch.setattr(remotive, "run_fetch", lambda: {"count": 3})

    assert dag.get_task("fetch").python_callable() == {"count": 3}


# --- job_sources_fetch: The Muse, Arbeitnow, Adzuna, Jooble ---------------------------------


@pytest.fixture(scope="module")
def sources_dag(bag):
    return bag.dags["job_sources_fetch"]


def test_there_is_one_independent_fetch_task_per_source(sources_dag) -> None:
    from jobpdf.ingestion.sources.registry import SOURCE_NAMES

    assert sorted(t.task_id for t in sources_dag.tasks) == sorted(
        f"fetch_{name}" for name in SOURCE_NAMES)
    assert all(not t.upstream_task_ids and not t.downstream_task_ids for t in sources_dag.tasks)


def test_the_sources_dag_never_catches_up_overlaps_or_retries(sources_dag) -> None:
    assert sources_dag.catchup is False and sources_dag.max_active_runs == 1
    for task in sources_dag.tasks:
        assert task.retries == 0 and task.execution_timeout == timedelta(minutes=10)


def test_the_sources_schedule_always_finds_the_cache_old_enough(sources_dag) -> None:
    from jobpdf.ingestion.sources.runner import SCHEDULED_MAX_AGE_S

    schedule = croniter(sources_dag.schedule_interval, datetime(2026, 10, 1, tzinfo=timezone.utc))
    runs = [schedule.get_next(datetime) for _ in range(30)]
    gaps = [(b - a).total_seconds() for a, b in zip(runs, runs[1:], strict=False)]

    assert min(gaps) > SCHEDULED_MAX_AGE_S
    assert 86400 / min(gaps) <= 3


def test_the_sources_dag_runs_off_the_hour_so_it_does_not_start_with_remotive(
        sources_dag, dag) -> None:
    first = croniter(sources_dag.schedule_interval, datetime(2026, 10, 1, tzinfo=timezone.utc))
    other = croniter(dag.schedule_interval, datetime(2026, 10, 1, tzinfo=timezone.utc))

    assert first.get_next(datetime) != other.get_next(datetime)


def test_a_task_runs_its_own_source_and_reports_a_summary(sources_dag, monkeypatch) -> None:
    from jobpdf.ingestion.sources import runner

    seen = []
    monkeypatch.setattr(runner, "run_source",
                        lambda adapter: seen.append(adapter.name) or {"source": adapter.name})
    for key in ("ADZUNA_APP_ID", "ADZUNA_APP_KEY", "JOOBLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    results = {t.task_id: t.python_callable() for t in sources_dag.tasks}

    assert seen == [t.task_id.removeprefix("fetch_") for t in sources_dag.tasks]
    assert results == {t.task_id: {"source": t.task_id.removeprefix("fetch_")}
                       for t in sources_dag.tasks}


# --- vacancy_ingest (JM-33) ---------------------------------------------------------------


@pytest.fixture(scope="module")
def ingest_dag(bag):
    return bag.dags["vacancy_ingest"]


def test_ingestion_is_one_task_that_never_catches_up_overlaps_or_retries(ingest_dag) -> None:
    assert [t.task_id for t in ingest_dag.tasks] == ["ingest"]
    assert ingest_dag.catchup is False and ingest_dag.max_active_runs == 1
    task = ingest_dag.get_task("ingest")
    assert task.retries == 0 and task.execution_timeout == timedelta(hours=1)


def test_ingestion_runs_after_both_fetch_dags_every_time(ingest_dag, dag, sources_dag) -> None:
    def runs(d):
        it = croniter(d.schedule_interval, datetime(2026, 10, 1, tzinfo=timezone.utc))
        return [it.get_next(datetime) for _ in range(9)]

    fetch_runs = sorted(runs(dag) + runs(sources_dag))
    for ingest_run in runs(ingest_dag)[1:]:
        before = [r for r in fetch_runs if r < ingest_run]
        assert before and ingest_run - before[-1] >= timedelta(minutes=20)
        assert ingest_run - before[-1] <= timedelta(minutes=40)


def test_the_ingest_task_runs_the_ingestion_and_returns_its_summary(
        ingest_dag, monkeypatch) -> None:
    from jobpdf.ingestion import vacancies

    monkeypatch.setattr(vacancies, "run_ingest", lambda: {"sources": ["summary"]})

    assert ingest_dag.get_task("ingest").python_callable() == {"sources": ["summary"]}
