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
    assert set(bag.dags) == {"remotive_fetch"}


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
