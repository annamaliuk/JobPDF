"""Airflow DAG: fetch The Muse, Arbeitnow, Adzuna and Jooble postings on a schedule.

One task per source, so one source failing does not stop the others, and each shows its own
status in Airflow. A source whose key is missing is reported as ``skipped``, not failed.
All logic lives in ``jobpdf.ingestion.sources``; this file only schedules it.

Every 8 hours, like ``remotive_fetch``, offset by 20 minutes. ``retries=0``: a retry would
spend more requests, and each fetch already falls back to its newest cache.
``max_active_runs=1`` keeps two runs from racing on the cache files.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

try:  # Airflow 3
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2
    from airflow.decorators import dag, task

from jobpdf.ingestion.sources.registry import SOURCE_NAMES

SCHEDULE = "20 */8 * * *"
START_DATE = datetime(2026, 10, 1, tzinfo=timezone.utc)


@dag(
    dag_id="job_sources_fetch",
    schedule=SCHEDULE,
    start_date=START_DATE,
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 0, "execution_timeout": timedelta(minutes=10)},
    tags=["jobpdf", "ingestion"],
)
def job_sources_fetch():
    def make(name: str):
        @task(task_id=f"fetch_{name}")
        def fetch() -> dict:
            from jobpdf.ingestion.sources.registry import build_adapter
            from jobpdf.ingestion.sources.runner import run_source

            return run_source(build_adapter(name))

        return fetch()

    for source in SOURCE_NAMES:
        make(source)


job_sources_fetch()
