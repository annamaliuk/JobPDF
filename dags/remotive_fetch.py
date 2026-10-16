"""Airflow DAG: fetch Remotive postings on a schedule (JM-31).

One task today. Extraction (JM-48), normalization, embedding and the upsert (JM-32/33) will
follow it as further tasks. All logic lives in ``jobpdf.ingestion.remotive``; this file only
schedules it, so it stays thin and the logic is tested without Airflow.

Three runs a day (every 8 hours) stay under Remotive's advised 4 requests a day and leave
one spare for a manual fetch. ``retries=0`` is deliberate: a retry would spend another
request, and the fetch already falls back to the newest cache. ``max_active_runs=1`` keeps
two runs from racing on the cache files.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

try:  # Airflow 3
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2
    from airflow.decorators import dag, task

SCHEDULE = "0 */8 * * *"
START_DATE = datetime(2026, 10, 1, tzinfo=timezone.utc)


@dag(
    dag_id="remotive_fetch",
    schedule=SCHEDULE,
    start_date=START_DATE,
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 0, "execution_timeout": timedelta(minutes=5)},
    tags=["jobpdf", "ingestion", "remotive"],
)
def remotive_fetch():
    @task
    def fetch() -> dict:
        from jobpdf.ingestion.remotive import run_fetch  # imported late: keeps DAG parsing fast

        return run_fetch()

    fetch()


remotive_fetch()
