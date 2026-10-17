"""Airflow DAG: store the fetched postings as normalized, embedded vacancies (JM-33).

Runs after the fetch DAGs (``remotive_fetch`` at :00, ``job_sources_fetch`` at :20) and reads
what they cached; it never fetches. One task, because the normalization service and the
embedding model are heavy to load and are built once per run. Inside it every source is
handled separately: a failing source never stops the others, and the task is marked failed
at the end if any source failed. All logic lives in ``jobpdf.ingestion.vacancies``.

``retries=0``: reruns are harmless (nothing unchanged is reprocessed) but an outage would
only be retried in vain, and the next scheduled run picks the postings up anyway.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

try:  # Airflow 3
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2
    from airflow.decorators import dag, task

SCHEDULE = "50 */8 * * *"
START_DATE = datetime(2026, 10, 1, tzinfo=timezone.utc)


@dag(
    dag_id="vacancy_ingest",
    schedule=SCHEDULE,
    start_date=START_DATE,
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 0, "execution_timeout": timedelta(hours=1)},
    tags=["jobpdf", "ingestion"],
)
def vacancy_ingest():
    @task
    def ingest() -> dict:
        from jobpdf.ingestion.vacancies import run_ingest  # late: keeps DAG parsing fast

        return run_ingest()

    ingest()


vacancy_ingest()
