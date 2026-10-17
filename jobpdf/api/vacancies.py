"""Where the API gets the vacancies it ranks against (JM-36).

A CV request only reads vacancies, so the app keeps them in memory and reloads them from the
``vacancies`` table (JM-32) at most every ``VACANCY_CACHE_TTL_S`` seconds. The ingestion DAG
writes every 8 hours, so a few minutes of delay costs nothing, and a request never waits for
a database round trip for several thousand rows.

Failure rules (decided by the implementer, to be reviewed):

- A reload that fails keeps serving the previous list (stale beats empty) and tries again
  after ``VACANCY_RETRY_S``. With nothing loaded yet the list is empty, and the pipeline
  then reports its ``no_vacancies`` warning: a low-confidence result, never a crash.
- Only the exception *type* is logged; a database error can carry the connection string.
- One thread reloads at a time; the others wait for it and then use its result.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence

import psycopg

from jobpdf.ingestion.store import VacancyStore
from jobpdf.ranking.models import VacancyFeatures

log = logging.getLogger(__name__)

VACANCY_CACHE_TTL_S = 300
VACANCY_RETRY_S = 30
DB_CONNECT_TIMEOUT_S = 5

Loader = Callable[[], Sequence[VacancyFeatures]]


class VacancyProvider:
    """``get()`` returns the current vacancies, reloading when the copy is old."""

    def __init__(self, load: Loader, *, ttl_s: float = VACANCY_CACHE_TTL_S,
                 retry_s: float = VACANCY_RETRY_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._load = load
        self._ttl_s = ttl_s
        self._retry_s = retry_s
        self._clock = clock
        self._lock = threading.Lock()
        self._items: Sequence[VacancyFeatures] = ()
        self._next_try = 0.0

    def get(self) -> Sequence[VacancyFeatures]:
        with self._lock:
            now = self._clock()
            if now < self._next_try:
                return self._items
            try:
                self._items = tuple(self._load())
            except Exception as exc:
                log.warning("vacancy reload failed (%s); keeping %d vacancies",
                            type(exc).__name__, len(self._items))
                self._next_try = now + self._retry_s
            else:
                log.info("vacancies reloaded: %d", len(self._items))
                self._next_try = now + self._ttl_s
            return self._items


def postgres_loader(url: str) -> Loader:
    """Loads every stored vacancy over a short-lived connection."""

    def load() -> Sequence[VacancyFeatures]:
        with psycopg.connect(url, connect_timeout=DB_CONNECT_TIMEOUT_S) as conn:
            return VacancyStore(conn).load()

    return load
