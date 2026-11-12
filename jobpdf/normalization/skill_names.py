"""Skill names in the person's language (JM-46), for the web report.

The report prints every skill by its standard name, and the index holds that name in English.
The same index already holds ESCO's Ukrainian name of each skill (the ``uk`` ``preferred``
row of ``taxonomy_alias``), so Ukrainian names need no new table and no rebuild: this module
reads them once and keeps them in memory.

    choose_language(...)        which language the page is asked for ("en" or "uk")
    IndexSkillNames(load).get() concept id -> Ukrainian name, for the concepts that have one
    postgres_loader(url)        the ``load`` that reads them from the index

A skill without a Ukrainian name keeps its English one; nothing is ever left blank. Nothing here
touches a CV: only concept ids go in and names come out. If the database cannot be read the
report is shown in English, the failure is logged by type only, and the next try is made after
``RETRY_AFTER_S`` so a database that is down is not asked on every request.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping

from psycopg import sql

from jobpdf.normalization.db import connect
from jobpdf.normalization.index_build import DEFAULT_TABLE

log = logging.getLogger(__name__)

ENGLISH = "en"
UKRAINIAN = "uk"
LANGUAGES: tuple[str, ...] = (ENGLISH, UKRAINIAN)
LANGUAGE_FIELD = "lang"
RETRY_AFTER_S = 60.0

NamesLoader = Callable[[], Mapping[str, str]]


def choose_language(requested: str | None) -> str:
    """The language asked for in the form; anything but a known code means English."""
    value = (requested or "").strip().lower()
    return value if value in LANGUAGES else ENGLISH


def preferred_language(accept_language: str | None) -> str:
    """The language to preselect on the upload form, from the browser's ``Accept-Language``:
    Ukrainian when it is the first language listed, otherwise English."""
    if not accept_language:
        return ENGLISH
    first = accept_language.split(",")[0].split(";")[0].strip().lower()
    return UKRAINIAN if first == UKRAINIAN or first.startswith(UKRAINIAN + "-") else ENGLISH


class IndexSkillNames:
    """Ukrainian skill names, loaded on first use and kept for the life of the process.

    ``load`` returns ``{concept id: Ukrainian name}``. It runs at most once at a time, and after
    a failure not again before ``retry_after_s``. The map is replaced whole, never edited, so a
    reader never sees a half-loaded one.
    """

    def __init__(self, load: NamesLoader, *, retry_after_s: float = RETRY_AFTER_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._load = load
        self._retry_after_s = retry_after_s
        self._clock = clock
        self._lock = threading.Lock()
        self._names: Mapping[str, str] | None = None
        self._failed_at: float | None = None

    def get(self) -> Mapping[str, str]:
        """The names, or an empty mapping while they cannot be loaded."""
        names = self._names
        if names is not None:
            return names
        with self._lock:
            if self._names is not None:
                return self._names
            if (self._failed_at is not None
                    and self._clock() - self._failed_at < self._retry_after_s):
                return {}
            try:
                loaded = dict(self._load())
            except Exception as exc:  # the report must still be made, in English
                self._failed_at = self._clock()
                log.warning("ukrainian skill names unavailable (%s)", type(exc).__name__)
                return {}
            self._failed_at = None
            self._names = loaded
            log.info("ukrainian skill names loaded: %d", len(loaded))
            return loaded


def postgres_loader(url: str, table: str = DEFAULT_TABLE) -> NamesLoader:
    """A loader for ``IndexSkillNames`` reading the preferred Ukrainian label of each concept."""
    query = sql.SQL(
        "SELECT canonical_id, min(alias) FROM {} "
        "WHERE lang = %s AND alias_kind = 'preferred' GROUP BY canonical_id"
    ).format(sql.Identifier(table))

    def load() -> dict[str, str]:
        with connect(url) as conn:
            rows = conn.execute(query, [UKRAINIAN]).fetchall()
        return {concept_id: name.strip() for concept_id, name in rows if name and name.strip()}

    return load


def no_names() -> Mapping[str, str]:
    """The names when there is no database: nothing, so every skill keeps its English name."""
    return {}
