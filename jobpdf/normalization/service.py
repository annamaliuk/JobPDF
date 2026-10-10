"""The one shared place to normalize skills (JM-20).

The CV pipeline (FastAPI) and vacancy ingestion (Airflow) both go through this
module, so they build the matcher the same way and get identical concept IDs for
identical strings. It is an in-process module, not an HTTP service, and adds no
matching behaviour of its own: everything is delegated to JM-18's SkillMatcher.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

import psycopg
from pydantic import BaseModel, ConfigDict

from jobpdf.extraction.llm import AnthropicToolCaller, MissingAPIKeyError, ToolCaller
from jobpdf.normalization import match_prompt
from jobpdf.normalization.db import database_url
from jobpdf.normalization.index import Candidate, SkillIndex
from jobpdf.normalization.match_cache import (
    InMemorySkillMatchCache,
    SkillMatch,
    SkillMatchCache,
    index_version,
)
from jobpdf.normalization.matcher import SkillIndexLike, SkillMatcher

log = logging.getLogger(__name__)

LLM_DISABLED = "disabled (ANTHROPIC_API_KEY missing)"
DEFAULT_PROFILE_CONTEXT = "CV"
INDEX_UNAVAILABLE_VERSION = "unavailable"
WARM_UP_TEXT = "warm-up"


class ServiceVersions(BaseModel):
    """What produced a set of concept IDs; stored next to them (JM-38)."""

    model_config = ConfigDict(frozen=True)

    taxonomy: str | None  # index_meta taxonomy.esco_version, e.g. "v1.2.1"
    index: str  # match_cache.index_version(index_meta), or "unavailable"
    matcher_prompt: str  # MATCHER_PROMPT_VERSION


class ProfileSkills(BaseModel):
    """Every raw skill of a profile with its match, keyed by its path in the profile."""

    matches: dict[str, SkillMatch]  # e.g. "skills[3]", "experience[1].skills_used[0]"
    versions: ServiceVersions


class _SkillNamed(Protocol):
    name: str


class _Job(Protocol):
    skills_used: list[str]


class _Profile(Protocol):
    """CandidateProfile and ExtractedProfile both fit: skills[].name, experience[].skills_used."""

    skills: Sequence[_SkillNamed]
    experience: Sequence[_Job]


class NormalizationService:
    """A thin, shared wrapper around one SkillMatcher.

    Every dependency is passed in. ``index`` is only used by ``status()`` for a
    live database probe; ``llm_model`` is None when the LLM tier is disabled.
    """

    def __init__(
        self,
        matcher: SkillMatcher,
        versions: ServiceVersions,
        *,
        index: object | None = None,
        llm_model: str | None = None,
        cache_type: str = "unknown",
    ) -> None:
        self._matcher = matcher
        self._versions = versions
        self._index = index
        self._llm_model = llm_model
        self._cache_type = cache_type

    @property
    def versions(self) -> ServiceVersions:
        return self._versions

    def normalize_skills(
        self, raws: Sequence[str], context: str | None = None
    ) -> list[SkillMatch]:
        """One SkillMatch per input, in input order."""
        return self._matcher.match_many(list(raws), context)

    def normalize_skill_groups(
        self, groups: Mapping[str, Sequence[str]], context: str | None = None
    ) -> dict[str, list[SkillMatch]]:
        """Several named lists (e.g. a vacancy's required / nice_to_have) in one call.

        One ``match_many`` call means at most one batched LLM call for all groups;
        results are split back with group order and item order kept.
        """
        flat = [raw for raws in groups.values() for raw in raws]
        matches = self.normalize_skills(flat, context) if flat else []
        result: dict[str, list[SkillMatch]] = {}
        at = 0
        for name, raws in groups.items():
            result[name] = matches[at : at + len(raws)]
            at += len(raws)
        return result

    def normalize_profile(
        self, profile: _Profile, context: str | None = DEFAULT_PROFILE_CONTEXT
    ) -> ProfileSkills:
        """Every raw skill string of a CV profile, matched in one call; the profile is untouched."""
        located = list(profile_skill_paths(profile))
        matches = self.normalize_skills([raw for _, raw in located], context) if located else []
        return ProfileSkills(
            matches={path: match for (path, _), match in zip(located, matches, strict=True)},
            versions=self._versions,
        )

    @staticmethod
    def concept_ids(matches: Iterable[SkillMatch]) -> set[str]:
        """Matched IDs only; unmatched results stay in the match lists, just not here."""
        return {m.concept_id for m in matches if m.concept_id is not None}

    def status(self) -> dict[str, str | None]:
        """Health summary for /health (JM-28). Never contains secret values or URLs."""
        return {
            "index": "ok" if _index_reachable(self._index) else "unavailable",
            "taxonomy_version": self._versions.taxonomy,
            "index_version": self._versions.index,
            "matcher_prompt_version": self._versions.matcher_prompt,
            "llm": "enabled" if self._llm_model else LLM_DISABLED,
            "llm_model": self._llm_model,
            "cache": self._cache_type,
        }

    def close(self) -> None:
        close = getattr(self._index, "close", None)
        if callable(close):
            close()


def profile_skill_paths(profile: _Profile) -> Iterable[tuple[str, str]]:
    """(path, raw skill) for every place a JM-11 profile holds a raw skill name."""
    for i, skill in enumerate(profile.skills):
        yield f"skills[{i}]", skill.name
    for j, job in enumerate(profile.experience):
        for k, raw in enumerate(job.skills_used):
            yield f"experience[{j}].skills_used[{k}]", raw


def _index_reachable(index: object | None) -> bool:
    """A live probe, so /health turns red when the database drops after startup."""
    if index is None or getattr(index, "available", True) is False:
        return False
    connection = getattr(index, "connection", None)
    if connection is None:  # nothing to probe (e.g. an in-memory index)
        return True
    try:
        connection.execute("SELECT 1")
    except psycopg.Error:
        return False
    return True


# --- building from the environment ------------------------------------------------


class _UnavailableIndex:
    """Stand-in when the database can't be reached at build time.

    Every lookup raises psycopg.OperationalError, which the matcher already turns
    into ``index_unavailable`` results, so callers degrade instead of crashing.
    """

    available = False

    @property
    def meta(self) -> dict[str, object]:
        return {}

    def lookup_exact(self, raw: str) -> list[Candidate]:
        raise psycopg.OperationalError("skill index database unavailable since startup")

    def search(self, raws: list[str], k: int = 10) -> list[list[Candidate]]:
        raise psycopg.OperationalError("skill index database unavailable since startup")


def build_normalization_service(
    cache: SkillMatchCache | None = None, *, warm_up: bool = True
) -> NormalizationService:
    """Build the service from the environment: DATABASE_URL, ANTHROPIC_API_KEY/_MODEL.

    Never raises for a missing key or an unreachable database: it degrades and
    says so in logs and ``status()``. ``cache`` is where JM-19's Postgres cache
    plugs in. ``warm_up`` loads the e5 model now, so startup pays for it rather
    than the first request (and concurrent first requests can't load it twice).
    """
    started = time.perf_counter()
    index = _connect_index()
    tool_caller = _anthropic_caller()
    cache = cache if cache is not None else InMemorySkillMatchCache()
    if warm_up and not isinstance(index, _UnavailableIndex):
        _warm_up(index)
    service = NormalizationService(
        SkillMatcher(index, tool_caller, cache),
        _versions(index),
        index=index,
        llm_model=tool_caller.model if tool_caller is not None else None,
        cache_type=type(cache).__name__,
    )
    status = service.status()
    log.info("normalization service built in %.0f ms: index=%s llm=%s cache=%s",
             (time.perf_counter() - started) * 1000, status["index"], status["llm"],
             status["cache"])
    return service


def _connect_index() -> SkillIndexLike:
    url = database_url()
    if not url:
        log.warning("DATABASE_URL missing; skill index unavailable")
        return _UnavailableIndex()
    try:
        return SkillIndex.connect(url)
    except psycopg.Error as exc:
        # The URL is never logged: it carries the database password.
        log.warning("skill index database unreachable (%s); matching degrades to "
                    "index_unavailable until the service is rebuilt", type(exc).__name__)
        return _UnavailableIndex()


def _anthropic_caller() -> ToolCaller | None:
    try:
        return AnthropicToolCaller()
    except MissingAPIKeyError:
        log.warning("ANTHROPIC_API_KEY missing; LLM tier disabled (exact and vector tiers only)")
        return None


def _warm_up(index: SkillIndexLike) -> None:
    started = time.perf_counter()
    try:
        index.embedder.embed([WARM_UP_TEXT])  # type: ignore[attr-defined]
    except Exception as exc:  # a missing model must not stop the service; search degrades
        log.warning("embedding model warm-up failed (%s)", type(exc).__name__)
        return
    log.info("embedding model loaded in %.0f ms", (time.perf_counter() - started) * 1000)


def _versions(index: SkillIndexLike) -> ServiceVersions:
    if isinstance(index, _UnavailableIndex):
        return ServiceVersions(taxonomy=None, index=INDEX_UNAVAILABLE_VERSION,
                               matcher_prompt=match_prompt.MATCHER_PROMPT_VERSION)
    meta = index.meta or {}
    taxonomy = meta.get("taxonomy")
    esco_version = taxonomy.get("esco_version") if isinstance(taxonomy, Mapping) else None
    return ServiceVersions(taxonomy=esco_version, index=index_version(meta),
                           matcher_prompt=match_prompt.MATCHER_PROMPT_VERSION)


# --- one instance per process ------------------------------------------------------

_service: NormalizationService | None = None
_service_lock = threading.Lock()


def get_normalization_service() -> NormalizationService:
    """The process-wide service, built on first call (FastAPI startup, or once per Airflow task).

    Thread-safety assumptions (FastAPI may call it concurrently):
    - the first build is guarded by a lock, so it happens once;
    - SkillIndex shares one psycopg 3 connection, which serializes concurrent queries;
    - InMemorySkillMatchCache relies on single dict get/set being atomic under the GIL;
    - the model is loaded at build (warm-up), so no request triggers a lazy double load;
    - SkillMatcher keeps no per-call state on the instance.
    After a database outage the index stays unavailable: /health (``status()``) shows
    it, and a restart or ``reset_normalization_service()`` rebuilds it.
    """
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = build_normalization_service()
    return _service


def reset_normalization_service() -> None:
    """Close and forget the process-wide service (tests, or a rebuild after an outage)."""
    global _service
    with _service_lock:
        service, _service = _service, None
    if service is not None:
        service.close()
