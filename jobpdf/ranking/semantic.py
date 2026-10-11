"""Semantic similarity between a candidate's jobs and vacancies (JM-23).

Both sides are lists of texts: one per candidate job, and (for now) one per
vacancy (title + responsibilities). Texts are embedded with JM-17's e5 Embedder,
so the prefix, normalization and model revision are exactly the index's. Vacancy
vectors are computed once at ingestion (``embed_vacancy``); candidate vectors are
computed per request and never stored. Raw cosine similarities from e5 sit in a
narrow band, so they are rescaled with a floor and ceiling measured on labelled
pairs (``semantic_calibration.json``), never with hard-coded values.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from jobpdf.normalization.embedder import (
    MODEL_NAME,
    MODEL_REVISION,
    QUERY_PREFIX,
    Embedder,
    TextEmbedder,
    shared_embedder,
)
from jobpdf.ranking.models import VacancyFeatures

log = logging.getLogger(__name__)

# Bump when the way texts are built changes: stored vacancy vectors then go stale.
CANDIDATE_TEXT_VERSION = "1"
VACANCY_TEXT_VERSION = "1"
AGGREGATION = "max"  # or "top2_mean"
AGGREGATIONS = ("max", "top2_mean")
MODEL_MAX_TOKENS = 512  # e5-small's limit; read from the model when it is loaded
MAX_TEXT_CHARS = 1500  # truncation heuristic, only when no tokenizer is reachable
CALIBRATION_PATH = Path(__file__).with_name("semantic_calibration.json")
FALLBACK_SKILLS_PATH = "skills[*]"

# Warning codes.
EMPTY_EXPERIENCE_ITEM = "empty_experience_item"
NO_EXPERIENCE_TEXT = "no_experience_text"
TRUNCATED = "truncated"
NO_RESPONSIBILITIES = "no_responsibilities"
NO_VACANCY_EMBEDDING = "no_vacancy_embedding"
STALE_VACANCY_EMBEDDING = "stale_vacancy_embedding"
STALE_VACANCY_TEXT = "stale_vacancy_text"
NO_CANDIDATE_TEXT = "no_candidate_text"
DIMENSION_MISMATCH = "dimension_mismatch"
UNCALIBRATED = "uncalibrated"

TokenCounter = Callable[[str], int]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CandidateText(_Frozen):
    path: str  # e.g. "experience[2]", or "skills[*]" for the fallback
    text: str
    truncated: bool = False


class CandidateVectors(_Frozen):
    """A candidate's job vectors for one request; never stored."""

    vectors: list[list[float]]
    paths: list[str]  # vectors[i] belongs to paths[i]
    text_version: str
    embedding_version: str
    warnings: list[str]


class SemanticScore(_Frozen):
    score: float | None  # rescaled to 0..1; None when it can't be computed
    raw: float | None  # aggregated cosine before rescaling
    item_scores: list[float]  # raw cosine per candidate job, in job order
    best_item_index: int | None  # index into CandidateVectors.paths
    calibrated: bool
    warnings: list[str]


class Calibration(BaseModel):
    """Measured rescaling bounds plus where they came from."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    floor: float | None = None
    ceil: float | None = None
    provisional: bool = True
    model: str | None = None
    embedding_version: str | None = None
    dataset: str | None = None
    dataset_sha256: str | None = None
    measured_at: str | None = None
    n_related: int | None = None
    n_unrelated: int | None = None

    @property
    def is_valid(self) -> bool:
        return (
            self.floor is not None and self.ceil is not None
            and math.isfinite(self.floor) and math.isfinite(self.ceil)
            and self.ceil > self.floor
        )


class _Job(Protocol):
    title: str
    company: str | None
    description: str | None
    quote: str


class _NamedSkill(Protocol):
    name: str


class _Profile(Protocol):
    experience: Sequence[_Job]
    skills: Sequence[_NamedSkill]


# --- text builders -----------------------------------------------------------------


def build_candidate_texts(
    profile: _Profile,
    count_tokens: TokenCounter | None = None,
    token_limit: int = MODEL_MAX_TOKENS,
) -> tuple[list[CandidateText], list[str]]:
    """One text per job (title, company, description or else quote), with its path.

    Empty jobs are skipped; with no usable job at all, the skill names become one
    fallback text so the candidate still gets a (weaker) semantic signal.
    """
    warnings: list[str] = []
    texts: list[CandidateText] = []
    for j, job in enumerate(profile.experience):
        text = _join([job.title, job.company, job.description or job.quote])
        if not text:
            _add(warnings, EMPTY_EXPERIENCE_ITEM)
            continue
        texts.append(_candidate_text(f"experience[{j}]", text, count_tokens, token_limit))
    if not texts:
        _add(warnings, NO_EXPERIENCE_TEXT)
        skills = ", ".join(s.name.strip() for s in profile.skills if s.name.strip())
        if skills:
            texts.append(_candidate_text(FALLBACK_SKILLS_PATH, skills, count_tokens, token_limit))
    if any(t.truncated for t in texts):
        _add(warnings, TRUNCATED)
    return texts, warnings


def build_vacancy_texts(vacancy: VacancyFeatures) -> list[str]:
    """Exactly one text for now: the title plus the responsibilities."""
    return [_join([vacancy.title, *vacancy.responsibilities])]


def vacancy_text_warnings(
    vacancy: VacancyFeatures,
    count_tokens: TokenCounter | None = None,
    token_limit: int = MODEL_MAX_TOKENS,
) -> list[str]:
    warnings = [] if _join(vacancy.responsibilities) else [NO_RESPONSIBILITIES]
    if any(_is_truncated(t, count_tokens, token_limit) for t in build_vacancy_texts(vacancy)):
        warnings.append(TRUNCATED)
    return warnings


def _join(parts: Sequence[str | None]) -> str:
    return "\n".join(p.strip() for p in parts if p and p.strip())


def _candidate_text(
    path: str, text: str, count_tokens: TokenCounter | None, token_limit: int
) -> CandidateText:
    return CandidateText(path=path, text=text,
                         truncated=_is_truncated(text, count_tokens, token_limit))


def _is_truncated(text: str, count_tokens: TokenCounter | None, token_limit: int) -> bool:
    if count_tokens is not None:
        return count_tokens(text) > token_limit
    return len(text) > MAX_TEXT_CHARS


def _add(warnings: list[str], code: str) -> None:
    if code not in warnings:
        warnings.append(code)


# --- embedding ---------------------------------------------------------------------

def default_embedder() -> Embedder:
    """The process-wide e5 Embedder, the same instance the skill index (JM-20) uses,
    so the model is loaded and held in memory only once."""
    return shared_embedder()


def embedding_version(embedder: TextEmbedder | None = None) -> str:
    """Identifies model, pinned revision and prefix: vectors are only comparable
    when this string is identical."""
    name = getattr(embedder, "model_name", MODEL_NAME)
    revision = getattr(embedder, "revision", MODEL_REVISION)
    prefix = getattr(embedder, "prefix", QUERY_PREFIX)
    return f"{name}@{revision}|{prefix}"


def embed_vacancy(
    vacancy: VacancyFeatures, embedder: TextEmbedder | None = None
) -> VacancyFeatures:
    """A copy of ``vacancy`` with embedding, embedding_version and vacancy_text_version
    filled. Ingestion calls this once per posting; the input is never changed."""
    embedder = embedder if embedder is not None else default_embedder()
    count_tokens, limit = _token_counter(embedder)
    warnings = vacancy_text_warnings(vacancy, count_tokens, limit)
    if warnings:
        log.warning("vacancy %s: %s", vacancy.id, ", ".join(warnings))
    vector = np.asarray(embedder.embed(build_vacancy_texts(vacancy)))[0]
    return vacancy.model_copy(update={
        "embedding": [float(x) for x in vector],
        "embedding_version": embedding_version(embedder),
        "vacancy_text_version": VACANCY_TEXT_VERSION,
    })


def embed_candidate(profile: _Profile, embedder: TextEmbedder | None = None) -> CandidateVectors:
    """All the candidate's job texts embedded in one batch."""
    embedder = embedder if embedder is not None else default_embedder()
    count_tokens, limit = _token_counter(embedder)
    texts, warnings = build_candidate_texts(profile, count_tokens, limit)
    vectors = np.asarray(embedder.embed([t.text for t in texts])) if texts else np.zeros((0, 0))
    return CandidateVectors(
        vectors=[[float(x) for x in row] for row in vectors],
        paths=[t.path for t in texts],
        text_version=CANDIDATE_TEXT_VERSION,
        embedding_version=embedding_version(embedder),
        warnings=warnings,
    )


def _token_counter(embedder: TextEmbedder) -> tuple[TokenCounter | None, int]:
    """Count tokens like the model will (prefix and special tokens included)."""
    model = getattr(embedder, "model", None)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return None, MODEL_MAX_TOKENS
    prefix = getattr(embedder, "prefix", "")
    limit = int(getattr(model, "max_seq_length", None) or MODEL_MAX_TOKENS)
    return (lambda text: len(tokenizer(prefix + text)["input_ids"])), limit


# --- scoring -----------------------------------------------------------------------


def semantic_scores(
    candidate: CandidateVectors,
    vacancies: Sequence[VacancyFeatures],
    calibration: Calibration | None = None,
    aggregation: str = AGGREGATION,
) -> list[SemanticScore]:
    """One SemanticScore per vacancy, in input order, from one matrix product.

    Vectors are unit length, so cosine similarity is the dot product. Whenever
    vectors can't be compared safely (missing, other model, other text version,
    other dimension), the score is None with a warning instead of a wrong number.
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
    started = time.perf_counter()
    calibration = calibration if calibration is not None else CALIBRATION
    calibrated = (calibration.is_valid
                  and calibration.embedding_version == candidate.embedding_version)
    items = np.asarray(candidate.vectors, dtype=np.float64)
    results: list[SemanticScore | None] = [None] * len(vacancies)
    comparable: list[int] = []
    for i, vacancy in enumerate(vacancies):
        problem = _problem(candidate, items, vacancy)
        if problem is None:
            comparable.append(i)
        else:
            results[i] = SemanticScore(score=None, raw=None, item_scores=[],
                                       best_item_index=None, calibrated=calibrated,
                                       warnings=[problem])
    if comparable:
        matrix = np.asarray([vacancies[i].embedding for i in comparable], dtype=np.float64)
        for i, row in zip(comparable, matrix @ items.T, strict=True):
            results[i] = _score_row(row, calibration, calibrated, aggregation)

    final = [r for r in results if r is not None]
    log.info("semantic scores: %d vacancies, %d scored, %d without score, calibrated=%s, "
             "%.0f ms", len(vacancies), len(comparable), len(vacancies) - len(comparable),
             calibrated, (time.perf_counter() - started) * 1000)
    return final


def _problem(
    candidate: CandidateVectors, items: np.ndarray, vacancy: VacancyFeatures
) -> str | None:
    if not candidate.vectors:
        return NO_CANDIDATE_TEXT
    if not vacancy.embedding:
        return NO_VACANCY_EMBEDDING
    if vacancy.embedding_version != candidate.embedding_version:
        return STALE_VACANCY_EMBEDDING
    if vacancy.vacancy_text_version != VACANCY_TEXT_VERSION:
        return STALE_VACANCY_TEXT
    if len(vacancy.embedding) != items.shape[1]:
        return DIMENSION_MISMATCH
    return None


def _score_row(
    row: np.ndarray, calibration: Calibration, calibrated: bool, aggregation: str
) -> SemanticScore:
    item_scores = [float(x) for x in row]
    ranked = sorted(item_scores, reverse=True)
    raw = ranked[0] if aggregation == "max" else sum(ranked[:2]) / len(ranked[:2])
    if calibrated:
        assert calibration.floor is not None and calibration.ceil is not None
        score = _clip((raw - calibration.floor) / (calibration.ceil - calibration.floor))
        warnings: list[str] = []
    else:
        score, warnings = _clip(raw), [UNCALIBRATED]
    return SemanticScore(score=score, raw=raw, item_scores=item_scores,
                         best_item_index=int(np.argmax(row)), calibrated=calibrated,
                         warnings=warnings)


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


# --- calibration -------------------------------------------------------------------


def load_calibration(path: Path = CALIBRATION_PATH) -> Calibration:
    """The measured floor/ceil, or an invalid Calibration (never raises, never invents)."""
    try:
        data: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        calibration = Calibration.model_validate(data)
    except (OSError, ValueError, ValidationError) as exc:
        log.warning("%s: semantic calibration not loaded from %s (%s)", UNCALIBRATED,
                    Path(path).name, type(exc).__name__)
        return Calibration()
    if not calibration.is_valid:
        log.warning("%s: semantic calibration in %s has no usable floor/ceil", UNCALIBRATED,
                    Path(path).name)
    return calibration


CALIBRATION = load_calibration(CALIBRATION_PATH)
