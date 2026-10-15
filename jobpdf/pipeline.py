"""CV pipeline orchestration (JM-28): uploaded bytes in, ranked matches out.

    parse -> sections -> extraction -> grounding -> normalization -> embedding -> ranking

``analyze_cv`` is the one function both front doors call: the JSON API (JM-29) and the
web page (JM-41). It holds no HTTP, auth or response formatting (JM-29, JM-30); it only
chains the stages, collects what they report and says what produced the result.

Inputs. ``content`` and ``filename`` are the upload; only the filename's extension is
used (and never logged). The bytes go to a private directory inside ``jobpdf/temp/``
(git-ignored) because the parsers read paths; the directory is removed on every exit,
including errors. A crash can still leave one behind, so each call first removes
directories there that are older than an hour. Vacancies
are an argument, not loaded here, so the demo can run on synthetic vacancies while the
vacancy tables (JM-38) and ingestion (JM-31 to JM-33) are built.

Failure rules (decided by the implementer, to be reviewed):

- Hard failure, ``PipelineError(stage, code)``: the file cannot be used (unsupported
  type, too large, corrupt, encrypted, too many pages, no text), or extraction fails
  (no LLM key, invalid output twice, any error from the LLM call). Without a profile
  there is nothing to rank.
- Soft failure, the result carries a ``PipelineWarning``: unrecognised skills and an
  unreachable index or LLM tier (matches are reported with their own codes), failed or
  unavailable embeddings (ranking falls back to skills only), an empty vacancy list,
  and everything the stages already report as warnings.

Privacy. Nothing is stored: the extraction cache is off (it would write the profile to
disk), the temporary file is deleted (see the cleanup above), and logs carry counts,
codes and timings only.
Extraction warnings are free text that embeds CV text (job titles, quote previews), so
they are turned into stable codes here and the text is dropped. Exceptions from the LLM
call are replaced by a code without chaining, because their messages can hold CV text.

Threading. The function is synchronous and keeps no state, so a FastAPI ``def``
endpoint (run in its threadpool) can call it concurrently. Pass ``normalizer`` and
``scorer`` built once at startup (``get_normalization_service()``, ``select_scorer()``);
the defaults build the shared normalizer on first use and rank with the linear scorer.
"""

from __future__ import annotations

import datetime as dt
import logging
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from jobpdf.extraction.extract import ExtractionError, extract_profile
from jobpdf.extraction.grounding import GroundingResult, ground
from jobpdf.extraction.llm import LLMResponseError, MissingAPIKeyError, ToolCaller
from jobpdf.extraction.models import ParsedDocument, ParseError, Section
from jobpdf.extraction.parse_docx import NO_TEXT_WARNING, TEXT_BOX_WARNING
from jobpdf.extraction.parse_pdf import NO_TEXT_LAYER_WARNING
from jobpdf.extraction.parser import parse
from jobpdf.extraction.prompt import mask_pii
from jobpdf.extraction.schema import CandidateProfile
from jobpdf.extraction.sections import (
    NO_HEADINGS_WARNING,
    STYLE_GUARD_WARNING,
    STYLE_SKIPPED_WARNING,
    split_sections,
)
from jobpdf.normalization.embedder import TextEmbedder
from jobpdf.normalization.match_cache import SkillMatch
from jobpdf.normalization.service import (
    NormalizationService,
    ServiceVersions,
    get_normalization_service,
)
from jobpdf.ranking.models import CandidateSkills, VacancyFeatures
from jobpdf.ranking.score import RankingResult, Scorer, VacancyAdjustment, rank_vacancies
from jobpdf.ranking.semantic import CandidateVectors, embed_candidate

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # the description's limit; JM-30 should also enforce it early
ALLOWED_SUFFIXES = (".pdf", ".docx")
TEMP_DIR = Path(__file__).parent / "temp"  # where uploads are written while they are parsed
TEMP_PREFIX = "jobpdf-cv-"
STALE_AFTER_SECONDS = 60 * 60

Stage = Literal[
    "parse", "sections", "extraction", "grounding", "normalization", "embedding", "ranking"
]
STAGES: tuple[Stage, ...] = (
    "parse", "sections", "extraction", "grounding", "normalization", "embedding", "ranking",
)

# Hard-failure codes. The parse ones are ParseError reasons ("encrypted", "corrupt",
# "unsupported_type", "too_large", "too_many_pages") passed through unchanged.
NO_TEXT = "no_text"
LLM_NOT_CONFIGURED = "llm_not_configured"
LLM_ERROR = "llm_error"
EXTRACTION_INVALID = "extraction_invalid"

# Soft-failure codes added by the pipeline itself.
EMBEDDING_FAILED = "embedding_failed"
NO_VACANCIES = "no_vacancies"
EXTRACTION_WARNING = "extraction_warning"

PARSE_WARNING_CODES = {
    NO_TEXT_LAYER_WARNING: "no_text_layer",
    TEXT_BOX_WARNING: "text_boxes_skipped",
    NO_TEXT_WARNING: "no_text",
}
SECTION_WARNING_CODES = {
    NO_HEADINGS_WARNING: "no_section_headings",
    STYLE_SKIPPED_WARNING: "heading_style_skipped",
    STYLE_GUARD_WARNING: "heading_style_ambiguous",
}
# JM-11's postprocess warnings are sentences that embed CV text; these fragments of
# their fixed wording are all that is looked at.
EXPERIENCE_WARNING_FRAGMENTS = (
    (": no start date", "experience_no_start_date"),
    (": no end date", "experience_no_end_date"),
    (": ends before it starts", "experience_ends_before_start"),
)


class PipelineError(Exception):
    """The CV cannot be analysed. ``stage`` and ``code`` are stable; the message never
    holds CV text, so it is safe to log and to show."""

    def __init__(self, stage: Stage, code: str) -> None:
        super().__init__(f"{stage}: {code}")
        self.stage: Stage = stage
        self.code: str = code


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PipelineWarning(_Frozen):
    """Something that reduced the result without stopping it: a stable code and how
    many items it affected. Never holds CV text."""

    stage: Stage
    code: str
    count: int = 1


class PipelineVersions(_Frozen):
    """What produced the result (the API reports these next to it)."""

    extraction_prompt: str
    extraction_model: str
    profile_schema: str
    normalization: ServiceVersions
    embedding: str | None  # None when embeddings were unavailable
    candidate_text: str | None
    ranking: str
    scorer: str


class PipelineResult(_Frozen):
    profile: CandidateProfile
    skill_matches: dict[str, SkillMatch]  # profile path -> match, incl. unrecognised skills
    candidate_skills: CandidateSkills
    grounding: GroundingResult
    ranking: RankingResult
    warnings: list[PipelineWarning]
    versions: PipelineVersions
    timings_ms: dict[str, int]  # per stage, plus "total"
    input_tokens: int  # extraction LLM usage
    output_tokens: int


# --- entry point -------------------------------------------------------------------


def analyze_cv(
    content: bytes,
    filename: str,
    vacancies: Sequence[VacancyFeatures],
    *,
    caller: ToolCaller | None = None,
    normalizer: NormalizationService | None = None,
    embedder: TextEmbedder | None = None,
    scorer: Scorer | None = None,
    adjustments: Mapping[str, VacancyAdjustment] | None = None,
    today: dt.date | None = None,
    max_bytes: int = MAX_UPLOAD_BYTES,
    temp_dir: Path | None = None,
) -> PipelineResult:
    """Run one CV through every stage and rank ``vacancies`` for it.

    Raises ``PipelineError`` when the CV cannot be used; everything else degrades into
    ``PipelineResult.warnings``. ``caller``, ``normalizer`` and ``embedder`` default to
    the real services; tests pass fakes. ``temp_dir`` defaults to ``TEMP_DIR``. Nothing
    is stored.
    """
    started = time.perf_counter()
    timings: dict[str, int] = {}
    warnings = _Warnings()

    with _timed(timings, "parse"):
        doc = _parse(content, filename, max_bytes, temp_dir if temp_dir is not None else TEMP_DIR)
    warnings.add_many("parse", (PARSE_WARNING_CODES.get(w, "parse_warning") for w in doc.warnings))
    if not doc.full_text.strip():
        raise PipelineError("parse", NO_TEXT)

    with _timed(timings, "sections"):
        sections, section_warnings = split_sections(doc)
    warnings.add_many(
        "sections", (SECTION_WARNING_CODES.get(w, "sections_warning") for w in section_warnings)
    )

    with _timed(timings, "extraction"):
        extraction = _extract(doc, sections, caller, today)
    profile = extraction.profile
    warnings.add_many("extraction", (classify_extraction_warning(w) for w in extraction.warnings))

    with _timed(timings, "grounding"):
        masked_text, _ = mask_pii(doc.full_text)  # what the LLM saw, same length as the CV
        grounding = ground(profile, doc, sections, searched_text=masked_text)
    warnings.add_many("grounding", (w.code for w in grounding.warnings))

    normalizer = normalizer if normalizer is not None else get_normalization_service()
    with _timed(timings, "normalization"):
        skills = normalizer.normalize_profile(profile)
    candidate_skills = CandidateSkills.from_skill_matches(skills.matches)
    for match in skills.matches.values():
        warnings.add_many("normalization", match.warnings)

    vectors: CandidateVectors | None = None
    with _timed(timings, "embedding"):
        try:
            vectors = embed_candidate(profile, embedder)
        except Exception as exc:  # a missing model must not lose the skills-only ranking
            log.warning("%s: candidate embedding failed (%s); ranking on skills only",
                        EMBEDDING_FAILED, type(exc).__name__)
            warnings.add("embedding", EMBEDDING_FAILED)
    if vectors is not None:
        warnings.add_many("embedding", vectors.warnings)

    if not vacancies:
        warnings.add("ranking", NO_VACANCIES)
    options = {} if scorer is None else {"scorer": scorer}
    with _timed(timings, "ranking"):
        ranking = rank_vacancies(candidate_skills, vectors, vacancies, adjustments, **options)
    warnings.add_many("ranking", ranking.warnings)

    timings["total"] = round((time.perf_counter() - started) * 1000)
    result = PipelineResult(
        profile=profile,
        skill_matches=skills.matches,
        candidate_skills=candidate_skills,
        grounding=grounding,
        ranking=ranking,
        warnings=warnings.collect(),
        versions=PipelineVersions(
            extraction_prompt=extraction.prompt_version,
            extraction_model=extraction.model,
            profile_schema=profile.schema_version,
            normalization=skills.versions,
            embedding=vectors.embedding_version if vectors is not None else None,
            candidate_text=vectors.text_version if vectors is not None else None,
            ranking=ranking.ranking_version,
            scorer=ranking.scorer_name,
        ),
        timings_ms=timings,
        input_tokens=extraction.input_tokens,
        output_tokens=extraction.output_tokens,
    )
    _log_summary(result)
    return result


def classify_extraction_warning(text: str) -> str:
    """A stable code for one of JM-11's free-text warnings, without any of its text."""
    if text.startswith("quote not found:"):
        return "quote_not_found"
    if text.startswith("possible personal data in output at"):
        return "possible_personal_data"
    if text.startswith("experience "):
        for fragment, code in EXPERIENCE_WARNING_FRAGMENTS:
            if fragment in text:
                return code
    return EXTRACTION_WARNING


def remove_stale_temp(
    temp_dir: Path | None = None, max_age_seconds: float = STALE_AFTER_SECONDS
) -> int:
    """Delete upload directories a crashed process left behind; returns how many.

    Only directories with this module's prefix are touched, and only once they are older
    than ``max_age_seconds`` (a request takes seconds, so a live one is never removed).
    """
    temp_dir = temp_dir if temp_dir is not None else TEMP_DIR
    removed = 0
    cutoff = time.time() - max_age_seconds
    for entry in temp_dir.glob(f"{TEMP_PREFIX}*"):
        try:
            if entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        except OSError:  # raced with another cleanup
            continue
    if removed:
        log.warning("removed %d stale upload directories from the temp folder", removed)
    return removed


# --- stages that need more than one call -------------------------------------------


def _parse(content: bytes, filename: str, max_bytes: int, temp_dir: Path) -> ParsedDocument:
    """Parse upload bytes through a private temporary file that is always removed."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise PipelineError("parse", "unsupported_type")
    if len(content) > max_bytes:
        raise PipelineError("parse", "too_large")
    temp_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_temp(temp_dir)
    with tempfile.TemporaryDirectory(prefix=TEMP_PREFIX, dir=temp_dir) as directory:
        path = Path(directory) / f"cv{suffix}"  # a fixed name: the upload's own never touches disk
        path.write_bytes(content)
        try:
            return parse(path)
        except ParseError as exc:
            raise PipelineError("parse", exc.reason) from None


def _extract(doc: ParsedDocument, sections: list[Section], caller: ToolCaller | None,
             today: dt.date | None):
    """Extraction with the disk cache off and every failure reduced to a code."""
    try:
        return extract_profile(doc, sections, caller, use_cache=False, today=today)
    except MissingAPIKeyError:
        raise PipelineError("extraction", LLM_NOT_CONFIGURED) from None
    except ExtractionError:
        raise PipelineError("extraction", EXTRACTION_INVALID) from None
    except LLMResponseError:
        raise PipelineError("extraction", LLM_ERROR) from None
    except Exception as exc:  # SDK errors after its own retries; only the type is kept
        log.warning("%s: extraction call failed (%s)", LLM_ERROR, type(exc).__name__)
        raise PipelineError("extraction", LLM_ERROR) from None


# --- helpers -----------------------------------------------------------------------


class _Warnings:
    """Counts (stage, code) pairs so a hundred unmatched skills are one warning."""

    def __init__(self) -> None:
        self._counts: Counter[tuple[Stage, str]] = Counter()

    def add(self, stage: Stage, code: str) -> None:
        self._counts[(stage, code)] += 1

    def add_many(self, stage: Stage, codes: Iterable[str]) -> None:
        for code in codes:
            self.add(stage, code)

    def collect(self) -> list[PipelineWarning]:
        ordered = sorted(self._counts.items(), key=lambda kv: (STAGES.index(kv[0][0]), kv[0][1]))
        return [PipelineWarning(stage=stage, code=code, count=count)
                for (stage, code), count in ordered]


@contextmanager
def _timed(timings: dict[str, int], stage: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = round((time.perf_counter() - started) * 1000)


def _log_summary(result: PipelineResult) -> None:
    """Counts, codes and timings only: never a skill, a title, a quote or a filename."""
    unrecognised = sum(1 for m in result.skill_matches.values() if m.concept_id is None)
    log.info(
        "analysed a CV: skills=%d unrecognised=%d vacancies=%d bands=%s warnings=%s "
        "tokens=%d/%d timings_ms=%s",
        len(result.skill_matches), unrecognised,
        len(result.ranking.results) + result.ranking.hidden_count,
        {band.value: n for band, n in result.ranking.band_counts.items()},
        [f"{w.stage}:{w.code}x{w.count}" for w in result.warnings],
        result.input_tokens, result.output_tokens, result.timings_ms,
    )
