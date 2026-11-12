"""The API response contract (JM-29): what a caller gets back for one analysed CV.

``build_response(result)`` turns the pipeline's internal ``PipelineResult`` (JM-28) into
``AnalysisResponse``: the profile, ranked matches with bands, score components and
matched / missing skills, insights (JM-42, converted by ``insights_out``), warnings as
``code`` + ``detail``, and the versions that produced them. It holds no HTTP, auth or
file handling (JM-30, the app tickets).

Why a separate model. The internal result carries quotes, grounding offsets and scorer
internals that callers should not depend on. This contract is the public one: models are
frozen and reject unknown fields, and ``SCHEMA_VERSION`` changes when a field is removed
or changes meaning (adding an optional field does not).

Warnings. A ``code`` is stable and machine-readable; a ``detail`` is a fixed English
sentence from ``WARNING_DETAILS`` and never contains CV text, so it is safe to show.
An unknown code still gets a generic detail rather than being dropped.

Pagination. Nothing is stored between requests, so a page cannot be fetched without
re-running the pipeline. By default one response carries every match (``page_size=None``);
``page`` and ``page_size`` slice the same ordered list for callers that want it.
Hidden vacancies (eligibility checks) are never in ``matches``; they are counted.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from jobpdf.normalization.match_cache import SkillMatch
from jobpdf.normalization.service import ServiceVersions
from jobpdf.pipeline import PipelineResult, PipelineVersions
from jobpdf.ranking.insights import InsightsResult
from jobpdf.ranking.models import SkillRef
from jobpdf.ranking.score import Band, FitResult

SCHEMA_VERSION = "1"
GENERIC_DETAIL = "Something reduced the quality of this result."

# One fixed sentence per warning code. Codes come from the pipeline (JM-28) and the stages
# it runs; a test checks that every code those modules define has an entry.
WARNING_DETAILS: dict[str, str] = {
    # reading the file
    "no_text_layer": "The PDF has no text layer (likely a scan), so little could be read.",
    "text_boxes_skipped": "Text inside floating text boxes was not read.",
    "no_text": "No text could be found in the file.",
    "parse_warning": "The file was read with a problem.",
    "no_section_headings": "No section headings were found, so the CV was read as one block.",
    "heading_style_skipped": "Headings were found by wording only, not by style.",
    "heading_style_ambiguous": "Heading styles were ambiguous, so some sections may be merged.",
    "sections_warning": "The CV's sections could not be fully recognised.",
    # extraction and grounding
    "quote_not_found": "Some extracted details could not be found in the CV text.",
    "possible_personal_data": "Extracted details may contain personal data.",
    "experience_no_start_date": "A job has no start date and is not counted in total experience.",
    "experience_no_end_date": "A job has no end date and is not counted in total experience.",
    "experience_ends_before_start": "A job ends before it starts and is not counted.",
    "extraction_warning": "Some details of the CV could not be checked.",
    "ungrounded_quote": "Some extracted details could not be located in the CV.",
    "fuzzy_quote": "Some extracted details only approximately match the CV text.",
    "ambiguous_quote": "Some extracted details appear in several places in the CV.",
    "searched_text_length_mismatch": "Quote locations may be off: the checked text changed length.",
    # skill normalization
    "empty_skill": "An empty skill was ignored.",
    "ambiguous_alias": "Some skill names could mean more than one skill.",
    "no_vector_threshold": "Similar-skill matching was skipped because it is not calibrated.",
    "no_candidates": "No similar skills were found for some skill names.",
    "llm_unavailable": "The language model was unavailable; some skills were matched less well.",
    "index_unavailable": "The skill index was unavailable; some skills could not be recognised.",
    "llm_choice_not_in_candidates": "The language model chose a skill it was not offered; ignored.",
    "llm_item_missing": "The language model skipped some skills; they were left unmatched.",
    "llm_invalid_output": "The language model returned an unusable answer for some skills.",
    # semantic similarity and embeddings
    "embedding_failed": "Experience similarity could not be computed; matches use skills only.",
    "empty_experience_item": "A job with no text was ignored when comparing experience.",
    "no_experience_text": "The CV has no experience text to compare.",
    "truncated": "Some text was too long and was shortened before comparing.",
    "no_responsibilities": "The vacancy lists no responsibilities to compare.",
    "no_vacancy_embedding": "The vacancy has not been prepared for experience comparison.",
    "stale_vacancy_embedding": "The vacancy was prepared with an older model and was not compared.",
    "stale_vacancy_text": "The vacancy was prepared with an older format and was not compared.",
    "no_candidate_text": "There was no text from the CV to compare with this vacancy.",
    "dimension_mismatch": "The vacancy and the CV were prepared with incompatible models.",
    "uncalibrated": "Experience similarity is not calibrated yet, so it is only approximate.",
    # skill overlap
    "duplicate_requirement": "The vacancy lists the same skill more than once.",
    "no_vacancy_skills": "The vacancy lists no skills, so skills could not be compared.",
    "no_candidate_skills": "No skills were recognised in the CV.",
    # score and ranking
    "semantic_missing": "The fit uses skills only: experience similarity was not available.",
    "skill_missing": "The fit uses experience similarity only: skills could not be compared.",
    "unscored": "This vacancy could not be scored.",
    "band_without_semantic": "The band was set from skills alone.",
    "semantic_unavailable": "Experience similarity was unavailable for all vacancies.",
    "duplicate_vacancy_id": "Two vacancies had the same id; only one was ranked.",
    "no_vacancies": "There were no vacancies to rank.",
}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --- pieces ------------------------------------------------------------------------


class WarningOut(_Frozen):
    code: str
    detail: str
    stage: str | None = None  # pipeline stage, when the warning belongs to one
    count: int = 1  # how many items it affected


class SkillOut(_Frozen):
    name: str  # as written in the CV or vacancy
    label: str | None  # the standard skill name; None when not recognised
    concept_id: str | None
    recognised: bool
    low_confidence: bool = False
    matched_by: str | None = None  # "exact" | "vector" | "llm"; None for vacancy skills


class JobOut(_Frozen):
    title: str
    company: str | None
    location: str | None
    start: str | None
    end: str | None
    description: str | None
    skills: list[SkillOut]


class EducationOut(_Frozen):
    institution: str | None
    degree: str | None
    field: str | None
    start: str | None
    end: str | None


class LanguageOut(_Frozen):
    language: str
    level: str | None


class CertificationOut(_Frozen):
    name: str
    issuer: str | None
    year: str | None


class ProfileOut(_Frozen):
    summary: str | None
    skills: list[SkillOut]
    experience: list[JobOut]
    education: list[EducationOut]
    languages: list[LanguageOut]
    certifications: list[CertificationOut]
    total_experience_months: int | None


class VacancyOut(_Frozen):
    id: str
    source: str
    title: str
    company: str | None
    url: str | None
    posted_at: datetime | None
    location: str | None


class ComponentsOut(_Frozen):
    skill: float | None  # weighted skill overlap, 0..1
    required_coverage: float | None  # share of required skills the CV has, 0..1
    semantic: float | None  # experience similarity, calibrated 0..1
    semantic_raw: float | None  # cosine before rescaling


class SkillGroupOut(_Frozen):
    matched: list[SkillOut]
    missing: list[SkillOut]
    matched_count: int
    count: int


class MatchOut(_Frozen):
    rank: int  # 1-based position among all visible matches, not within the page
    vacancy: VacancyOut
    fit: float | None
    band: Band
    capped_from: Band | None  # the band before an eligibility cap; None when not capped
    components: ComponentsOut
    required: SkillGroupOut
    preferred: SkillGroupOut
    adjustment_reasons: list[str]
    warnings: list[WarningOut]


class Pagination(_Frozen):
    page: int
    page_size: int
    total: int  # visible matches
    pages: int


class WithinReach(_Frozen):
    vacancy_id: str
    missing: list[SkillOut]  # the one or two skills between the CV and a strong fit
    fit_now: float | None = None
    fit_after: float | None = None  # the fit with ``missing`` added
    required_matched_count: int | None = None  # required skills the CV has now
    required_count: int | None = None


class SkillToLearn(_Frozen):
    skill: SkillOut
    unlocks: int  # strong matches this skill would add
    vacancy_ids: list[str]  # the vacancies it would make strong
    required_in: int | None = None  # how many of the closest vacancies require it


class InDemand(_Frozen):
    skill: SkillOut
    count: int  # how many of the closest vacancies require it


class Insights(_Frozen):
    """JM-42's insights (``jobpdf.ranking.insights``), built by ``insights_out``.

    Empty with ``computed=False`` when they were not computed. ``working_set_size`` is
    how many of the closest vacancies they look at ("of your 50 closest").
    """

    computed: bool = False
    working_set_size: int | None = None
    within_reach: list[WithinReach] = Field(default_factory=list)
    skills_to_learn: list[SkillToLearn] = Field(default_factory=list)
    in_demand: list[InDemand] = Field(default_factory=list)


class Scoring(_Frozen):
    """How the fit was computed, for the 'how the fit is scored' section."""

    scorer: str
    weights: dict[str, float]
    thresholds: dict[str, float]
    semantic_calibrated: bool | None


class ExtractionVersions(_Frozen):
    prompt: str
    model: str
    profile_schema: str


class Versions(_Frozen):
    """What produced the result; results are only comparable when these match."""

    response_schema: str
    extraction: ExtractionVersions
    normalization: ServiceVersions
    embedding: str | None
    candidate_text: str | None
    ranking: str
    scorer: str


class Meta(_Frozen):
    timings_ms: dict[str, int]
    input_tokens: int
    output_tokens: int


class AnalysisResponse(_Frozen):
    profile: ProfileOut
    matches: list[MatchOut]
    pagination: Pagination
    band_counts: dict[Band, int]  # all visible matches, not just this page
    hidden_count: int
    insights: Insights
    scoring: Scoring
    warnings: list[WarningOut]
    versions: Versions
    meta: Meta


# --- building ----------------------------------------------------------------------


def build_response(
    result: PipelineResult,
    *,
    insights: Insights | None = None,
    page: int = 1,
    page_size: int | None = None,
) -> AnalysisResponse:
    """The public response for one pipeline result.

    ``page_size=None`` returns every match in one page. A page past the end is empty, not
    an error. Raises ValueError for ``page < 1`` or ``page_size < 1``.
    """
    if page < 1:
        raise ValueError("page must be 1 or more")
    if page_size is not None and page_size < 1:
        raise ValueError("page_size must be 1 or more")

    visible = result.ranking.results
    total = len(visible)
    size = page_size if page_size is not None else max(total, 1)
    start = (page - 1) * size
    chosen = visible[start : start + size]
    ranking = result.ranking
    return AnalysisResponse(
        profile=_profile(result),
        matches=[_match(fit, start + i + 1) for i, fit in enumerate(chosen)],
        pagination=Pagination(page=page, page_size=size, total=total,
                              pages=max(math.ceil(total / size), 1)),
        band_counts=dict(ranking.band_counts),
        hidden_count=ranking.hidden_count,
        insights=insights if insights is not None else Insights(),
        scoring=Scoring(scorer=ranking.scorer_name, weights=dict(ranking.weights),
                        thresholds=dict(ranking.thresholds),
                        semantic_calibrated=ranking.semantic_calibrated),
        warnings=[warning_out(w.code, w.stage, w.count) for w in result.warnings],
        versions=_versions(result.versions),
        meta=Meta(timings_ms=dict(result.timings_ms), input_tokens=result.input_tokens,
                  output_tokens=result.output_tokens),
    )


def insights_out(result: InsightsResult) -> Insights:
    """The public form of JM-42's insights."""
    return Insights(
        computed=True,
        working_set_size=result.working_set_size,
        within_reach=[
            WithinReach(vacancy_id=i.vacancy_id, missing=[skill_from_ref(r) for r in i.added],
                        fit_now=i.fit_now, fit_after=i.fit_after,
                        required_matched_count=i.required_matched_count,
                        required_count=i.required_count)
            for i in result.within_reach
        ],
        skills_to_learn=[
            SkillToLearn(skill=skill_from_ref(i.skill), unlocks=i.gain,
                         vacancy_ids=list(i.vacancy_ids), required_in=i.frequency)
            for i in result.skills_to_learn
        ],
        in_demand=[InDemand(skill=skill_from_ref(i.skill), count=i.count)
                   for i in result.in_demand],
    )


def warning_out(code: str, stage: str | None = None, count: int = 1) -> WarningOut:
    return WarningOut(code=code, detail=WARNING_DETAILS.get(code, GENERIC_DETAIL),
                      stage=stage, count=count)


def skill_from_match(match: SkillMatch) -> SkillOut:
    return SkillOut(name=match.raw, label=match.preferred_label,
                    concept_id=match.concept_id, recognised=match.concept_id is not None,
                    low_confidence=match.low_confidence, matched_by=match.tier.value)


def skill_from_ref(ref: SkillRef) -> SkillOut:
    return SkillOut(name=ref.raw, label=ref.label, concept_id=ref.concept_id,
                    recognised=ref.concept_id is not None)


def _profile(result: PipelineResult) -> ProfileOut:
    profile, matches = result.profile, result.skill_matches
    return ProfileOut(
        summary=profile.summary,
        skills=[skill_from_match(matches[f"skills[{i}]"]) for i in range(len(profile.skills))],
        experience=[
            JobOut(
                title=job.title, company=job.company, location=job.location, start=job.start,
                end=job.end, description=job.description,
                skills=[skill_from_match(matches[f"experience[{j}].skills_used[{k}]"])
                        for k in range(len(job.skills_used))],
            )
            for j, job in enumerate(profile.experience)
        ],
        education=[EducationOut(institution=e.institution, degree=e.degree, field=e.field,
                                start=e.start, end=e.end) for e in profile.education],
        languages=[LanguageOut(language=x.language, level=x.level) for x in profile.languages],
        certifications=[CertificationOut(name=c.name, issuer=c.issuer, year=c.year)
                        for c in profile.certifications],
        total_experience_months=profile.total_experience_months,
    )


def _match(fit: FitResult, rank: int) -> MatchOut:
    vacancy, overlap, components = fit.vacancy, fit.overlap, fit.components
    return MatchOut(
        rank=rank,
        vacancy=VacancyOut(id=vacancy.id, source=vacancy.source, title=vacancy.title,
                           company=vacancy.company, url=vacancy.url,
                           posted_at=vacancy.posted_at, location=vacancy.location_raw),
        fit=fit.fit,
        band=fit.band,
        capped_from=fit.band_before_cap if fit.band_before_cap is not fit.band else None,
        components=ComponentsOut(skill=components.skill,
                                 required_coverage=components.required_coverage,
                                 semantic=components.semantic,
                                 semantic_raw=components.semantic_raw),
        required=SkillGroupOut(
            matched=[skill_from_ref(r) for r in overlap.required_matched],
            missing=[skill_from_ref(r) for r in overlap.required_missing],
            matched_count=overlap.required_matched_count, count=overlap.required_count),
        preferred=SkillGroupOut(
            matched=[skill_from_ref(r) for r in overlap.preferred_matched],
            missing=[skill_from_ref(r) for r in overlap.preferred_missing],
            matched_count=overlap.preferred_matched_count, count=overlap.preferred_count),
        adjustment_reasons=list(fit.adjustment_reasons),
        warnings=[warning_out(code, "ranking") for code in _match_codes(fit)],
    )


def _match_codes(fit: FitResult) -> list[str]:
    """Every code that applies to this vacancy, once each, in a stable order."""
    semantic = fit.semantic.warnings if fit.semantic is not None else []
    return list(dict.fromkeys([*fit.warnings, *fit.overlap.warnings, *semantic]))


def _versions(v: PipelineVersions) -> Versions:
    return Versions(
        response_schema=SCHEMA_VERSION,
        extraction=ExtractionVersions(prompt=v.extraction_prompt, model=v.extraction_model,
                                      profile_schema=v.profile_schema),
        normalization=v.normalization, embedding=v.embedding,
        candidate_text=v.candidate_text, ranking=v.ranking, scorer=v.scorer,
    )


def unknown_codes(codes: Iterable[str]) -> set[str]:
    """Codes without an entry in ``WARNING_DETAILS`` (used by tests and monitoring)."""
    return {c for c in codes if c not in WARNING_DETAILS}


__all__ = [
    "SCHEMA_VERSION", "WARNING_DETAILS", "AnalysisResponse", "Insights", "build_response",
    "skill_from_match", "skill_from_ref", "unknown_codes", "warning_out",
]
