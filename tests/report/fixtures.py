"""Hand-built AnalysisResponse fixtures for the report tests: no pipeline, no network."""

from __future__ import annotations

from datetime import date, datetime, timezone

from jobpdf.api.response import (
    AnalysisResponse,
    ComponentsOut,
    ExtractionVersions,
    InDemand,
    Insights,
    JobOut,
    MatchOut,
    Meta,
    Pagination,
    ProfileOut,
    Scoring,
    SkillGroupOut,
    SkillOut,
    SkillToLearn,
    VacancyOut,
    Versions,
    WarningOut,
    WithinReach,
)
from jobpdf.normalization.service import ServiceVersions
from jobpdf.ranking.score import THRESHOLDS, Band

TODAY = date(2026, 10, 16)


def skill(name: str, label: str | None = None, *, recognised: bool = True,
          low_confidence: bool = False) -> SkillOut:
    label = label if label is not None else (name if recognised else None)
    return SkillOut(name=name, label=label if recognised else None,
                    concept_id=f"esco:{(label or name).lower()}" if recognised else None,
                    recognised=recognised, low_confidence=low_confidence,
                    matched_by="exact" if recognised else None)


def job(title: str, *, start: str | None = "2020-01", end: str | None = "present",
        skills: list[SkillOut] | None = None, company: str | None = "Acme") -> JobOut:
    return JobOut(title=title, company=company, location=None, start=start, end=end,
                  description=None, skills=skills or [])


def profile(*, skills: list[SkillOut] | None = None, jobs: list[JobOut] | None = None,
            months: int | None = 54, summary: str | None = None) -> ProfileOut:
    return ProfileOut(summary=summary,
                      skills=skills if skills is not None else [skill("Python"), skill("SQL")],
                      experience=jobs if jobs is not None else [job("Data Engineer")],
                      education=[], languages=[], certifications=[],
                      total_experience_months=months)


def group(matched: list[str], missing: list[str]) -> SkillGroupOut:
    return SkillGroupOut(matched=[skill(n) for n in matched], missing=[skill(n) for n in missing],
                         matched_count=len(matched), count=len(matched) + len(missing))


def match(vid: str, band: Band = Band.STRONG, *, rank: int = 1,
          required: tuple[list[str], list[str]] = (["Python", "SQL"], []),
          preferred: tuple[list[str], list[str]] = ([], []), semantic: float | None = 0.7,
          posted_days_ago: int | None = 2) -> MatchOut:
    posted = (datetime(2026, 10, 16 - posted_days_ago, 9, tzinfo=timezone.utc)
              if posted_days_ago is not None else None)
    return MatchOut(
        rank=rank,
        vacancy=VacancyOut(id=vid, source="remotive", title=f"Job {vid}", company=f"Co {vid}",
                           url=f"https://remotive.com/remote-jobs/{vid}", posted_at=posted,
                           location="Remote, Europe"),
        fit=0.8, band=band, capped_from=None,
        components=ComponentsOut(skill=0.8, required_coverage=0.8, semantic=semantic,
                                 semantic_raw=semantic),
        required=group(*required), preferred=group(*preferred),
        adjustment_reasons=[], warnings=[],
    )


def response(*, matches: list[MatchOut] | None = None, prof: ProfileOut | None = None,
             insights: Insights | None = None, warnings: list[WarningOut] | None = None,
             hidden: int = 0, weights: dict[str, float] | None = None) -> AnalysisResponse:
    matches = matches if matches is not None else [match("a")]
    counts = dict.fromkeys(Band, 0)
    for m in matches:
        counts[m.band] += 1
    return AnalysisResponse(
        profile=prof or profile(),
        matches=matches,
        pagination=Pagination(page=1, page_size=max(len(matches), 1), total=len(matches),
                              pages=1),
        band_counts=counts,
        hidden_count=hidden,
        insights=insights or Insights(),
        scoring=Scoring(scorer="linear", weights=weights or {"skill": 0.7, "semantic": 0.3},
                        thresholds=dict(THRESHOLDS), semantic_calibrated=True),
        warnings=warnings or [],
        versions=Versions(response_schema="1",
                          extraction=ExtractionVersions(prompt="p", model="m",
                                                        profile_schema="1.0"),
                          normalization=ServiceVersions(taxonomy=None, index="i",
                                                        matcher_prompt="m"),
                          embedding=None, candidate_text=None, ranking="1", scorer="linear"),
        meta=Meta(timings_ms={}, input_tokens=0, output_tokens=0),
    )


def insights(*, within=(), learn=(), demand=(), size: int | None = 50) -> Insights:
    return Insights(
        computed=True, working_set_size=size,
        within_reach=[WithinReach(vacancy_id=vid, missing=[skill(n) for n in add],
                                  required_matched_count=have, required_count=total)
                      for vid, add, have, total in within],
        skills_to_learn=[SkillToLearn(skill=skill(n), unlocks=gain, vacancy_ids=[],
                                      required_in=freq) for n, gain, freq in learn],
        in_demand=[InDemand(skill=skill(n), count=c) for n, c in demand],
    )
