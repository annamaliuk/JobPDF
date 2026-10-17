"""The report page's content (JM-43): the API response turned into plain sentences.

``build_report(response)`` takes the public ``AnalysisResponse`` (JM-29) and returns a
``Report``: strings, numbers and flags that a template prints. No HTML is made here and no
HTML is trusted: everything that came from a CV or a vacancy is printed by Jinja2 with
autoescaping on. The one value that needs more than escaping is a link, so ``safe_url`` lets
only http(s) addresses through.

Sections follow the project description: a profile summary and figures; best matches, each
with "You have 7 of 9 required skills" and what is missing; "within reach" and "skills worth
learning" (JM-42, shown only when the insights were computed); the candidate's skills, how
abbreviations were read and what was not recognised; all matches; plain notes on data
quality; how the fit is scored.

Wording rules: sentences in sentence case, numbers as digits, singular and plural handled,
nothing about the system's internals. Warnings use the fixed ``detail`` sentences of the
response, which never contain CV text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from jobpdf.api.response import AnalysisResponse, MatchOut, SkillOut
from jobpdf.ranking.score import Band

BEST_COUNT = 5  # matches shown as full rows; the rest are in the table
MAX_MISSING_SHOWN = 6
MAX_SEGMENTS = 16  # the meter caps here; the sentence next to it always has the real numbers
MAX_URL_CHARS = 2048
MONTHS_PER_YEAR = 12

BAND_LABELS = {
    Band.STRONG.value: "Strong match",
    Band.PARTIAL.value: "Partial match",
    Band.WEAK.value: "Weak match",
    Band.UNSCORED.value: "Not scored",
}
NO_JOBS = "There are no jobs to compare with your CV yet. Try again later."
NO_REQUIRED = "This job lists no required skills, so your skills could not be compared."


@dataclass(frozen=True)
class Figure:
    value: str
    label: str


@dataclass(frozen=True)
class Job:
    rank: int
    title: str
    company: str | None
    location: str | None
    source: str
    url: str | None  # http(s) only
    posted: str | None
    band: str  # "strong" | "partial" | "weak" | "unscored"
    band_label: str
    fit: str | None  # "82%"
    coverage: str  # "You have 7 of 9 required skills."
    required_short: str  # "7 of 9", or "none listed"
    segments: tuple[bool, ...]  # one per required skill (capped), True = you have it
    missing: str | None  # "Missing: Docker, Kubernetes and 2 more."
    nice_to_have: str | None  # "Nice to have: 1 of 3."
    lowered: str | None  # "Lowered from strong to partial after an eligibility check."


@dataclass(frozen=True)
class SkillLine:
    label: str
    unsure: bool


@dataclass(frozen=True)
class Report:
    summary: str | None
    figures: list[Figure]
    best: list[Job]
    rest: list[Job]  # every match, in rank order (the table), including the best ones
    reach: list[str] = field(default_factory=list)
    learn: list[str] = field(default_factory=list)
    skills: list[SkillLine] = field(default_factory=list)
    readings: list[str] = field(default_factory=list)
    unrecognised: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    scoring: list[str] = field(default_factory=list)
    empty: str | None = None


def build_report(response: AnalysisResponse) -> Report:
    jobs = [_job(m) for m in response.matches]
    by_id = {m.vacancy.id: m for m in response.matches}
    skills = _unique_skills(response.profile.skills)
    return Report(
        summary=_clean(response.profile.summary),
        figures=_figures(response, skills),
        best=jobs[:BEST_COUNT],
        rest=jobs,
        reach=_reach(response, by_id),
        learn=_learn(response),
        skills=[SkillLine(s.label or s.name, s.low_confidence) for s in skills if s.recognised],
        readings=_readings(skills),
        unrecognised=[s.name for s in skills if not s.recognised],
        notes=_notes(response),
        scoring=_scoring(response),
        empty=None if jobs else NO_JOBS,
    )


def safe_url(url: str | None) -> str | None:
    """``url`` when it is an http(s) address with a host, else None. Printing a vacancy's
    address into ``href`` is the one place where escaping alone is not enough: a
    ``javascript:`` address would run when clicked."""
    if not url or len(url) > MAX_URL_CHARS or any(c.isspace() or ord(c) < 32 for c in url):
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    # urlsplit has already lowercased the scheme
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    return url


def plural(count: int, one: str, many: str | None = None) -> str:
    return f"{count} {one if count == 1 else (many or one + 's')}"


def percent(value: float | None) -> str | None:
    """A 0..1 share as a whole percent, halves rounded up (``round`` would send 82.5 to 82)."""
    return None if value is None else f"{math.floor(value * 100 + 0.5)}%"


# --- sections ----------------------------------------------------------------------------


def _clean(text: str | None) -> str | None:
    text = (text or "").strip()
    return text or None


def _figures(response: AnalysisResponse, skills: list[SkillOut]) -> list[Figure]:
    figures = []
    months = response.profile.total_experience_months
    if months is not None:
        figures.append(Figure(_duration(months), "Experience"))
    recognised = sum(1 for s in skills if s.recognised)
    figures.append(Figure(f"{recognised} of {len(skills)}", "Skills recognised"))
    figures.append(Figure(str(response.band_counts.get(Band.STRONG, 0)), "Strong matches"))
    figures.append(Figure(str(response.pagination.total), "Jobs compared"))
    return figures


def _duration(months: int) -> str:
    years, rest = divmod(max(months, 0), MONTHS_PER_YEAR)
    if years == 0:
        return plural(rest, "month")
    if rest == 0:
        return plural(years, "year")
    return f"{plural(years, 'year')} {plural(rest, 'month')}"


def _job(match: MatchOut) -> Job:
    required, vacancy = match.required, match.vacancy
    missing = [_name(s) for s in required.missing]
    shown = missing[:MAX_MISSING_SHOWN]
    hidden = len(missing) - len(shown)
    missing_text = None
    if shown:
        extra = f" and {hidden} more" if hidden else ""
        missing_text = f"Missing: {', '.join(shown)}{extra}."
    nice = match.preferred
    return Job(
        rank=match.rank,
        title=vacancy.title,
        company=vacancy.company,
        location=vacancy.location,
        source=vacancy.source,
        url=safe_url(vacancy.url),
        posted=f"Posted {vacancy.posted_at:%d %b %Y}" if vacancy.posted_at else None,
        band=match.band.value,
        band_label=BAND_LABELS[match.band.value],
        fit=percent(match.fit),
        coverage=_coverage(required.matched_count, required.count),
        required_short=(f"{required.matched_count} of {required.count}" if required.count
                        else "none listed"),
        segments=_segments(required.matched_count, required.count),
        missing=missing_text,
        nice_to_have=f"Nice to have: {nice.matched_count} of {nice.count}." if nice.count else None,
        lowered=(f"Lowered from {match.capped_from.value} to {match.band.value} after an "
                 "eligibility check.") if match.capped_from else None,
    )


def _coverage(have: int, total: int) -> str:
    if total == 0:
        return NO_REQUIRED
    return f"You have {have} of {plural(total, 'required skill')}."


def _segments(have: int, total: int) -> tuple[bool, ...]:
    if total <= MAX_SEGMENTS:
        return (True,) * have + (False,) * (total - have)
    filled = round(have / total * MAX_SEGMENTS)
    return (True,) * filled + (False,) * (MAX_SEGMENTS - filled)


def _name(skill: SkillOut) -> str:
    return skill.label or skill.name


def _unique_skills(skills: list[SkillOut]) -> list[SkillOut]:
    """One entry per skill: recognised ones by concept, the others by their written name."""
    seen: set[str] = set()
    unique = []
    for skill in skills:
        key = skill.concept_id or f"raw:{skill.name.strip().lower()}"
        if key not in seen:
            seen.add(key)
            unique.append(skill)
    return unique


def _readings(skills: list[SkillOut]) -> list[str]:
    """Skills written differently from their standard name: "JS was read as JavaScript"."""
    lines = []
    for skill in skills:
        if skill.recognised and skill.label and skill.name.strip().lower() != skill.label.lower():
            lines.append(f"{skill.name.strip()} was read as {skill.label}.")
    return lines


def _reach(response: AnalysisResponse, matches: dict[str, MatchOut]) -> list[str]:
    if not response.insights.computed:
        return []
    lines = []
    for item in response.insights.within_reach:
        match = matches.get(item.vacancy_id)
        if match is None or not item.missing:
            continue
        names = " and ".join(_name(s) for s in item.missing)
        where = f" at {match.vacancy.company}" if match.vacancy.company else ""
        lines.append(f"{match.vacancy.title}{where}: add {names} to reach a strong match.")
    return lines


def _learn(response: AnalysisResponse) -> list[str]:
    if not response.insights.computed:
        return []
    return [
        f"{_name(item.skill)}: would make {plural(item.unlocks, 'more job')} a strong match."
        for item in response.insights.skills_to_learn
    ]


def _notes(response: AnalysisResponse) -> list[str]:
    notes = []
    for warning in response.warnings:
        suffix = f" (affects {warning.count} items)" if warning.count > 1 else ""
        notes.append(warning.detail + suffix)
    if response.hidden_count:
        notes.append(
            f"{plural(response.hidden_count, 'posting')} left out because your CV suggests "
            "you could not apply for them.")
    return notes


def _scoring(response: AnalysisResponse) -> list[str]:
    weights, limits = response.scoring.weights, response.scoring.thresholds
    skill, experience = weights.get("skill"), weights.get("semantic")
    if skill is not None and experience is not None:
        lines = [f"Each job's fit is {percent(skill)} your skills against the skills it asks "
                 f"for and {percent(experience)} how closely your past work resembles the "
                 "job's duties."]
    else:
        lines = ["Each job's fit is worked out from your skills and your past work."]
    strong, partial = limits.get("strong_coverage"), limits.get("partial_coverage")
    if strong is not None and partial is not None:
        lines.append(
            f"A strong match means you have at least {percent(strong)} of the required skills "
            f"and your experience is not far off. A partial match means at least "
            f"{percent(partial)}. Anything lower is weak.")
    if response.scoring.semantic_calibrated is not True:
        lines.append("The comparison of past work is approximate for now.")
    return lines
