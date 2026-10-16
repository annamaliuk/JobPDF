"""How far the JM-15 judge can be trusted, measured two ways.

Planted errors: copies of a real extraction with one known mistake each (a
fake skill, a dropped job, a shifted start date, a fake company). The judge
should flag every one; its catch rate per kind is the main trust number. A
plant only counts where the judge called the untouched item "supported" in the
baseline run, so an error the judge already flagged earns it no credit.

Label agreement: on DataTurks resumes (JM-14), whether the judge's verdict on
each job and education entry agrees with a fuzzy match against the human labels.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from rapidfuzz import fuzz

from jobpdf.extraction.eval_dataturks import Resume
from jobpdf.extraction.eval_metrics import MATCH_THRESHOLD, TEXT_FIELDS, match_key
from jobpdf.extraction.judge import JudgeResult, item_field
from jobpdf.extraction.schema import CandidateProfile, Skill
from jobpdf.normalization.text import normalize

PlantKind = Literal["fake_skill", "drop_job", "shift_start", "fake_company"]
PLANT_KINDS: tuple[PlantKind, ...] = ("fake_skill", "drop_job", "shift_start", "fake_company")
START_SHIFT_YEARS = 2
# Skills and companies to plant, tried in order until one is absent from the CV.
FAKE_SKILLS = ("COBOL", "Fortran", "SAP ABAP", "Erlang", "Simulink", "Haskell")
FAKE_COMPANIES = ("Northwind Traders", "Globex Corporation", "Initech", "Umbrella Holdings")


@dataclass(frozen=True)
class PlantedError:
    kind: PlantKind
    profile: CandidateProfile  # the extraction with the mistake in it
    path: str | None  # the item the judge must flag; None for drop_job


def plant_error(
    profile: CandidateProfile, kind: PlantKind, cv_text: str, baseline: JudgeResult
) -> PlantedError | None:
    """One planted copy of ``profile``, or None when this CV offers no fair target."""
    supported = {v.path for v in baseline.verdicts if v.verdict == "supported"}
    text_key = normalize(cv_text)
    copy = profile.model_copy(deep=True)

    if kind == "fake_skill":
        name = next((s for s in FAKE_SKILLS if normalize(s) not in text_key), None)
        if name is None:
            return None
        copy.skills.append(
            Skill(name=name, kind="hard", found_in=["skills_list"], quotes=[name])
        )
        return PlantedError(kind, copy, f"skills[{len(copy.skills) - 1}]")

    jobs = [i for i, _ in enumerate(copy.experience) if f"experience[{i}]" in supported]
    if kind == "drop_job":
        if not jobs:
            return None
        del copy.experience[jobs[0]]
        return PlantedError(kind, copy, None)

    if kind == "shift_start":
        index = next((i for i in jobs if copy.experience[i].start), None)
        if index is None:
            return None
        job = copy.experience[index]
        year, sep, month = job.start.partition("-")
        start = f"{int(year) - START_SHIFT_YEARS}{sep}{month}"
        copy.experience[index] = job.model_copy(update={"start": start})
        return PlantedError(kind, copy, f"experience[{index}]")

    if kind == "fake_company":
        index = next((i for i in jobs if copy.experience[i].company), None)
        company = next((c for c in FAKE_COMPANIES if normalize(c) not in text_key), None)
        if index is None or company is None:
            return None
        job = copy.experience[index]
        copy.experience[index] = job.model_copy(update={"company": company})
        return PlantedError(kind, copy, f"experience[{index}]")

    raise ValueError(f"unknown plant kind {kind!r}")


def caught(plant: PlantedError, judged: JudgeResult, baseline: JudgeResult) -> bool:
    """Did the judge flag the planted mistake?

    A dropped job is caught when the judge reports more missed jobs than in the
    baseline; any other plant when its item is not called "supported".
    """
    if plant.kind == "drop_job":
        def missed_jobs(result: JudgeResult) -> int:
            return sum(1 for m in result.missed if m.field == "experience")

        return missed_jobs(judged) > missed_jobs(baseline)
    verdict = next((v for v in judged.verdicts if v.path == plant.path), None)
    return verdict is not None and verdict.verdict != "supported"


def plant_report(outcomes: Iterable[tuple[PlantKind, bool | None]]) -> dict:
    """Per kind: planted, caught, skipped (no fair target) and the catch rate."""
    planted: Counter = Counter()
    hits: Counter = Counter()
    skipped: Counter = Counter()
    for kind, outcome in outcomes:
        if outcome is None:
            skipped[kind] += 1
            continue
        planted[kind] += 1
        hits[kind] += outcome
    report = {
        kind: {
            "planted": planted[kind],
            "caught": hits[kind],
            "skipped": skipped[kind],
            "catch_rate": _ratio(hits[kind], planted[kind]),
        }
        for kind in PLANT_KINDS
    }
    report["all"] = {
        "planted": sum(planted.values()),
        "caught": sum(hits.values()),
        "skipped": sum(skipped.values()),
        "catch_rate": _ratio(sum(hits.values()), sum(planted.values())),
    }
    return report


def label_agreement(resume: Resume, profile: CandidateProfile, judged: JudgeResult) -> Counter:
    """2x2 counts for jobs and education: judge "supported" vs a label match.

    A job matches when its title or company fuzzy-matches a Designation or
    Companies label; an education entry when its degree or institution matches
    a Degree or College Name label (the JM-14 rules). DataTurks labels are
    incomplete, so "supported" without a label match is common and not by
    itself a judge error.
    """
    gold: dict[str, list[str]] = {label: [] for label in TEXT_FIELDS.values()}
    for span in resume.spans:
        if span.label in gold:
            gold[span.label].append(match_key(resume.text[span.start : span.end]))

    def matches(value: str | None, label: str) -> bool:
        key = match_key(value or "")
        return bool(key) and any(
            fuzz.token_set_ratio(key, g) >= MATCH_THRESHOLD for g in gold[label] if g
        )

    counts: Counter = Counter()
    for verdict in judged.verdicts:
        name = item_field(verdict.path)
        index = int(verdict.path.split("[", 1)[1].rstrip("]"))
        if name == "experience":
            job = profile.experience[index]
            labeled = matches(job.title, TEXT_FIELDS["designation"]) or matches(
                job.company, TEXT_FIELDS["companies"]
            )
        elif name == "education":
            school = profile.education[index]
            labeled = matches(school.degree, TEXT_FIELDS["degree"]) or matches(
                school.institution, TEXT_FIELDS["college"]
            )
        else:
            continue
        judge_side = "supported" if verdict.verdict == "supported" else "flagged"
        counts[f"{judge_side}_{'labeled' if labeled else 'unlabeled'}"] += 1
    return counts


def agreement_report(counts: Counter) -> dict:
    keys = ("supported_labeled", "supported_unlabeled", "flagged_labeled", "flagged_unlabeled")
    report = {key: counts[key] for key in keys}
    total = sum(report.values())
    report["items"] = total
    report["agreement"] = _ratio(counts["supported_labeled"] + counts["flagged_unlabeled"], total)
    # Of the items the labels confirm, how many the judge also calls supported.
    report["supported_when_labeled"] = _ratio(
        counts["supported_labeled"], counts["supported_labeled"] + counts["flagged_labeled"]
    )
    return report


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None
