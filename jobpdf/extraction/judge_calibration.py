"""How far the JM-15 judge can be trusted, measured two ways.

Planted errors: copies of a real extraction with known mistakes (a fake skill,
a dropped job, a shifted start date, a fake company). The judge should flag
every one; its catch rate per kind is the main trust number. A plant only
counts where the judge called the untouched item "supported" in the baseline
run, so an error the judge already flagged earns it no credit.

To save judge calls, ``plant_copies`` packs the plants into as few copies as
possible. Within one copy every job carries at most one mistake and every
mistake is checked on its own item, so the plants do not mask each other: a CV
with three or more fair jobs needs one planted copy instead of four.

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
from jobpdf.extraction.schema import CandidateProfile, ExtractedJob, Skill
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
    profile: CandidateProfile  # the extraction with the mistake (and maybe others) in it
    path: str | None  # the item the judge must flag; None for drop_job
    dropped: ExtractedJob | None = None  # the removed job, for drop_job


def plant_error(
    profile: CandidateProfile, kind: PlantKind, cv_text: str, baseline: JudgeResult
) -> PlantedError | None:
    """One copy of ``profile`` with one mistake, or None when the CV has no fair target."""
    if not _targets(profile, kind, normalize(cv_text), baseline):
        return None
    plants, _ = _plant_copy(profile, [kind], normalize(cv_text), baseline)
    return plants[0]


def plant_copies(
    profile: CandidateProfile, cv_text: str, baseline: JudgeResult
) -> tuple[list[list[PlantedError]], list[PlantKind]]:
    """Every fair plant, packed into as few copies as possible.

    Returns the copies (each a list of plants sharing one profile, so one judge
    call) and the kinds this CV offers no fair target for. A plant that would
    land on a job already carrying a mistake moves to the next copy.
    """
    text_key = normalize(cv_text)
    pending = [k for k in PLANT_KINDS if _targets(profile, k, text_key, baseline)]
    unfair = [k for k in PLANT_KINDS if k not in pending]
    copies = []
    while pending:
        plants, pending = _plant_copy(profile, pending, text_key, baseline)
        copies.append(plants)
    return copies, unfair


def _targets(
    profile: CandidateProfile, kind: PlantKind, text_key: str, baseline: JudgeResult
) -> list[int | None]:
    """Fair targets in order of preference: job indices, or [None] for a new skill."""
    if kind == "fake_skill":
        return [None] if _absent(FAKE_SKILLS, text_key) else []
    supported = {v.path for v in baseline.verdicts if v.verdict == "supported"}
    jobs = [i for i in range(len(profile.experience)) if f"experience[{i}]" in supported]
    if kind == "drop_job":
        return jobs
    if kind == "shift_start":
        return [i for i in jobs if profile.experience[i].start]
    if kind == "fake_company":
        if _absent(FAKE_COMPANIES, text_key) is None:
            return []
        return [i for i in jobs if profile.experience[i].company]
    raise ValueError(f"unknown plant kind {kind!r}")


def _plant_copy(
    profile: CandidateProfile, kinds: list[PlantKind], text_key: str, baseline: JudgeResult
) -> tuple[list[PlantedError], list[PlantKind]]:
    """Plant as many of ``kinds`` as fit in one copy (one mistake per job); returns the rest."""
    chosen: list[tuple[PlantKind, int | None]] = []
    rest: list[PlantKind] = []
    taken: set[int] = set()
    for kind in kinds:
        index = next(
            (i for i in _targets(profile, kind, text_key, baseline) if i not in taken), -1
        )
        if index == -1:
            rest.append(kind)
            continue
        if index is not None:
            taken.add(index)
        chosen.append((kind, index))

    copy = profile.model_copy(deep=True)
    paths: dict[PlantKind, str | None] = {}
    drop_at: int | None = None
    for kind, index in chosen:
        if kind == "fake_skill":
            name = _absent(FAKE_SKILLS, text_key)
            copy.skills.append(
                Skill(name=name, kind="hard", found_in=["skills_list"], quotes=[name])
            )
            paths[kind] = f"skills[{len(copy.skills) - 1}]"
        elif kind == "drop_job":
            drop_at = index
        elif kind == "shift_start":
            job = copy.experience[index]
            year, sep, month = job.start.partition("-")
            start = f"{int(year) - START_SHIFT_YEARS}{sep}{month}"
            copy.experience[index] = job.model_copy(update={"start": start})
        else:  # fake_company
            company = _absent(FAKE_COMPANIES, text_key)
            copy.experience[index] = copy.experience[index].model_copy(
                update={"company": company}
            )

    # The drop goes last, and later jobs move up one place.
    dropped = None
    if drop_at is not None:
        dropped = copy.experience.pop(drop_at)
    for kind, index in chosen:
        if kind in ("shift_start", "fake_company"):
            shift = 1 if drop_at is not None and index > drop_at else 0
            paths[kind] = f"experience[{index - shift}]"

    plants = [
        PlantedError(kind, copy, paths.get(kind),
                     dropped if kind == "drop_job" else None)
        for kind, _ in chosen
    ]
    return plants, rest


def _absent(names: tuple[str, ...], text_key: str) -> str | None:
    return next((n for n in names if normalize(n) not in text_key), None)


def caught(plant: PlantedError, judged: JudgeResult, baseline: JudgeResult) -> bool:
    """Did the judge flag the planted mistake?

    A dropped job is caught when the judge lists a missed job whose quote names
    that job (its title, company or own quote) and the baseline did not; any
    other plant when its item is not called "supported". Matching the quote
    keeps a missed job reported because of another plant in the same copy from
    counting as a catch.
    """
    if plant.kind == "drop_job":
        names = _dropped_job_names(plant)

        def names_dropped(result: JudgeResult) -> bool:
            return any(
                m.field == "experience" and any(
                    fuzz.partial_ratio(name, normalize(m.quote)) >= MATCH_THRESHOLD
                    for name in names
                )
                for m in result.missed
            )

        return bool(names) and names_dropped(judged) and not names_dropped(baseline)
    verdict = next((v for v in judged.verdicts if v.path == plant.path), None)
    return verdict is not None and verdict.verdict != "supported"


def _dropped_job_names(plant: PlantedError) -> list[str]:
    """What a quote must name to point at the dropped job: its company, and its
    title unless a job left in the copy has the same title."""
    job = plant.dropped
    if job is None:
        return []
    others = {normalize(j.title or "") for j in plant.profile.experience}
    names = [normalize(job.company or "")]
    if normalize(job.title or "") not in others:
        names.append(normalize(job.title or ""))
    return [n for n in names if n]


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
