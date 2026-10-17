"""Synthetic candidates, vacancies and graded pairs for ranking evaluation (JM-25).

JM-26 measures the ranking against this set (Spearman per candidate, NDCG@10) and
re-measures the semantic calibration on it. The one rule that makes that measurement
honest: a pair's grade comes from how the pair was built, never from the scoring
formula. Two construction facts decide it, and the scorer sees neither:

- how the candidate's job history relates to the vacancy's role family (same,
  adjacent or unrelated; see ``ADJACENT`` in synthetic_catalog.py), and
- the share of the vacancy's required skills the generator gave the candidate,
  counted on ground-truth skill identities, not on what normalization later finds.

    relation     >= 80%   50-79%   25-49%   < 25%
    same           3        2        1        0
    adjacent       2        1        0        0
    unrelated      1        0        0        0

Preferred and soft skills never change a grade. Role relation outweighs skills here,
which the 0.7 skill / 0.3 semantic formula does not assume, so a formula that leans too
hard on skills shows up as a lower correlation instead of a perfect one.

Every candidate is paired with every vacancy. Each regular candidate has at least
``MIN_PAIRS_PER_GRADE`` pairs at every grade, so per-candidate metrics are defined; a
draw that misses this is redrawn from the same random stream, so output stays
deterministic. Every fifth candidate is a hard case (``TRAPS``):

- keyword_stuffer: lists every required skill of a vacancy in an unrelated family,
  with jobs in another family. That vacancy is grade 1.
- alias_only: every skill written as an alias where one exists ("postgres", "k8s").
  Graded by the table, since the skills are really there.
- career_changer: two older jobs in one family, the two recent ones in another.
  Graded against the recent family.
- title_only: latest job title copied from a vacancy in the same family, with under
  25% of its required skills. That vacancy is grade 0.

Vacancies carry canonical skill names and CVs often carry aliases, so a run through
JM-20 normalization also tests matching. ``truth_vacancy`` and
``truth_candidate_skills`` give the perfect-normalization variant, which separates
matching errors from ranking errors.

Everything is generated from fixed templates with a seeded ``random.Random``: no LLM,
no network, and the same arguments always write byte-identical files.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from jobpdf.extraction.schema import (
    CandidateProfile,
    ExtractedEducation,
    ExtractedJob,
    Skill,
    SpokenLanguage,
)
from jobpdf.normalization.text import normalize
from jobpdf.ranking.models import RANKING_VERSION, CandidateSkills, SkillRef, VacancyFeatures
from jobpdf.ranking.semantic import build_candidate_texts, build_vacancy_texts
from jobpdf.ranking.synthetic_catalog import (
    ADJACENT,
    COMPANIES,
    DEGREES,
    FAMILIES,
    LOCATIONS,
    SENIORITY,
    SKILLS,
    SOFT_SKILLS,
    UNIVERSITIES,
    UNKNOWN_SKILLS,
    Family,
)

# Bump when generation changes in any way that changes the output.
GENERATOR_VERSION = "1"
DEFAULT_SEED = 25
DEFAULT_VACANCIES_PER_FAMILY = 5
# Fewer leaves too few same-family vacancies for two pairs at each of grades 1 to 3.
MIN_VACANCIES_PER_FAMILY = 5
DEFAULT_CANDIDATES = 20
MIN_PAIRS_PER_GRADE = 2
MAX_ATTEMPTS = 2000
TRUTH_PREFIX = "synthetic:"
SOURCE = "synthetic"
BASE_DATE = datetime(2026, 10, 1, tzinfo=timezone.utc)
CURRENT_MONTH = (2026, 9)

# Rubric: share of required skills, then the grade by relation.
STRONG_SHARE = 0.8
PARTIAL_SHARE = 0.5
WEAK_SHARE = 0.25
GRADES = (0, 1, 2, 3)
GRADE_BAND = {3: "strong", 2: "partial", 1: "weak", 0: "weak"}
TRAPS = ("keyword_stuffer", "alias_only", "career_changer", "title_only")
TRAP_EVERY = 5  # every fifth candidate is a hard case
REGULAR_ALIAS_SHARE = 0.4  # chance a regular CV writes a skill as its alias
UKRAINIAN_ROUND = 1  # in the second round over the families, CVs use Ukrainian lines where offered

VACANCIES_FILE = "vacancies.jsonl"
CANDIDATES_FILE = "candidates.jsonl"
PAIRS_FILE = "pairs.jsonl"
TRUTH_FILE = "truth.json"
MANIFEST_FILE = "manifest.json"
DATA_FILES = (VACANCIES_FILE, CANDIDATES_FILE, PAIRS_FILE, TRUTH_FILE)

Relation = Literal["same", "adjacent", "unrelated"]


class DatasetError(ValueError):
    """The dataset on disk is incomplete or does not match its manifest."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SyntheticCandidate(_Frozen):
    id: str
    profile: CandidateProfile


class PairLabel(_Frozen):
    candidate_id: str
    vacancy_id: str
    grade: int = Field(ge=0, le=3)
    expected_band: Literal["strong", "partial", "weak"]
    relation: Relation
    required_coverage: float  # on ground truth, rounded to 4 decimals
    trap: str | None  # the candidate's hard-case type, if any
    trap_target: bool = False  # this pair is the one the hard case was built around


class VacancyTruth(_Frozen):
    family: str
    required: list[str]  # skill slugs, in the vacancy's order
    preferred: list[str]


class CandidateTruth(_Frozen):
    home_family: str  # the family of the recent job history; grades are relative to it
    trap: str | None
    lang: Literal["en", "uk"]
    skills: dict[str, str | None]  # skill name as written on the CV -> slug (None: soft/unknown)


class Truth(_Frozen):
    vacancies: dict[str, VacancyTruth]
    candidates: dict[str, CandidateTruth]


class Dataset(_Frozen):
    vacancies: list[VacancyFeatures]
    candidates: list[SyntheticCandidate]
    pairs: list[PairLabel]
    truth: Truth


# --- rubric ------------------------------------------------------------------------


def relation(a: str, b: str) -> Relation:
    if a == b:
        return "same"
    return "adjacent" if frozenset((a, b)) in ADJACENT else "unrelated"


def coverage_bucket(share: float) -> int:
    """3: >= 80%, 2: 50-79%, 1: 25-49%, 0: under 25%."""
    if share >= STRONG_SHARE:
        return 3
    if share >= PARTIAL_SHARE:
        return 2
    if share >= WEAK_SHARE:
        return 1
    return 0


def grade(rel: Relation, share: float) -> int:
    bucket = coverage_bucket(share)
    if rel == "same":
        return bucket
    if rel == "adjacent":
        return max(bucket - 1, 0)
    return 1 if bucket == 3 else 0


def required_share(have: set[str], required: Sequence[str]) -> float:
    return sum(1 for s in required if s in have) / len(required) if required else 0.0


# --- vacancies ---------------------------------------------------------------------


def _adjacent_families(key: str) -> list[str]:
    return [f for f in FAMILIES if f != key and relation(key, f) == "adjacent"]


def _unrelated_families(key: str) -> list[str]:
    return [f for f in FAMILIES if relation(key, f) == "unrelated"]


def _ref(name: str) -> SkillRef:
    return SkillRef(raw=name, normalized=normalize(name))


def _make_vacancy(seed: int, index: int, family: Family, j: int) -> tuple[VacancyFeatures,
                                                                          VacancyTruth]:
    rng = random.Random(f"{seed}:vacancy:{family.key}:{j}")
    core = list(family.core)
    required = rng.sample(core, rng.randint(5, min(8, len(core) - 2)))
    neighbours = sorted({s for f in _adjacent_families(family.key) for s in FAMILIES[f].core}
                        - set(core))
    rest = [s for s in core if s not in required] + neighbours
    preferred = rng.sample(rest, rng.randint(2, 4))
    level = rng.choice(SENIORITY)
    title = " ".join(p for p in (level, rng.choice(family.titles)) if p)
    lines = list(family.responsibilities)
    vacancy = VacancyFeatures(
        id=f"syn-v-{index:03d}",
        source=SOURCE,
        title=title,
        company=rng.choice(COMPANIES),
        posted_at=BASE_DATE - timedelta(days=index),
        location_raw=rng.choice(LOCATIONS),
        required_skills=[_ref(SKILLS[s].name) for s in required],
        preferred_skills=[_ref(SKILLS[s].name) for s in preferred],
        responsibilities=rng.sample(lines, rng.randint(3, min(5, len(lines)))),
        min_years_experience={"Junior": 1, "": 3, "Senior": 5}[level],
    )
    return vacancy, VacancyTruth(family=family.key, required=required, preferred=preferred)


def make_vacancies(seed: int, per_family: int) -> tuple[list[VacancyFeatures],
                                                        dict[str, VacancyTruth]]:
    vacancies: list[VacancyFeatures] = []
    truth: dict[str, VacancyTruth] = {}
    for j in range(per_family):
        for family in FAMILIES.values():  # interleaved, so ids don't group by family
            vacancy, vt = _make_vacancy(seed, len(vacancies) + 1, family, j)
            vacancies.append(vacancy)
            truth[vacancy.id] = vt
    return vacancies, truth


# --- candidates --------------------------------------------------------------------


@dataclass
class _Draw:
    home: str
    skills: list[str]  # slugs, in CV order
    job_families: list[str]  # most recent first
    target: str | None = None  # vacancy id the hard case is built around
    latest_title: str | None = None


def _trap_for(i: int) -> str | None:
    return TRAPS[(i // TRAP_EVERY) % len(TRAPS)] if i % TRAP_EVERY == TRAP_EVERY - 1 else None


def _draw_skills(rng: random.Random, home: str) -> list[str]:
    core = list(FAMILIES[home].core)
    picked = rng.sample(core, rng.randint(2, len(core) - 1))
    extra = sorted({s for f in _adjacent_families(home) for s in FAMILIES[f].core} - set(core))
    return picked + rng.sample(extra, min(len(extra), rng.randint(0, 3)))


def _draw(rng: random.Random, home: str, trap: str | None,
          vacancies: Sequence[VacancyFeatures], vtruth: dict[str, VacancyTruth]) -> _Draw:
    jobs = rng.randint(2, 3)
    if trap == "career_changer":
        old = rng.choice(_unrelated_families(home))
        core = list(FAMILIES[home].core)
        skills = (rng.sample(core, rng.randint(4, len(core) - 1))
                  + rng.sample([s for s in FAMILIES[old].core if s not in core], 3))
        return _Draw(home, skills, [home, home, old, old])
    if trap == "keyword_stuffer":
        target = rng.choice([v for v in vacancies if vtruth[v.id].family
                             in _unrelated_families(home)])
        own = [s for s in FAMILIES[home].core if s not in vtruth[target.id].required]
        skills = list(vtruth[target.id].required) + rng.sample(own, min(len(own), 3))
        return _Draw(home, skills, [home] * jobs, target=target.id)
    if trap == "title_only":
        target = rng.choice([v for v in vacancies if vtruth[v.id].family == home])
        required = set(vtruth[target.id].required)
        others = [s for s in FAMILIES[home].core if s not in required]
        extra = sorted({s for f in _adjacent_families(home) for s in FAMILIES[f].core}
                       - set(FAMILIES[home].core))
        skills = rng.sample(others, min(len(others), 2)) + rng.sample(extra, 2)
        return _Draw(home, skills, [home] * jobs, target=target.id, latest_title=target.title)
    return _Draw(home, _draw_skills(rng, home), [home] * jobs)


def _grade_all(draw: _Draw, vacancies: Sequence[VacancyFeatures],
               vtruth: dict[str, VacancyTruth]) -> dict[str, tuple[int, Relation, float]]:
    have = set(draw.skills)
    out = {}
    for v in vacancies:
        vt = vtruth[v.id]
        rel = relation(draw.home, vt.family)
        share = required_share(have, vt.required)
        out[v.id] = (grade(rel, share), rel, share)
    return out


def _accept(draw: _Draw, trap: str | None, grades: dict[str, tuple[int, Relation, float]]) -> bool:
    counts = Counter(g for g, _, _ in grades.values())
    if trap in (None, "alias_only", "career_changer"):
        return all(counts[g] >= MIN_PAIRS_PER_GRADE for g in GRADES)
    if len(counts) < 2:
        return False
    expected = {"keyword_stuffer": 1, "title_only": 0}[trap]
    return draw.target is not None and grades[draw.target][0] == expected


def _written_name(rng: random.Random, slug: str, alias_only: bool) -> str:
    skill = SKILLS[slug]
    if skill.aliases and (alias_only or rng.random() < REGULAR_ALIAS_SHARE):
        return rng.choice(skill.aliases)
    return skill.name


def _months_between(start: tuple[int, int], end: tuple[int, int]) -> int:
    return (end[0] - start[0]) * 12 + end[1] - start[1]


def _shift(ym: tuple[int, int], months: int) -> tuple[int, int]:
    total = ym[0] * 12 + ym[1] - 1 - months
    return total // 12, total % 12 + 1


def _ym(ym: tuple[int, int]) -> str:
    return f"{ym[0]:04d}-{ym[1]:02d}"


def _render(rng: random.Random, draw: _Draw, trap: str | None,
            lang: Literal["en", "uk"]) -> tuple[CandidateProfile, dict[str, str | None]]:
    names: dict[str, str | None] = {}
    by_slug: dict[str, str] = {}
    for slug in draw.skills:
        name = _written_name(rng, slug, trap == "alias_only")
        if name in names:  # two slugs written the same way: keep the canonical name
            name = SKILLS[slug].name
        names[name] = slug
        by_slug[slug] = name
    for name in rng.sample(SOFT_SKILLS, 2):
        names[name] = None
    unknown = rng.choice(UNKNOWN_SKILLS) if rng.random() < 0.5 else None
    if unknown:
        names[unknown] = None

    experience: list[ExtractedJob] = []
    end: tuple[int, int] | None = None  # None = present
    cursor = CURRENT_MONTH
    total = 0
    for n, fam_key in enumerate(draw.job_families):
        family = FAMILIES[fam_key]
        months = rng.randint(14, 40)
        start = _shift(cursor, months)
        total += _months_between(start, cursor)
        if n == 0 and draw.latest_title:
            title = draw.latest_title
        else:
            title = rng.choice(family.titles)
        company = rng.choice(COMPANIES)
        own = [by_slug[s] for s in draw.skills if s in family.core] or [
            SKILLS[s].name for s in family.core[:2]]
        a, b = (rng.sample(own, 2) if len(own) >= 2 else [own[0], own[0]])
        lines = family.cv_lines_uk if lang == "uk" and family.cv_lines_uk else family.cv_lines
        description = rng.choice(lines).format(a=a, b=b)
        experience.append(ExtractedJob(
            title=title, company=company, location=None, start=_ym(start),
            end="present" if end is None else _ym(end), description=description,
            skills_used=[s for s in dict.fromkeys((a, b)) if s in description],
            quote=f"{title}, {company}",
        ))
        end = _shift(start, 1)
        cursor = end
    grad_end = _shift(cursor, rng.randint(1, 6))
    degree = rng.choice(DEGREES)
    university = rng.choice(UNIVERSITIES)
    education = [ExtractedEducation(
        institution=university, degree=degree, field="Computer Science",
        start=str(grad_end[0] - (4 if degree == "BSc" else 2)), end=str(grad_end[0]),
        quote=f"{degree}, {university}",
    )]
    languages = [SpokenLanguage(language="English", level=rng.choice(("B2", "C1")))]
    if lang == "uk":
        languages.append(SpokenLanguage(language="Ukrainian", level="native"))
    skills = [Skill(name=name, kind="hard" if slug or name in UNKNOWN_SKILLS else "soft",
                    found_in=["skills_list"], quotes=[name]) for name, slug in names.items()]
    profile = CandidateProfile(
        summary=f"{experience[0].title} with about {round(total / 12)} years of experience.",
        skills=skills, experience=experience, education=education, languages=languages,
        certifications=[], total_experience_months=total,
    )
    return profile, names


def make_candidates(seed: int, count: int, vacancies: Sequence[VacancyFeatures],
                    vtruth: dict[str, VacancyTruth]) -> tuple[
                        list[SyntheticCandidate], dict[str, CandidateTruth], list[PairLabel]]:
    families = list(FAMILIES)
    candidates: list[SyntheticCandidate] = []
    ctruth: dict[str, CandidateTruth] = {}
    pairs: list[PairLabel] = []
    for i in range(count):
        cid = f"syn-c-{i + 1:03d}"
        trap = _trap_for(i)
        home = families[i % len(families)]
        rng = random.Random(f"{seed}:candidate:{i}")
        for _ in range(MAX_ATTEMPTS):
            draw = _draw(rng, home, trap, vacancies, vtruth)
            grades = _grade_all(draw, vacancies, vtruth)
            if _accept(draw, trap, grades):
                break
        else:
            raise RuntimeError(f"{cid}: no draw met the grade rules in {MAX_ATTEMPTS} attempts")
        lang: Literal["en", "uk"] = (
            "uk" if FAMILIES[home].cv_lines_uk and i // len(families) == UKRAINIAN_ROUND else "en")
        profile, names = _render(rng, draw, trap, lang)
        candidates.append(SyntheticCandidate(id=cid, profile=profile))
        ctruth[cid] = CandidateTruth(home_family=home, trap=trap, lang=lang, skills=names)
        for v in vacancies:
            g, rel, share = grades[v.id]
            pairs.append(PairLabel(
                candidate_id=cid, vacancy_id=v.id, grade=g, expected_band=GRADE_BAND[g],
                relation=rel, required_coverage=round(share, 4), trap=trap,
                trap_target=v.id == draw.target,
            ))
    return candidates, ctruth, pairs


def generate(seed: int = DEFAULT_SEED, vacancies_per_family: int = DEFAULT_VACANCIES_PER_FAMILY,
             candidates: int = DEFAULT_CANDIDATES) -> Dataset:
    if vacancies_per_family < MIN_VACANCIES_PER_FAMILY or candidates < 1:
        raise ValueError(f"need at least {MIN_VACANCIES_PER_FAMILY} vacancies per family "
                         "and 1 candidate")
    vacancies, vtruth = make_vacancies(seed, vacancies_per_family)
    cands, ctruth, pairs = make_candidates(seed, candidates, vacancies, vtruth)
    return Dataset(vacancies=vacancies, candidates=cands, pairs=pairs,
                   truth=Truth(vacancies=vtruth, candidates=ctruth))


# --- files -------------------------------------------------------------------------


def _line(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)


def _jsonl(models: Sequence[BaseModel]) -> bytes:
    return "".join(_line(m) + "\n" for m in models).encode("utf-8")


def sha256(raw: bytes) -> str:
    """Hash with LF line endings, so a Windows checkout (autocrlf) hashes the same."""
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def serialize(dataset: Dataset, seed: int, vacancies_per_family: int) -> dict[str, bytes]:
    """File name -> bytes, manifest included. No timestamps, so output is reproducible."""
    files = {
        VACANCIES_FILE: _jsonl(dataset.vacancies),
        CANDIDATES_FILE: _jsonl(dataset.candidates),
        PAIRS_FILE: _jsonl(dataset.pairs),
        TRUTH_FILE: (json.dumps(dataset.truth.model_dump(mode="json"), sort_keys=True,
                                ensure_ascii=False, indent=1) + "\n").encode("utf-8"),
    }
    manifest = {
        "description": "Synthetic candidate-vacancy pairs for ranking evaluation (JM-25). "
                       "Grades come from construction, not from the scoring formula; see "
                       "jobpdf/ranking/synthetic.py. All names and companies are fictional.",
        "generator_version": GENERATOR_VERSION,
        "ranking_version": RANKING_VERSION,
        "seed": seed,
        "vacancies_per_family": vacancies_per_family,
        "counts": {"vacancies": len(dataset.vacancies), "candidates": len(dataset.candidates),
                   "pairs": len(dataset.pairs)},
        "grade_counts": {str(g): n for g, n in sorted(
            Counter(p.grade for p in dataset.pairs).items())},
        "files": {name: sha256(raw) for name, raw in files.items()},
    }
    files[MANIFEST_FILE] = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    return files


def write_dataset(dataset: Dataset, out_dir: Path, seed: int,
                  vacancies_per_family: int) -> dict[str, bytes]:
    files = serialize(dataset, seed, vacancies_per_family)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, raw in files.items():
        (out_dir / name).write_bytes(raw)
    return files


def load_dataset(directory: Path) -> Dataset:
    """Read a written dataset; raises DatasetError if a file is missing or changed."""
    try:
        manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
        raw = {name: (directory / name).read_bytes() for name in DATA_FILES}
    except FileNotFoundError as exc:
        raise DatasetError(f"missing dataset file: {Path(exc.filename).name}") from None
    changed = [n for n in DATA_FILES if manifest.get("files", {}).get(n) != sha256(raw[n])]
    if changed:
        raise DatasetError(f"files do not match the manifest: {', '.join(changed)}")

    def rows(name: str) -> list[dict]:
        return [json.loads(line) for line in raw[name].decode("utf-8").splitlines() if line]

    return Dataset(
        vacancies=[VacancyFeatures.model_validate(r) for r in rows(VACANCIES_FILE)],
        candidates=[SyntheticCandidate.model_validate(r) for r in rows(CANDIDATES_FILE)],
        pairs=[PairLabel.model_validate(r) for r in rows(PAIRS_FILE)],
        truth=Truth.model_validate_json(raw[TRUTH_FILE]),
    )


# --- variants for evaluation -------------------------------------------------------


def truth_key(slug: str) -> str:
    return TRUTH_PREFIX + slug


def _truth_ref(slug: str, written: str) -> SkillRef:
    return SkillRef(concept_id=truth_key(slug), raw=written, normalized=normalize(written),
                    label=SKILLS[slug].name)


def truth_vacancy(dataset: Dataset, vacancy: VacancyFeatures) -> VacancyFeatures:
    """The vacancy with perfectly normalized skills (ground-truth keys)."""
    vt = dataset.truth.vacancies[vacancy.id]
    return vacancy.model_copy(update={
        "required_skills": [_truth_ref(s, SKILLS[s].name) for s in vt.required],
        "preferred_skills": [_truth_ref(s, SKILLS[s].name) for s in vt.preferred],
    })


def truth_candidate_skills(dataset: Dataset, candidate_id: str) -> CandidateSkills:
    """The candidate's skills with perfect normalization; soft and unknown skills keep
    a raw key, as unrecognised skills do in production."""
    refs: dict[str, SkillRef] = {}
    sources: dict[str, list[str]] = {}
    for j, (written, slug) in enumerate(dataset.truth.candidates[candidate_id].skills.items()):
        ref = _truth_ref(slug, written) if slug else SkillRef(raw=written,
                                                              normalized=normalize(written))
        refs.setdefault(ref.key, ref)
        sources.setdefault(ref.key, []).append(f"skills[{j}]")
    return CandidateSkills(refs=refs, sources=sources)


CALIBRATION_PER_CANDIDATE = 2  # related and unrelated pairs each, per candidate


def calibration_pairs(dataset: Dataset) -> dict:
    """Pairs in scripts/calibrate_semantic.py's format: grade 3 related, grade 0 not.

    Up to ``CALIBRATION_PER_CANDIDATE`` of each per candidate, first by vacancy id. The
    CV text is the most recent job, built exactly as JM-23 builds candidate texts.
    """
    vacancies = {v.id: v for v in dataset.vacancies}
    candidates = {c.id: c for c in dataset.candidates}
    out = []
    taken: Counter[tuple[str, bool]] = Counter()
    for pair in sorted(dataset.pairs, key=lambda p: (p.candidate_id, p.vacancy_id)):
        if pair.grade not in (0, 3):
            continue
        related = pair.grade == 3
        if taken[(pair.candidate_id, related)] >= CALIBRATION_PER_CANDIDATE:
            continue
        texts, _ = build_candidate_texts(candidates[pair.candidate_id].profile)
        if not texts:
            continue
        taken[(pair.candidate_id, related)] += 1
        lang = dataset.truth.candidates[pair.candidate_id].lang
        out.append({
            "id": f"{pair.candidate_id}--{pair.vacancy_id}",
            "lang": f"{lang}-en",
            "related": related,
            "cv_text": texts[0].text,
            "vacancy_text": build_vacancy_texts(vacancies[pair.vacancy_id])[0],
        })
    return {
        "description": "Generated from the JM-25 synthetic ranking dataset: grade 3 pairs are "
                       "related, grade 0 pairs unrelated. All names and companies are "
                       "fictional.",
        "pairs": out,
    }
