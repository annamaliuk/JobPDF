import math
from datetime import datetime, timezone

import numpy as np
import pytest

from jobpdf.extraction.schema import ExtractedJob, ExtractedProfile, SkillMention
from jobpdf.ranking.models import RANKING_VERSION, CandidateSkills, SkillRef, VacancyFeatures
from jobpdf.ranking.overlap import SkillOverlap
from jobpdf.ranking.score import (
    BAND_ORDER,
    BAND_WITHOUT_SEMANTIC,
    DUPLICATE_VACANCY_ID,
    PARTIAL_COVERAGE,
    PARTIAL_FIT,
    SEMANTIC_MISSING,
    SEMANTIC_UNAVAILABLE,
    SKILL_MISSING,
    STRONG_COVERAGE,
    STRONG_FIT,
    UNSCORED,
    W_EXPERIENCE,
    W_SEMANTIC,
    W_SKILL,
    Band,
    FitComponents,
    FitResult,
    LinearScorer,
    RankingResult,
    VacancyAdjustment,
    _check_weights,
    assign_band,
    experience_fit,
    fit_warnings,
    rank_vacancies,
    skill_coverage,
)
from jobpdf.ranking.semantic import (
    VACANCY_TEXT_VERSION,
    Calibration,
    CandidateVectors,
    embed_candidate,
    embed_vacancy,
)

B = Band
PYTHON = "http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d"


def components(skill=None, semantic=None, experience=None) -> FitComponents:
    return FitComponents(skill=skill, required_coverage=None, semantic=semantic,
                         semantic_raw=None, experience=experience)


# --- fit score ---------------------------------------------------------------------


def test_default_weights() -> None:
    assert (W_SKILL, W_SEMANTIC, W_EXPERIENCE) == (0.40, 0.45, 0.15)
    assert LinearScorer().weights == {"skill": 0.40, "semantic": 0.45, "experience": 0.15}
    assert LinearScorer.name == "linear"


def test_hand_computed_fit() -> None:
    fit = LinearScorer().score(components(skill=0.5, semantic=0.8, experience=0.6))

    assert fit == pytest.approx(0.40 * 0.5 + 0.45 * 0.8 + 0.15 * 0.6) == pytest.approx(0.65)


@pytest.mark.parametrize(
    ("skill", "semantic", "experience", "fit", "warnings"),
    [
        (0.5, 0.8, 0.6, 0.65, []),
        (0.5, 0.8, None, (0.40 * 0.5 + 0.45 * 0.8) / 0.85, []),  # 47% / 53%
        (0.5, None, 0.6, (0.40 * 0.5 + 0.15 * 0.6) / 0.55, [SEMANTIC_MISSING]),  # 73% / 27%
        (None, 0.8, 0.6, (0.45 * 0.8 + 0.15 * 0.6) / 0.60, [SKILL_MISSING]),  # 75% / 25%
        (0.5, None, None, 0.5, [SEMANTIC_MISSING]),
        (None, 0.8, None, 0.8, [SKILL_MISSING]),
        (None, None, 0.6, None, [UNSCORED]),  # years alone are not a fit
        (None, None, None, None, [UNSCORED]),
    ],
    ids=["all", "no_years", "no_semantic", "no_skills", "skills_only", "semantic_only",
         "years_only", "nothing"],
)
def test_a_missing_part_shares_its_weight_out(skill, semantic, experience, fit,
                                              warnings) -> None:
    comps = components(skill, semantic, experience)

    result = LinearScorer().score(comps)

    assert result == (None if fit is None else pytest.approx(fit))
    assert fit_warnings(comps, result) == warnings


@pytest.mark.parametrize("weights", [(0.5, 0.5, 0.15), (1.2, -0.2, 0.0), (0.4, 0.4, 0.1)])
def test_weights_must_be_non_negative_and_add_up_to_one(weights) -> None:
    with pytest.raises(ValueError, match="add up to 1"):
        LinearScorer(*weights)
    with pytest.raises(ValueError):
        _check_weights(*weights)


@pytest.mark.parametrize(
    ("years", "months", "part"),
    [(3, 48, 1.0), (3, 36, 1.0), (3, 24, 0.6), (5, 36, 0.2), (5, 30, 0.0), (5, 0, 0.0),
     (None, 36, None), (0, 36, None), (3, None, None)],
)
def test_experience_fit(years, months, part) -> None:
    assert experience_fit(years, months) == (None if part is None else pytest.approx(part))


def test_float_rounding_in_weights_is_accepted() -> None:
    assert LinearScorer(0.1 + 0.2, 0.55, 0.15).weights["skill"] == pytest.approx(0.3)


# --- bands -------------------------------------------------------------------------


def test_thresholds() -> None:
    assert (STRONG_FIT, STRONG_COVERAGE, PARTIAL_FIT, PARTIAL_COVERAGE) == (0.70, 0.8, 0.65, 0.5)
    assert sorted(Band, key=BAND_ORDER.get) == [B.STRONG, B.PARTIAL, B.WEAK, B.UNSCORED]


@pytest.mark.parametrize(
    ("fit", "band"),
    [(STRONG_FIT, B.STRONG), (1.0, B.STRONG), (STRONG_FIT - 1e-9, B.PARTIAL),
     (PARTIAL_FIT, B.PARTIAL), (PARTIAL_FIT - 1e-9, B.WEAK), (0.0, B.WEAK),
     (None, B.UNSCORED)],
)
def test_fit_boundaries(fit, band: Band) -> None:
    assert assign_band(fit, 1.0) == (band, band)


@pytest.mark.parametrize(
    ("coverage", "band"),
    [(STRONG_COVERAGE, B.STRONG), (STRONG_COVERAGE - 1e-9, B.PARTIAL),
     (PARTIAL_COVERAGE, B.PARTIAL), (PARTIAL_COVERAGE - 1e-9, B.WEAK)],
)
def test_skill_coverage_boundaries(coverage, band: Band) -> None:
    assert assign_band(0.9, coverage) == (band, band)


@pytest.mark.parametrize(
    ("fit", "band"),
    [(0.95, B.PARTIAL), (PARTIAL_FIT, B.PARTIAL), (PARTIAL_FIT - 1e-9, B.WEAK),
     (None, B.UNSCORED)],
)
def test_vacancy_without_skills_is_at_most_partial(fit, band: Band) -> None:
    assert assign_band(fit, None)[0] is band


@pytest.mark.parametrize(
    ("skill", "semantic", "band"),
    [(1.0, 0.1, B.WEAK), (0.5, 0.9, B.PARTIAL), (0.2, 1.0, B.WEAK), (0.8, 0.9, B.STRONG)],
    ids=["all_skills_unrelated_work", "half_skills_right_background", "title_only",
         "most_skills_right_background"],
)
def test_strong_needs_both_the_fit_and_the_skills(skill, semantic, band: Band) -> None:
    fit = LinearScorer().score(components(skill, semantic, experience=1.0))

    assert assign_band(fit, skill)[0] is band


def test_skill_coverage_is_the_required_share_else_the_overlap_score() -> None:
    def overlap(score, coverage):
        return SkillOverlap(
            score=score, required_coverage=coverage, required_matched=[], required_missing=[],
            preferred_matched=[], preferred_missing=[], required_matched_count=0,
            required_count=0, preferred_matched_count=0, preferred_count=0, warnings=[])

    assert skill_coverage(overlap(0.83, 0.5)) == 0.5
    assert skill_coverage(overlap(0.6, None)) == 0.6  # only nice-to-have skills listed
    assert skill_coverage(overlap(None, None)) is None


# --- caps --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cap", "expected"),
    [(B.STRONG, B.STRONG), (B.PARTIAL, B.PARTIAL), (B.WEAK, B.WEAK),
     (B.UNSCORED, B.UNSCORED), (None, B.STRONG)],
)
def test_cap_lowers_a_strong_band(cap, expected: Band) -> None:
    assert assign_band(0.9, 1.0, cap) == (expected, B.STRONG)


def test_cap_never_raises_a_band() -> None:
    assert assign_band(0.1, 1.0, B.STRONG) == (B.WEAK, B.WEAK)
    assert assign_band(None, 1.0, B.PARTIAL) == (B.UNSCORED, B.UNSCORED)


def test_adjustment_defaults() -> None:
    assert VacancyAdjustment().model_dump() == {"hidden": False, "band_cap": None,
                                                "reasons": []}


# --- rank_vacancies ------------------------------------------------------------------

VERSION = "fake/e5@rev1|query: "
CAL = Calibration(floor=0.0, ceil=1.0, embedding_version=VERSION)  # score == cosine
A, C, D, E = (f"jobpdf:skill/{s}" for s in ("alpha", "charlie", "delta", "echo"))
VECTORS = CandidateVectors(vectors=[[1.0, 0.0]], paths=["experience[0]"], text_version="1",
                           embedding_version=VERSION, warnings=[])


def ref(key: str) -> SkillRef:
    return SkillRef(concept_id=key, raw=key[-5:], normalized=key[-5:])


CANDIDATE = CandidateSkills().with_added([ref(A), ref(C)])


def vac(id: str, required=(), preferred=(), sem: float | None = None,
        posted: datetime | None = None) -> VacancyFeatures:
    """A vacancy whose semantic score against VECTORS is exactly ``sem``."""
    embedded = sem is not None
    return VacancyFeatures(
        id=id, source="synthetic", title=f"Job {id}", posted_at=posted,
        required_skills=[ref(k) for k in required], preferred_skills=[ref(k) for k in preferred],
        embedding=[sem, math.sqrt(1 - sem * sem)] if embedded else None,
        embedding_version=VERSION if embedded else None,
        vacancy_text_version=VACANCY_TEXT_VERSION if embedded else None,
    )


def rank(vacancies, **kwargs) -> RankingResult:
    kwargs.setdefault("calibration", CAL)
    return rank_vacancies(CANDIDATE, kwargs.pop("vectors", VECTORS), vacancies, **kwargs)


def ids(results: list[FitResult]) -> list[str]:
    return [r.vacancy.id for r in results]


def test_bands_fits_and_order() -> None:
    result = rank([
        vac("weak", required=[A, D, E], sem=0.2),  # fit 0.26
        vac("strong", required=[A, C], sem=0.5),  # fit 0.74
        vac("partial", required=[A, D], sem=0.9),  # fit 0.71, half the skills
        vac("unscored"),  # no skills, no embedding
    ])

    assert ids(result.results) == ["strong", "partial", "weak", "unscored"]
    strong = result.results[0]
    assert (strong.band, strong.band_before_cap) == (Band.STRONG, Band.STRONG)
    assert strong.fit == pytest.approx((0.40 * 1.0 + 0.45 * 0.5) / 0.85)
    assert strong.components.skill == 1.0 and strong.components.required_coverage == 1.0
    assert strong.components.semantic == pytest.approx(0.5)
    assert strong.components.semantic_raw == pytest.approx(0.5)
    assert result.results[3].fit is None and result.results[3].warnings == [UNSCORED]
    assert result.band_counts == {Band.STRONG: 1, Band.PARTIAL: 1, Band.WEAK: 1,
                                  Band.UNSCORED: 1}
    assert result.semantic_calibrated is True
    assert result.scorer_name == "linear"
    assert result.weights == {"skill": 0.40, "semantic": 0.45, "experience": 0.15}
    assert result.thresholds == {"strong_fit": 0.70, "strong_coverage": 0.8,
                                 "partial_fit": 0.65, "partial_coverage": 0.5}
    assert result.ranking_version == RANKING_VERSION == strong.ranking_version
    assert result.warnings == [] and result.hidden == [] and result.hidden_count == 0


def test_hidden_vacancies_are_scored_counted_and_kept() -> None:
    adjustments = {"far": VacancyAdjustment(hidden=True, reasons=["location: on-site in Lima"])}

    result = rank([vac("far", required=[A, C], sem=0.9), vac("near", required=[A], sem=0.5)],
                  adjustments=adjustments)

    assert ids(result.results) == ["near"] and ids(result.hidden) == ["far"]
    far = result.hidden[0]
    assert far.band is Band.STRONG and far.fit is not None
    assert far.adjustment_reasons == ["location: on-site in Lima"]
    assert result.hidden_count == 1
    assert sum(result.band_counts.values()) == 1


def test_cap_from_an_adjustment_lowers_the_band_and_keeps_the_original() -> None:
    adjustments = {"senior": VacancyAdjustment(band_cap=Band.PARTIAL, reasons=["years: 8+"]),
                   "junior": VacancyAdjustment(band_cap=Band.STRONG)}

    result = rank([vac("senior", required=[A, C], sem=0.9), vac("junior", required=[D])],
                  adjustments=adjustments)

    senior = next(r for r in result.results if r.vacancy.id == "senior")
    junior = next(r for r in result.results if r.vacancy.id == "junior")
    assert (senior.band, senior.band_before_cap) == (Band.PARTIAL, Band.STRONG)
    assert senior.adjustment_reasons == ["years: 8+"]
    assert (junior.band, junior.band_before_cap) == (Band.WEAK, Band.WEAK)  # never raised


def test_ties_are_broken_by_fit_then_date_then_id() -> None:
    utc = timezone.utc
    vacancies = [
        vac("b-old", required=[A], sem=0.5, posted=datetime(2026, 9, 1, tzinfo=utc)),
        vac("z-undated", required=[A], sem=0.5),
        vac("a-new-naive", required=[A], sem=0.5, posted=datetime(2026, 10, 9)),
        vac("c-new-aware", required=[A], sem=0.5, posted=datetime(2026, 10, 9, tzinfo=utc)),
        vac("better-fit", required=[A], sem=0.6),
    ]

    result = rank(vacancies)

    assert {r.band for r in result.results} == {Band.STRONG}
    assert ids(result.results) == ["better-fit", "a-new-naive", "c-new-aware", "b-old",
                                   "z-undated"]
    assert rank(vacancies) == result  # identical every time
    assert ids(rank(list(reversed(vacancies))).results) == ids(result.results)


def test_none_fit_sorts_last_within_a_band() -> None:
    result = rank([vac("no-fit"), vac("weak-fit", required=[D], sem=0.1)])

    assert [(r.vacancy.id, r.band) for r in result.results] == [
        ("weak-fit", Band.WEAK), ("no-fit", Band.UNSCORED)]


def test_a_custom_scorer_changes_the_fits_and_so_the_order() -> None:
    vacancies = [
        vac("skills-heavy", required=[A, D], preferred=[C], sem=0.3),  # skill 0.6
        vac("semantic-heavy", required=[A, D], sem=0.45),  # skill 0.5
    ]

    default = rank(vacancies)
    skills_only = rank(vacancies, scorer=LinearScorer(0.85, 0.0, 0.15))

    assert ids(default.results) == ["semantic-heavy", "skills-heavy"]
    assert [r.band for r in default.results] == [Band.WEAK] * 2
    assert ids(skills_only.results) == ["skills-heavy", "semantic-heavy"]
    assert skills_only.weights == {"skill": 0.85, "semantic": 0.0, "experience": 0.15}


def test_experience_counts_in_the_fit_and_so_in_the_band() -> None:
    def asks(id: str, years: int | None) -> VacancyFeatures:
        return vac(id, required=[A, D], sem=0.9).model_copy(
            update={"min_years_experience": years})

    vacancies = [asks("needs-5", 5), asks("needs-2", 2), asks("states-none", None)]

    result = rank(vacancies, experience_months=36)
    without = rank(vacancies)

    assert ids(result.results) == ["needs-2", "states-none", "needs-5"]
    assert [r.components.experience for r in result.results] == [1.0, None,
                                                                  pytest.approx(0.2)]
    assert [r.band for r in result.results] == [Band.PARTIAL, Band.PARTIAL, Band.WEAK]
    assert {r.fit for r in without.results} == {result.results[1].fit}  # unchanged when unknown


def test_a_scorer_without_weights_is_reported_by_name() -> None:
    class Constant:
        name = "constant"

        def score(self, components: FitComponents) -> float | None:
            return 0.5

    result = rank([vac("x", required=[A], sem=0.9)], scorer=Constant())

    assert (result.scorer_name, result.weights, result.results[0].fit) == ("constant", {}, 0.5)


def test_assign_band_agrees_with_rank_vacancies() -> None:
    vacancies = [vac("a", required=[A, C], sem=0.2), vac("b", preferred=[A, D], sem=0.9),
                 vac("c", sem=0.9), vac("d", required=[A, C])]

    for result in rank(vacancies).results:
        assert assign_band(result.fit, skill_coverage(result.overlap)) == (
            result.band, result.band_before_cap)


def test_without_candidate_vectors_ranking_uses_skills_only() -> None:
    result = rank([vac("a", required=[A, C], sem=0.9), vac("b", required=[A, D], sem=0.9),
                   vac("c", sem=0.9)], vectors=None)

    assert result.warnings == [SEMANTIC_UNAVAILABLE] and result.semantic_calibrated is None
    assert ids(result.results) == ["a", "b", "c"]
    a, b, c = result.results
    assert a.semantic is None and a.components.semantic is None
    assert a.fit == 1.0 and a.band is Band.STRONG
    assert a.warnings == [SEMANTIC_MISSING, BAND_WITHOUT_SEMANTIC]
    assert (b.fit, b.band, b.warnings) == (0.5, Band.WEAK, [SEMANTIC_MISSING])
    assert (c.band, c.fit, c.warnings) == (Band.UNSCORED, None, [UNSCORED])


def test_duplicate_ids_are_kept_and_share_an_adjustment() -> None:
    result = rank([vac("dup", required=[A], sem=0.9), vac("dup", required=[D], sem=0.9),
                   vac("other", required=[A], sem=0.9)],
                  adjustments={"dup": VacancyAdjustment(hidden=True)})

    assert result.warnings == [DUPLICATE_VACANCY_ID]
    assert ids(result.hidden) == ["dup", "dup"] and ids(result.results) == ["other"]


def test_empty_vacancy_list() -> None:
    result = rank([])

    assert (result.results, result.hidden, result.hidden_count) == ([], [], 0)
    assert result.band_counts == dict.fromkeys(Band, 0)
    assert result.semantic_calibrated is None and result.warnings == []


def test_ranking_result_json_round_trip() -> None:
    result = rank([vac("a", required=[A, C], preferred=[E], sem=0.9,
                       posted=datetime(2026, 10, 1, tzinfo=timezone.utc)),
                   vac("b", required=[D]), vac("c")],
                  adjustments={"b": VacancyAdjustment(hidden=True, reasons=["x"])})

    payload = result.model_dump_json()

    assert '"band":"strong"' in payload
    assert RankingResult.model_validate_json(payload) == result


# --- end to end with the real overlap and semantic code --------------------------------


class FakeEmbedder:
    """Known texts -> unit vectors, with the identity attributes JM-23 reads."""

    model_name = "fake/e5"
    revision = "rev1"
    prefix = "query: "

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.vectors[t] for t in texts], dtype=np.float32)


def test_end_to_end_with_real_overlap_semantic_and_a_fake_embedder() -> None:
    job = "Data Engineer\nAcme Widgets\nBuilt ETL pipelines"
    data_vacancy = "Data Engineer\nBuild ETL pipelines"
    chef_vacancy = "Pastry Chef\nBake cakes"
    embedder = FakeEmbedder({job: [1.0, 0.0], data_vacancy: [0.9, math.sqrt(0.19)],
                             chef_vacancy: [0.1, math.sqrt(0.99)]})
    profile = ExtractedProfile(
        summary=None,
        skills=[SkillMention(name="Python", kind="hard", found_in="skills_list",
                             quote="Python")],
        experience=[ExtractedJob(title="Data Engineer", company="Acme Widgets", location=None,
                                 start="2021", end="present",
                                 description="Built ETL pipelines", skills_used=[],
                                 quote="Data Engineer, Acme Widgets")],
        education=[], languages=[], certifications=[],
    )
    python = SkillRef(concept_id=PYTHON, raw="Python", normalized="python")
    candidate = CandidateSkills().with_added([python])
    vacancies = [
        embed_vacancy(VacancyFeatures(id="chef", source="synthetic", title="Pastry Chef",
                                      responsibilities=["Bake cakes"],
                                      required_skills=[ref(D)]), embedder),
        embed_vacancy(VacancyFeatures(id="data", source="synthetic", title="Data Engineer",
                                      responsibilities=["Build ETL pipelines"],
                                      required_skills=[python]), embedder),
    ]
    calibration = Calibration(floor=0.2, ceil=0.8, embedding_version=VERSION)

    result = rank_vacancies(candidate, embed_candidate(profile, embedder), vacancies,
                            calibration=calibration)

    assert ids(result.results) == ["data", "chef"]
    data, chef = result.results
    assert data.band is Band.STRONG and chef.band is Band.WEAK
    assert data.semantic.best_item_index == 0 and data.semantic.calibrated
    assert data.components.semantic == pytest.approx(1.0)  # (0.9 - 0.2) / 0.6, clipped
    assert data.fit == pytest.approx(1.0)
    assert chef.components.semantic == 0.0 and chef.fit == pytest.approx(0.0)
