import pytest

from jobpdf.ranking.overlap import SkillOverlap
from jobpdf.ranking.score import (
    BAND_ORDER,
    PARTIAL_COVERAGE,
    SEM_MIN,
    SEMANTIC_MISSING,
    SKILL_MISSING,
    STRONG_COVERAGE,
    UNSCORED,
    W_SEMANTIC,
    W_SKILL,
    Band,
    FitComponents,
    LinearScorer,
    VacancyAdjustment,
    _check_weights,
    assign_band,
    fit_warnings,
)

B = Band


def overlap(score: float | None = None, coverage: float | None = None) -> SkillOverlap:
    """A SkillOverlap with only the numbers banding reads."""
    return SkillOverlap(
        score=score, required_coverage=coverage, required_matched=[], required_missing=[],
        preferred_matched=[], preferred_missing=[], required_matched_count=0,
        required_count=0, preferred_matched_count=0, preferred_count=0, warnings=[],
    )


def components(skill=None, semantic=None, coverage=None, raw=None) -> FitComponents:
    return FitComponents(skill=skill, required_coverage=coverage, semantic=semantic,
                         semantic_raw=raw)


# --- fit score ---------------------------------------------------------------------


def test_default_weights() -> None:
    assert (W_SKILL, W_SEMANTIC) == (0.7, 0.3)
    assert LinearScorer().weights == {"skill": 0.7, "semantic": 0.3}
    assert LinearScorer.name == "linear"


def test_hand_computed_fit() -> None:
    fit = LinearScorer().score(components(skill=0.625, semantic=0.4))

    assert fit == pytest.approx(0.7 * 0.625 + 0.3 * 0.4) == pytest.approx(0.5575)


@pytest.mark.parametrize(
    ("skill", "semantic", "fit", "warnings"),
    [
        (0.625, 0.4, 0.5575, []),
        (0.625, None, 0.625, [SEMANTIC_MISSING]),
        (None, 0.4, 0.4, [SKILL_MISSING]),
        (None, None, None, [UNSCORED]),
    ],
    ids=["both", "semantic_missing", "skill_missing", "neither"],
)
def test_fit_table(skill, semantic, fit, warnings) -> None:
    comps = components(skill=skill, semantic=semantic)

    result = LinearScorer().score(comps)

    assert result == (None if fit is None else pytest.approx(fit))
    assert fit_warnings(comps, result) == warnings


@pytest.mark.parametrize(("w_skill", "w_semantic"), [(0.6, 0.6), (1.2, -0.2), (0.5, 0.4)])
def test_weights_must_be_non_negative_and_add_up_to_one(w_skill, w_semantic) -> None:
    with pytest.raises(ValueError, match="add up to 1"):
        LinearScorer(w_skill, w_semantic)
    with pytest.raises(ValueError):
        _check_weights(w_skill, w_semantic)


def test_float_rounding_in_weights_is_accepted() -> None:
    assert LinearScorer(0.1 + 0.2, 0.7).weights["skill"] == pytest.approx(0.3)


# --- bands -------------------------------------------------------------------------


def test_thresholds() -> None:
    assert (STRONG_COVERAGE, PARTIAL_COVERAGE, SEM_MIN) == (0.8, 0.5, 0.3)
    assert sorted(Band, key=BAND_ORDER.get) == [B.STRONG, B.PARTIAL, B.WEAK, B.UNSCORED]


@pytest.mark.parametrize(
    ("coverage", "semantic", "band"),
    [
        (STRONG_COVERAGE, SEM_MIN, B.STRONG),  # both exactly at their thresholds
        (1.0, 0.9, B.STRONG),
        (STRONG_COVERAGE, SEM_MIN - 1e-9, B.PARTIAL),  # strong coverage, low semantic
        (STRONG_COVERAGE - 1e-9, 0.9, B.PARTIAL),
        (PARTIAL_COVERAGE, 0.0, B.PARTIAL),  # exactly at the partial threshold
        (PARTIAL_COVERAGE - 1e-9, 0.9, B.WEAK),
        (0.0, 1.0, B.WEAK),
    ],
)
def test_band_boundaries(coverage: float, semantic: float, band: Band) -> None:
    assert assign_band(overlap(score=coverage, coverage=coverage), semantic) == (band, band)


def test_missing_semantic_never_blocks_strong() -> None:
    assert assign_band(overlap(score=0.9, coverage=0.9), None) == (B.STRONG, B.STRONG)
    assert assign_band(overlap(score=0.6, coverage=0.6), None) == (B.PARTIAL, B.PARTIAL)


def test_nice_to_have_only_uses_the_overlap_score_as_basis() -> None:
    assert assign_band(overlap(score=0.8, coverage=None), 0.5)[0] is B.STRONG
    assert assign_band(overlap(score=0.5, coverage=None), 0.5)[0] is B.PARTIAL
    assert assign_band(overlap(score=0.2, coverage=None), 0.5)[0] is B.WEAK


def test_required_coverage_wins_over_the_overlap_score() -> None:
    # all preferred matched pushes the score up, but only 1 of 2 required is matched
    assert assign_band(overlap(score=0.83, coverage=0.5), 0.9)[0] is B.PARTIAL


@pytest.mark.parametrize(
    ("semantic", "band"),
    [(1.0, B.PARTIAL), (SEM_MIN, B.PARTIAL), (SEM_MIN - 1e-9, B.WEAK), (None, B.UNSCORED)],
)
def test_vacancy_without_skills_is_at_most_partial(semantic, band: Band) -> None:
    assert assign_band(overlap(score=None, coverage=None), semantic)[0] is band


# --- caps --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cap", "expected"),
    [(B.STRONG, B.STRONG), (B.PARTIAL, B.PARTIAL), (B.WEAK, B.WEAK),
     (B.UNSCORED, B.UNSCORED), (None, B.STRONG)],
)
def test_cap_lowers_a_strong_band(cap, expected: Band) -> None:
    assert assign_band(overlap(score=1.0, coverage=1.0), 0.9, cap) == (expected, B.STRONG)


def test_cap_never_raises_a_band() -> None:
    assert assign_band(overlap(score=0.1, coverage=0.1), 0.9, B.STRONG) == (B.WEAK, B.WEAK)
    assert assign_band(overlap(), None, B.PARTIAL) == (B.UNSCORED, B.UNSCORED)


def test_adjustment_defaults() -> None:
    assert VacancyAdjustment().model_dump() == {"hidden": False, "band_cap": None,
                                                "reasons": []}
