import math

import pytest

from jobpdf.ranking.learned import FeedbackRow, LearnedLinearScorer, NotEnoughData
from jobpdf.ranking.score import Band, LinearScorer
from jobpdf.ranking.scorer_eval import (
    INCONCLUSIVE,
    LEARNED_BETTER,
    LINEAR_BETTER,
    MIN_SESSIONS_FOR_VERDICT,
    compare_scorers,
    evaluate_session,
    make_folds,
    ndcg_at_k,
    rank_session,
    spearman,
    verdict,
)
from tests.ranking.feedback_data import make_feedback


def row(vacancy: str, skill, semantic, label=0, band=None, session="s") -> FeedbackRow:
    return FeedbackRow(session=session, vacancy_id=vacancy, skill=skill, semantic=semantic,
                       label=label, band=band)


# --- metrics -----------------------------------------------------------------------


def test_ndcg_hand_computed() -> None:
    # useful cards at ranks 1 and 3: DCG = 1 + 1/log2(4); the ideal puts them at 1 and 2
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))

    assert ndcg_at_k([1, 0, 1]) == pytest.approx(expected)
    assert ndcg_at_k([1, 1, 0]) == pytest.approx(1.0)


def test_ndcg_is_none_without_a_useful_card_and_respects_the_cutoff() -> None:
    assert ndcg_at_k([0, 0, 0]) is None
    assert ndcg_at_k([0, 0, 1], k=2) == pytest.approx(0.0)  # the useful card is past the cut-off


def test_spearman_known_values() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # ties share their mean rank: ranks [1, 2.5, 2.5, 4] against [1, 2, 3, 4]
    assert spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(4.5 / math.sqrt(4.5 * 5.0))


def test_spearman_is_none_when_undefined() -> None:
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert spearman([1], [1]) is None


# --- one session -------------------------------------------------------------------


def test_cards_are_ordered_band_first_then_fit() -> None:
    rows = [
        row("partial_high", 0.9, 0.9, band=Band.PARTIAL),
        row("strong_low", 0.1, 0.1, band=Band.STRONG),
        row("partial_low", 0.2, 0.2, band=Band.PARTIAL),
    ]

    order = [r.vacancy_id for r in rank_session(rows, LinearScorer())]

    assert order == ["strong_low", "partial_high", "partial_low"]


def test_without_bands_the_score_decides_and_missing_scores_go_last() -> None:
    rows = [row("none", None, None), row("low", 0.1, 0.1), row("high", 0.9, 0.9),
            row("tie_b", 0.5, 0.5), row("tie_a", 0.5, 0.5)]

    order = [r.vacancy_id for r in rank_session(rows, LinearScorer())]

    assert order == ["high", "tie_a", "tie_b", "low", "none"]


def test_weights_cannot_change_the_order_across_bands() -> None:
    rows = [row("a", 0.9, 0.1, label=1, band=Band.STRONG),
            row("b", 0.1, 0.9, label=0, band=Band.PARTIAL)]

    linear = evaluate_session(rows, LinearScorer())
    learned = evaluate_session(rows, LearnedLinearScorer(0.0, 1.0))

    assert linear == learned


def test_a_session_without_a_useful_card_cannot_be_scored() -> None:
    assert evaluate_session([row("a", 0.5, 0.5), row("b", 0.2, 0.2)], LinearScorer()) is None


# --- folds -------------------------------------------------------------------------


def test_every_session_is_tested_exactly_once_and_never_trained_on_in_its_own_fold() -> None:
    data = make_feedback(0.5, sessions=23, per_session=4, seed=1)

    folds = make_folds(data, folds=5, seed=0)

    tested = [s for _, test in folds for s in {r.session for r in test}]
    assert sorted(tested) == sorted({r.session for r in data})  # once each
    for train, test in folds:
        assert not {r.session for r in train} & {r.session for r in test}


def test_folds_are_deterministic_and_capped_by_the_number_of_sessions() -> None:
    data = make_feedback(0.5, sessions=3, per_session=4, seed=1)

    assert make_folds(data, folds=10, seed=7) == make_folds(data, folds=10, seed=7)
    assert len(make_folds(data, folds=10, seed=7)) == 3


# --- comparison --------------------------------------------------------------------


def test_the_learned_scorer_wins_when_the_prior_is_wrong() -> None:
    data = make_feedback(0.2, sessions=100, per_session=12, seed=1)

    result = compare_scorers(data)

    assert result.learned_ndcg > result.linear_ndcg
    assert result.ndcg_diff_ci[0] > 0  # the interval excludes zero
    assert result.wins > result.losses
    assert result.verdict == LEARNED_BETTER
    assert result.n_sessions == 100 and result.n_skipped_folds == 0
    assert all(w["skill"] < 0.3 for w in result.fold_weights)


def test_no_difference_is_reported_as_inconclusive_when_the_prior_is_right() -> None:
    data = make_feedback(0.7, sessions=100, per_session=12, seed=1)

    result = compare_scorers(data)

    assert abs(result.ndcg_diff) < 0.01
    assert result.verdict == INCONCLUSIVE


def test_the_comparison_is_reproducible() -> None:
    data = make_feedback(0.2, sessions=40, per_session=10, seed=2)

    assert compare_scorers(data, seed=5) == compare_scorers(data, seed=5)


def test_few_sessions_never_get_a_verdict() -> None:
    data = make_feedback(0.2, sessions=12, per_session=10, seed=1)  # 12 < MIN_SESSIONS_FOR_VERDICT

    result = compare_scorers(data)

    assert result.n_sessions < MIN_SESSIONS_FOR_VERDICT
    assert result.verdict == INCONCLUSIVE


def test_sessions_without_a_useful_card_are_skipped_and_counted() -> None:
    data = make_feedback(0.2, sessions=40, per_session=10, seed=1)
    data += [r.model_copy(update={"session": "all-no", "label": 0}) for r in data[:10]]

    result = compare_scorers(data)

    assert result.n_skipped_sessions >= 1


def test_too_little_feedback_to_compare_is_an_error() -> None:
    with pytest.raises(NotEnoughData):
        compare_scorers(make_feedback(0.2, sessions=2, per_session=5, seed=1))


def test_verdict_needs_the_interval_to_exclude_zero() -> None:
    n = MIN_SESSIONS_FOR_VERDICT
    assert verdict((0.01, 0.05), n) == LEARNED_BETTER
    assert verdict((-0.05, -0.01), n) == LINEAR_BETTER
    assert verdict((-0.01, 0.05), n) == INCONCLUSIVE
    assert verdict((0.01, 0.05), n - 1) == INCONCLUSIVE
