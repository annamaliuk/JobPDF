import json
import logging
import math
from pathlib import Path

import pytest

from jobpdf.ranking import learned
from jobpdf.ranking.learned import (
    LEARNED,
    LINEAR,
    SCORER_ENV,
    UNKNOWN_SCORER,
    WEIGHTS_ENV,
    WEIGHTS_INVALID,
    WEIGHTS_MISSING,
    WEIGHTS_STALE_CALIBRATION,
    WEIGHTS_STALE_RANKING,
    FeedbackRow,
    LearnedLinearScorer,
    NotEnoughData,
    WeightsUnusable,
    calibration_fingerprint,
    fit_weights,
    load_learned_scorer,
    read_feedback,
    select_scorer,
    write_weights,
)
from jobpdf.ranking.models import CandidateSkills, VacancyFeatures
from jobpdf.ranking.score import FitComponents, LinearScorer, rank_vacancies
from jobpdf.ranking.semantic import UNCALIBRATED, Calibration
from tests.ranking.feedback_data import make_feedback

CAL = Calibration(floor=0.78, ceil=0.89, embedding_version="e5|query: ")
OTHER_CAL = Calibration(floor=0.70, ceil=0.90, embedding_version="e5|query: ")


def components(skill=None, semantic=None) -> FitComponents:
    return FitComponents(skill=skill, required_coverage=None, semantic=semantic,
                         semantic_raw=None)


@pytest.fixture(scope="module")
def rows() -> list[FeedbackRow]:
    return make_feedback(0.2, sessions=100, per_session=12, seed=1)  # truth: mostly semantic


@pytest.fixture
def weights_file(tmp_path: Path, rows: list[FeedbackRow]) -> Path:
    path = tmp_path / "learned_weights.json"
    write_weights(fit_weights(rows), path, dataset_sha256="abc", calibration=CAL)
    return path


def edit(path: Path, **changes: object) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


# --- fitting -----------------------------------------------------------------------


def test_fit_recovers_the_weight_the_labels_were_made_with(rows: list[FeedbackRow]) -> None:
    summary = fit_weights(rows)

    assert abs(summary.w_skill - 0.2) <= 0.08
    assert summary.informative
    assert summary.log_loss < summary.log_loss_prior  # better than the 0.7/0.3 prior
    assert (summary.n_rows, summary.n_skipped) == (1200, 0)


def test_weights_are_two_decimals_and_add_up_to_one(rows: list[FeedbackRow]) -> None:
    summary = fit_weights(rows)

    assert math.isclose(summary.w_skill + summary.w_semantic, 1.0)
    assert round(summary.w_skill, 2) == summary.w_skill >= 0
    assert round(summary.w_semantic, 2) == summary.w_semantic >= 0


def test_a_strong_prior_wins_and_no_prior_follows_the_data(rows: list[FeedbackRow]) -> None:
    assert fit_weights(rows, prior_strength=1e6).w_skill == 0.7
    assert fit_weights(rows, prior_strength=0.0).w_skill < 0.3


def test_little_data_is_pulled_toward_the_prior() -> None:
    few = make_feedback(0.2, sessions=4, per_session=12, seed=3)  # 48 cards

    free = fit_weights(few, prior_strength=0.0).w_skill
    shrunk = fit_weights(few).w_skill

    assert free < shrunk < 0.7


def test_feedback_that_works_against_the_score_keeps_the_prior(rows: list[FeedbackRow]) -> None:
    inverted = [r.model_copy(update={"label": 1 - r.label}) for r in rows]

    summary = fit_weights(inverted)

    assert not summary.informative
    assert summary.slope == 0.0
    assert summary.w_skill == 0.7


def test_rows_missing_a_score_are_skipped_and_counted(rows: list[FeedbackRow]) -> None:
    partial = [rows[0].model_copy(update={"skill": None}),
               rows[1].model_copy(update={"semantic": None})]

    summary = fit_weights(partial + rows)

    assert (summary.n_rows, summary.n_skipped) == (len(rows), 2)  # the two partial rows are unused


def test_too_few_rows_or_a_single_label_is_refused(rows: list[FeedbackRow]) -> None:
    with pytest.raises(NotEnoughData, match="at least 30"):
        fit_weights(rows[:29])
    only_useful = [r.model_copy(update={"label": 1}) for r in rows[:100]]
    with pytest.raises(NotEnoughData, match="each label"):
        fit_weights(only_useful)


def test_fit_is_deterministic(rows: list[FeedbackRow]) -> None:
    assert fit_weights(rows) == fit_weights(rows)


# --- feedback files ----------------------------------------------------------------


def test_read_feedback_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "fb.jsonl"
    path.write_text(
        '{"session":"a","vacancy_id":"v1","skill":0.5,"semantic":null,"band":"partial","label":1}\n'
        "\n"
        '{"session":"a","vacancy_id":"v2","skill":0.1,"semantic":0.2,"label":0}\n',
        encoding="utf-8",
    )

    first, second = read_feedback(path)

    assert (first.skill, first.semantic, first.band.value, first.label) == (0.5, None, "partial", 1)
    assert (second.band, second.label) == (None, 0)


@pytest.mark.parametrize("bad_line", [
    # an extra field, which is how CV text could leak into the file
    '{"session":"a","vacancy_id":"v","skill":0.5,"label":1,"cv_text":"secret words"}',
    '{"session":"a","vacancy_id":"v","skill":1.5,"label":1}',  # score out of range
    '{"session":"a","vacancy_id":"v","skill":0.5,"label":2}',  # label not 0/1
    '{"session":"","vacancy_id":"v","skill":0.5,"label":1}',  # empty session
    "not json at all secret words",
])
def test_invalid_rows_are_named_by_line_number_never_by_content(
    tmp_path: Path, bad_line: str
) -> None:
    path = tmp_path / "fb.jsonl"
    good = '{"session":"a","vacancy_id":"v1","skill":0.5,"semantic":0.5,"label":1}'
    path.write_text(f"{good}\n{bad_line}\n", encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        read_feedback(path)

    assert "line(s): 2" in str(caught.value)
    assert "secret" not in str(caught.value)


# --- the scorer --------------------------------------------------------------------


def test_learned_scorer_is_a_linear_scorer_with_its_own_name_and_weights() -> None:
    scorer = LearnedLinearScorer(0.2, 0.8)

    assert isinstance(scorer, LinearScorer)
    assert scorer.name == "learned_linear"
    assert scorer.weights == {"skill": 0.2, "semantic": 0.8}
    assert scorer.score(components(skill=0.5, semantic=1.0)) == pytest.approx(0.9)


def test_missing_part_falls_back_exactly_like_the_linear_scorer() -> None:
    learned_scorer, linear = LearnedLinearScorer(0.2, 0.8), LinearScorer()

    for c in (components(skill=0.4), components(semantic=0.6), components()):
        assert learned_scorer.score(c) == linear.score(c)


def test_other_weights_can_flip_an_order() -> None:
    a, b = components(skill=0.9, semantic=0.1), components(skill=0.5, semantic=0.9)

    assert LinearScorer().score(a) > LinearScorer().score(b)
    assert LearnedLinearScorer(0.2, 0.8).score(a) < LearnedLinearScorer(0.2, 0.8).score(b)


def test_weights_that_do_not_add_up_are_rejected() -> None:
    with pytest.raises(ValueError):
        LearnedLinearScorer(0.5, 0.6)


def test_ranking_result_says_which_scorer_ran() -> None:
    vacancy = VacancyFeatures(id="v1", source="synthetic", title="Data analyst")

    result = rank_vacancies(CandidateSkills(), None, [vacancy],
                            scorer=LearnedLinearScorer(0.2, 0.8))

    assert result.scorer_name == "learned_linear"
    assert result.weights == {"skill": 0.2, "semantic": 0.8}


# --- the weights file --------------------------------------------------------------


def test_weights_file_round_trip_and_provenance(weights_file: Path, rows) -> None:
    stored = json.loads(weights_file.read_text(encoding="utf-8"))
    scorer = load_learned_scorer(weights_file, calibration=CAL)

    assert scorer.weights == stored["weights"]
    assert stored["n_rows"] == len(rows) and stored["dataset_sha256"] == "abc"
    assert stored["calibration_fingerprint"] == calibration_fingerprint(CAL)
    assert stored["log_loss"] < stored["log_loss_prior"]


@pytest.mark.parametrize(("change", "code"), [
    ({"weights": {"skill": 0.5, "semantic": 0.6}}, WEIGHTS_INVALID),  # does not add up to 1
    ({"weights": {"skill": 1.2, "semantic": -0.2}}, WEIGHTS_INVALID),  # negative
    ({"weights": {"skill": 0.5}}, WEIGHTS_INVALID),  # part missing
    ({"format_version": 99}, WEIGHTS_INVALID),
    ({"surprise": 1}, WEIGHTS_INVALID),  # unknown field
    ({"ranking_version": "0"}, WEIGHTS_STALE_RANKING),
    ({"calibration_fingerprint": "0123456789abcdef"}, WEIGHTS_STALE_CALIBRATION),
])
def test_unusable_weights_files_are_refused_with_a_code(
    weights_file: Path, change: dict, code: str
) -> None:
    edit(weights_file, **change)

    with pytest.raises(WeightsUnusable) as caught:
        load_learned_scorer(weights_file, calibration=CAL)

    assert caught.value.code == code


def test_missing_and_garbled_files_are_refused(tmp_path: Path) -> None:
    with pytest.raises(WeightsUnusable) as missing:
        load_learned_scorer(tmp_path / "nope.json", calibration=CAL)
    garbled = tmp_path / "bad.json"
    garbled.write_text("{not json", encoding="utf-8")
    with pytest.raises(WeightsUnusable) as bad:
        load_learned_scorer(garbled, calibration=CAL)

    assert (missing.value.code, bad.value.code) == (WEIGHTS_MISSING, WEIGHTS_INVALID)


def test_weights_go_stale_when_the_calibration_is_remeasured(weights_file: Path) -> None:
    with pytest.raises(WeightsUnusable) as caught:
        load_learned_scorer(weights_file, calibration=OTHER_CAL)

    assert caught.value.code == WEIGHTS_STALE_CALIBRATION


def test_calibration_fingerprint() -> None:
    assert calibration_fingerprint(CAL) == calibration_fingerprint(
        Calibration(floor=0.78, ceil=0.89, embedding_version="e5|query: ", provisional=False))
    assert calibration_fingerprint(CAL) != calibration_fingerprint(OTHER_CAL)
    assert calibration_fingerprint(CAL) != calibration_fingerprint(
        Calibration(floor=0.78, ceil=0.89, embedding_version="other model"))
    assert calibration_fingerprint(Calibration()) == UNCALIBRATED


# --- the feature flag --------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(SCORER_ENV, raising=False)
    monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    monkeypatch.setattr(learned, "WEIGHTS_PATH", tmp_path / "default_weights.json")  # absent


@pytest.mark.parametrize("value", [None, "", "  ", LINEAR])
def test_flag_is_off_by_default_even_when_valid_weights_exist(
    monkeypatch: pytest.MonkeyPatch, weights_file: Path, value: str | None
) -> None:
    # A valid file at the default path must not switch anything on by itself: the
    # fallback would otherwise hide a flag that defaults to "learned".
    monkeypatch.setattr(learned, "WEIGHTS_PATH", weights_file)
    if value is not None:
        monkeypatch.setenv(SCORER_ENV, value)

    scorer = select_scorer(calibration=CAL)

    assert type(scorer) is LinearScorer
    assert scorer.weights == {"skill": 0.7, "semantic": 0.3}


def test_the_default_weights_path_is_used_when_the_flag_is_on(
    monkeypatch: pytest.MonkeyPatch, weights_file: Path
) -> None:
    monkeypatch.setattr(learned, "WEIGHTS_PATH", weights_file)
    monkeypatch.setenv(SCORER_ENV, LEARNED)

    assert isinstance(select_scorer(calibration=CAL), LearnedLinearScorer)


def test_flag_on_uses_the_learned_weights(
    monkeypatch: pytest.MonkeyPatch, weights_file: Path
) -> None:
    monkeypatch.setenv(SCORER_ENV, LEARNED)
    monkeypatch.setenv(WEIGHTS_ENV, str(weights_file))

    scorer = select_scorer(calibration=CAL)

    assert isinstance(scorer, LearnedLinearScorer)
    assert scorer.weights["skill"] < 0.3


@pytest.mark.parametrize("value", ["learned", " Learned ", "LEARNED"])
def test_flag_value_ignores_case_and_spaces(
    monkeypatch: pytest.MonkeyPatch, weights_file: Path, value: str
) -> None:
    monkeypatch.setenv(SCORER_ENV, value)

    assert isinstance(select_scorer(weights_path=weights_file, calibration=CAL),
                      LearnedLinearScorer)


def test_an_explicit_name_beats_the_environment(
    monkeypatch: pytest.MonkeyPatch, weights_file: Path
) -> None:
    monkeypatch.setenv(SCORER_ENV, LEARNED)
    assert type(select_scorer(LINEAR, weights_path=weights_file, calibration=CAL)) is LinearScorer

    monkeypatch.setenv(SCORER_ENV, LINEAR)
    assert isinstance(select_scorer(LEARNED, weights_path=weights_file, calibration=CAL),
                      LearnedLinearScorer)


def test_flag_on_without_weights_falls_back_and_says_why(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(SCORER_ENV, LEARNED)

    with caplog.at_level(logging.WARNING):
        scorer = select_scorer(calibration=CAL)

    assert type(scorer) is LinearScorer
    assert WEIGHTS_MISSING in caplog.text


@pytest.mark.parametrize(("change", "code"), [
    ({"ranking_version": "0"}, WEIGHTS_STALE_RANKING),
    ({"calibration_fingerprint": "0123456789abcdef"}, WEIGHTS_STALE_CALIBRATION),
    ({"weights": {"skill": 0.9, "semantic": 0.9}}, WEIGHTS_INVALID),
])
def test_stale_or_broken_weights_never_reach_ranking(
    weights_file: Path, caplog: pytest.LogCaptureFixture, change: dict, code: str
) -> None:
    edit(weights_file, **change)

    with caplog.at_level(logging.WARNING):
        scorer = select_scorer(LEARNED, weights_path=weights_file, calibration=CAL)

    assert type(scorer) is LinearScorer
    assert code in caplog.text


def test_unknown_flag_value_falls_back_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(SCORER_ENV, "gradient_boosting")

    with caplog.at_level(logging.WARNING):
        scorer = select_scorer(calibration=CAL)

    assert type(scorer) is LinearScorer
    assert UNKNOWN_SCORER in caplog.text


def test_selection_never_raises_on_a_garbage_file(tmp_path: Path) -> None:
    path = tmp_path / "w.json"
    path.write_bytes(b"\xff\xfe\x00 not utf-8")

    assert type(select_scorer(LEARNED, weights_path=path, calibration=CAL)) is LinearScorer
