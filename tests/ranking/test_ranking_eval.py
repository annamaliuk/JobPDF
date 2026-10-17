"""JM-26 ranking evaluation: metrics against hand-computed values, then whole runs on
JM-25's synthetic set with no network, model or API key."""

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobpdf.normalization.text import normalize
from jobpdf.ranking.ranking_eval import (
    CURRENT,
    TFIDF,
    Variant,
    bootstrap_mean_ci,
    build_pairs,
    by_candidate,
    decide,
    evaluate_candidate,
    grouped_folds,
    kendall_tau_b,
    macro_f1,
    ndcg_graded,
    roc_auc,
    run_evaluation,
    sign_test_p,
    tfidf_similarities,
    tune,
)
from jobpdf.ranking.score import W_SKILL
from jobpdf.ranking.synthetic import generate, truth_key
from jobpdf.ranking.synthetic_catalog import SKILLS
from tests.normalization.fakes import FakeEmbedder

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def dataset():
    return generate()


@pytest.fixture(scope="module")
def perfect(dataset):
    return build_pairs(dataset, "perfect")


# --- metrics -----------------------------------------------------------------------


def test_ndcg_graded_is_one_for_the_ideal_order_and_uses_exponential_gains() -> None:
    assert ndcg_graded([3, 2, 0, 0]) == 1.0
    # gains 7, 1 at ranks 1 and 2 against the ideal 7, 1: swapped order
    swapped = (1 / math.log2(2) + 7 / math.log2(3)) / (7 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_graded([1, 3]) == pytest.approx(swapped)
    assert ndcg_graded([0, 0, 0]) is None


def test_ndcg_graded_only_counts_the_top_k() -> None:
    assert ndcg_graded([0, 0, 3], k=2) == 0.0


def test_kendall_tau_b_matches_hand_computed_values() -> None:
    assert kendall_tau_b([1, 2, 3], [1, 2, 3]) == 1.0
    assert kendall_tau_b([1, 2, 3], [3, 2, 1]) == -1.0
    # 2 concordant, 0 discordant, 1 pair tied only in b: 2 / sqrt(2 * 3)
    assert kendall_tau_b([1, 2, 3], [1, 1, 2]) == pytest.approx(2 / math.sqrt(6))
    assert kendall_tau_b([1, 1, 1], [1, 2, 3]) is None


def test_macro_f1_averages_per_band_f1() -> None:
    expected = ["strong", "strong", "partial", "weak"]
    predicted = ["strong", "partial", "partial", "weak"]
    assert macro_f1(expected, predicted) == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)


def test_roc_auc_counts_ties_as_half() -> None:
    assert roc_auc([3.0, 2.0], [1.0, 2.0]) == pytest.approx(3.5 / 4)
    assert roc_auc([], [1.0]) is None


def test_bootstrap_is_seeded_and_collapses_on_constant_values() -> None:
    values = [0.1, 0.5, 0.9, 0.3]
    assert bootstrap_mean_ci(values, seed=1) == bootstrap_mean_ci(values, seed=1)
    assert bootstrap_mean_ci([0.4] * 5) == pytest.approx((0.4, 0.4))
    assert bootstrap_mean_ci([]) is None


def test_sign_test_drops_zeros() -> None:
    assert sign_test_p([1, 1, 1, 1, 1]) == pytest.approx(2 / 32)
    assert sign_test_p([1, -1, 0]) == 1.0
    assert sign_test_p([0, 0]) is None


def test_tfidf_similarity_is_one_for_identical_text_and_zero_for_disjoint() -> None:
    sims = tfidf_similarities(["python airflow pipelines"],
                              ["Python Airflow pipelines", "pastry chef"])
    assert sims[0, 0] == pytest.approx(1.0)
    assert sims[0, 1] == 0.0


def test_grouped_folds_put_every_candidate_in_exactly_one_test_fold() -> None:
    ids = [f"c{i}" for i in range(20)]
    folds = grouped_folds(ids, 5, seed=3)
    tests = [c for _, test in folds for c in test]
    assert sorted(tests) == sorted(ids)
    assert all(not set(train) & set(test) for train, test in folds)
    assert folds == grouped_folds(ids, 5, seed=3)


def test_decide_changes_only_when_the_interval_is_above_zero() -> None:
    assert decide({"ci95": [0.01, 0.05]}, ["a", "a", "b"])["recommendation"] == "change to a"
    assert decide({"ci95": [-0.01, 0.05]}, ["a"])["recommendation"] == "keep current"


# --- ranking the synthetic set -----------------------------------------------------


def test_product_order_puts_bands_first(perfect) -> None:
    groups = by_candidate(perfect)
    pairs = next(iter(groups.values()))
    result = evaluate_candidate(pairs, Variant(CURRENT, W_SKILL))
    order = {"strong": 0, "partial": 1, "weak": 2}
    ranks = [order[b] for b in result.predicted_bands]
    assert ranks == sorted(ranks)
    assert result.spearman is not None and result.ndcg is not None


def test_tune_picks_the_variant_that_wins_on_training_candidates(perfect) -> None:
    groups = by_candidate(perfect)
    held_out, chosen = tune(groups, [Variant(TFIDF, None, False, False),
                                     Variant(CURRENT, W_SKILL)])
    assert chosen == [CURRENT] * 5
    assert len(held_out) == len(groups)


def test_taxonomy_helps_the_alias_only_candidate(dataset, perfect) -> None:
    raw = build_pairs(dataset, "raw")
    [cid] = [c for c, t in dataset.truth.candidates.items() if t.trap == "alias_only"]
    variant = Variant(CURRENT, W_SKILL)
    with_truth = evaluate_candidate(by_candidate(perfect)[cid], variant).spearman
    without = evaluate_candidate(by_candidate(raw)[cid], variant).spearman
    assert with_truth > without


class FakeNormalizer:
    """Maps names and aliases to ground-truth keys, like a perfect JM-20 would."""

    def __init__(self) -> None:
        self.lookup = {normalize(n): s.slug for s in SKILLS.values()
                       for n in (s.name, *s.aliases)}

    def _match(self, raw: str):
        slug = self.lookup.get(normalize(raw))
        return SimpleNamespace(concept_id=truth_key(slug) if slug else None, raw=raw,
                               normalized=normalize(raw),
                               preferred_label=SKILLS[slug].name if slug else None)

    def normalize_profile(self, profile):
        matches = {f"skills[{i}]": self._match(s.name) for i, s in enumerate(profile.skills)}
        return SimpleNamespace(matches=matches)

    def normalize_skill_groups(self, groups):
        return {name: [self._match(r) for r in raws] for name, raws in groups.items()}


def test_real_mode_with_a_perfect_normalizer_matches_the_perfect_mode(dataset, perfect) -> None:
    real = build_pairs(dataset, "real", normalizer=FakeNormalizer())
    assert [p.overlap.score for p in real] == [p.overlap.score for p in perfect]


def test_real_mode_needs_a_normalizer(dataset) -> None:
    with pytest.raises(ValueError, match="normalizer"):
        build_pairs(dataset, "real")


def test_a_full_run_with_a_fake_embedder_reports_every_variant(dataset) -> None:
    embedder = FakeEmbedder()
    embedder.model_name, embedder.revision = "fake/e5", "rev1"
    pairs = build_pairs(dataset, "perfect", embedder=embedder)
    report = run_evaluation(pairs)

    assert report["has_semantic"] and report["semantic_auc"] is not None
    assert {"current", "skills_only", "semantic_only", "fit_order", "tfidf"} <= set(
        report["variants"])
    assert report["variants"]["semantic_only"]["spearman"]["mean"] != (
        report["variants"]["skills_only"]["spearman"]["mean"])
    assert report["oracle_spearman"] >= report["variants"]["current"]["spearman"]["mean"]
    assert len(report["tuning"]["weight"]["chosen_per_fold"]) == 5
    text = json.dumps(report)
    assert "Engineer" not in text and "syn-v-" not in text


def test_script_prints_numbers_and_writes_the_report_under_its_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = importlib.util.spec_from_file_location("eval_ranking",
                                                  ROOT / "scripts" / "eval_ranking.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = tmp_path / "ranking_eval.json"

    code = module.main(["--mode", "raw", "--no-semantic", "--write", "--results",
                        str(results)])

    printed = capsys.readouterr().out
    stored = json.loads(results.read_text(encoding="utf-8"))
    assert code == 0
    assert "current" in printed and "tfidf" in printed
    assert "Engineer" not in printed
    assert stored["raw"]["provenance"]["dataset_files"]["pairs.jsonl"]
    assert stored["raw"]["has_semantic"] is False


def test_script_refuses_real_mode_without_the_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = importlib.util.spec_from_file_location("eval_ranking",
                                                  ROOT / "scripts" / "eval_ranking.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    normalizer = FakeNormalizer()
    normalizer.versions = SimpleNamespace(index="unavailable")
    results = tmp_path / "ranking_eval.json"

    code = module.main(["--mode", "real", "--no-semantic", "--write", "--results",
                        str(results)], normalizer=normalizer)

    assert code == 2
    assert "skill index" in capsys.readouterr().err
    assert not results.exists()
