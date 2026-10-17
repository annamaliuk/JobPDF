"""Offline evaluation of the ranking on JM-25's synthetic pairs (JM-26).

The question this answers: for each candidate, does the ranking put the vacancies that
fit best (by how the pairs were built, never by the formula) at the top? Everything is
computed per candidate, because one candidate's 40 pairs are not independent, then
averaged over candidates with a bootstrap interval.

Metrics, per candidate:

- Spearman correlation between list position and grade (primary). Position is product
  order: band, then fit, then newest posting, then id, exactly as ``rank_vacancies``.
- NDCG@k with graded gains ``2**grade - 1`` (JM-27's ``ndcg_at_k`` is binary).
- Kendall tau-b, a cross-check that handles the many grade-0 ties more cleanly.
- Band agreement with JM-25's expected band: accuracy and macro-F1.
- Across all pairs: ROC AUC of the raw semantic cosine, grade 3 against grade 0.

Matching modes say how skills become keys: ``perfect`` uses JM-25's ground truth,
``raw`` uses ``normalize()`` only (no taxonomy, the baseline for the core claim), and
``real`` runs JM-20's normalization service. Variants say how the score is combined:
the current weights, skills only, semantic only, a weight sweep, fit order without
bands, and a TF-IDF text baseline with no skills and no model.

Two tuning steps pick values by grouped cross-validation (folds split candidates, never
pairs): the skill weight, and the band thresholds. A tuned value is recommended only
when its held-out paired difference to the current setting has a 95% bootstrap
interval above zero (``decide``); otherwise the current constant stays.

Pure numpy; prints and stores numbers only, never CV or vacancy text.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np

from jobpdf.normalization.text import normalize
from jobpdf.ranking.models import CandidateSkills, SkillRef, VacancyFeatures
from jobpdf.ranking.overlap import SkillOverlap, skill_overlap
from jobpdf.ranking.score import (
    BAND_ORDER,
    PARTIAL_COVERAGE,
    SEM_MIN,
    STRONG_COVERAGE,
    W_SKILL,
    Band,
    FitComponents,
    LinearScorer,
    assign_band,
)
from jobpdf.ranking.scorer_eval import spearman
from jobpdf.ranking.semantic import (
    Calibration,
    build_vacancy_texts,
    embed_candidate,
    embed_vacancy,
    semantic_scores,
)
from jobpdf.ranking.synthetic import (
    Dataset,
    SyntheticCandidate,
    truth_candidate_skills,
    truth_vacancy,
)

MODES = ("perfect", "raw", "real")
DEFAULT_K = 10
DEFAULT_FOLDS = 5
DEFAULT_SEED = 26
BOOTSTRAP_SAMPLES = 2000
WEIGHT_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)
STRONG_GRID = (0.7, 0.75, 0.8, 0.85, 0.9)
PARTIAL_GRID = (0.4, 0.5, 0.6)
SEM_MIN_GRID = (0.2, 0.3, 0.4)
# A 95% interval on the headline Spearman wider than this means the set is too small
# to decide anything: regenerate JM-25 larger (--vacancies-per-family 10 --candidates 40).
NOISY_CI_WIDTH = 0.1
BANDS = ("strong", "partial", "weak")
CURRENT = "current"
TFIDF = "tfidf"
_TOKEN = re.compile(r"\w+", re.UNICODE)


# --- metrics -----------------------------------------------------------------------


def ndcg_graded(grades_in_rank_order: Sequence[int], k: int = DEFAULT_K) -> float | None:
    """NDCG@k with gain 2**grade - 1; None when no item has a positive grade."""
    if not any(g > 0 for g in grades_in_rank_order):
        return None

    def dcg(grades: Sequence[int]) -> float:
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))

    return dcg(grades_in_rank_order) / dcg(sorted(grades_in_rank_order, reverse=True))


def kendall_tau_b(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Kendall's tau-b; None if either side is constant or the lists are too short."""
    if len(a) != len(b) or len(a) < 2:
        return None
    concordant = discordant = ties_a = ties_b = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif (da > 0) == (db > 0):
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b))
    return (concordant - discordant) / denom if denom else None


def macro_f1(expected: Sequence[str], predicted: Sequence[str],
             labels: Sequence[str] = BANDS) -> float | None:
    """Unweighted mean F1 over the labels that occur in either list."""
    scores = []
    for label in labels:
        tp = sum(e == label and p == label for e, p in zip(expected, predicted, strict=True))
        n_pred = sum(p == label for p in predicted)
        n_true = sum(e == label for e in expected)
        if n_pred == 0 and n_true == 0:
            continue
        scores.append(2 * tp / (n_pred + n_true))
    return sum(scores) / len(scores) if scores else None


def roc_auc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """Probability that a random positive outranks a random negative; ties count half."""
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def bootstrap_mean_ci(values: Sequence[float], samples: int = BOOTSTRAP_SAMPLES,
                      seed: int = DEFAULT_SEED) -> tuple[float, float] | None:
    """95% percentile interval of the mean, resampling the values (one per candidate)."""
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(samples, len(arr)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test_p(differences: Sequence[float]) -> float | None:
    """Two-sided exact sign test; zero differences are dropped."""
    signs = [d for d in differences if d != 0]
    if not signs:
        return None
    n, pos = len(signs), sum(d > 0 for d in signs)
    tail = sum(math.comb(n, i) for i in range(min(pos, n - pos) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# --- pairs -------------------------------------------------------------------------


@dataclass(frozen=True)
class PairData:
    """Everything a variant needs about one candidate-vacancy pair, computed once."""

    candidate_id: str
    vacancy_id: str
    grade: int
    expected_band: str
    trap: str | None
    overlap: SkillOverlap
    semantic: float | None  # calibrated 0..1
    semantic_raw: float | None  # cosine before rescaling
    tfidf: float
    posted: float | None


class Normalizer(Protocol):
    """The two calls of JM-20's NormalizationService that the real mode needs."""

    def normalize_profile(self, profile: Any) -> Any: ...

    def normalize_skill_groups(self, groups: Mapping[str, Sequence[str]]) -> Any: ...


def _profile_raw_skills(profile: Any) -> Iterable[tuple[str, str]]:
    for i, skill in enumerate(profile.skills):
        yield f"skills[{i}]", skill.name
    for j, job in enumerate(profile.experience):
        for k, raw in enumerate(job.skills_used):
            yield f"experience[{j}].skills_used[{k}]", raw


def _raw_candidate_skills(profile: Any) -> CandidateSkills:
    refs: dict[str, SkillRef] = {}
    sources: dict[str, list[str]] = {}
    for path, raw in _profile_raw_skills(profile):
        ref = SkillRef(raw=raw, normalized=normalize(raw))
        if ref.is_empty:
            continue
        refs.setdefault(ref.key, ref)
        sources.setdefault(ref.key, []).append(path)
    return CandidateSkills(refs=refs, sources=sources)


def _skill_sides(dataset: Dataset, mode: str, normalizer: Normalizer | None) -> tuple[
        dict[str, VacancyFeatures], dict[str, CandidateSkills]]:
    if mode == "perfect":
        return ({v.id: truth_vacancy(dataset, v) for v in dataset.vacancies},
                {c.id: truth_candidate_skills(dataset, c.id) for c in dataset.candidates})
    if mode == "raw":  # vacancies are stored with raw keys already
        return ({v.id: v for v in dataset.vacancies},
                {c.id: _raw_candidate_skills(c.profile) for c in dataset.candidates})
    if mode != "real":
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if normalizer is None:
        raise ValueError("the real mode needs a normalizer (JM-20's NormalizationService)")
    vacancies = {}
    for v in dataset.vacancies:
        groups = normalizer.normalize_skill_groups({
            "required": [r.raw for r in v.required_skills],
            "preferred": [r.raw for r in v.preferred_skills]})
        vacancies[v.id] = v.model_copy(update={
            "required_skills": [SkillRef.from_match(m) for m in groups["required"]],
            "preferred_skills": [SkillRef.from_match(m) for m in groups["preferred"]]})
    candidates = {c.id: CandidateSkills.from_skill_matches(
        normalizer.normalize_profile(c.profile).matches) for c in dataset.candidates}
    return vacancies, candidates


def _candidate_text(c: SyntheticCandidate) -> str:
    p = c.profile
    parts = [p.summary or ""] + [f"{j.title} {j.description or ''}" for j in p.experience]
    return " ".join(parts + [s.name for s in p.skills])


def _vacancy_text(v: VacancyFeatures) -> str:
    return " ".join(build_vacancy_texts(v) + [r.raw for r in v.required_skills]
                    + [r.raw for r in v.preferred_skills])


def tfidf_similarities(left: Sequence[str], right: Sequence[str]) -> np.ndarray:
    """Cosine of smoothed TF-IDF vectors, left x right, with one vocabulary over both."""
    docs = [[t.lower() for t in _TOKEN.findall(text)] for text in [*left, *right]]
    vocab = {t: i for i, t in enumerate(sorted({t for d in docs for t in d}))}
    counts = np.zeros((len(docs), len(vocab)), dtype=np.float64)
    for row, doc in enumerate(docs):
        for token, n in Counter(doc).items():
            counts[row, vocab[token]] = n
    df = (counts > 0).sum(axis=0)
    idf = np.log((1 + len(docs)) / (1 + df)) + 1
    weights = counts * idf
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    weights = np.divide(weights, norms, out=np.zeros_like(weights), where=norms > 0)
    return weights[: len(left)] @ weights[len(left):].T


def _timestamp(value: datetime | None) -> float | None:
    if value is None:
        return None
    return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).timestamp()


def build_pairs(dataset: Dataset, mode: str, *, embedder: Any = None,
                normalizer: Normalizer | None = None,
                calibration: Calibration | None = None) -> list[PairData]:
    """One PairData per labelled pair. Without an embedder the semantic part is None,
    so semantic variants fall back to skills (as production does without vectors)."""
    vacancies, candidates = _skill_sides(dataset, mode, normalizer)
    ordered_v = [vacancies[v.id] for v in dataset.vacancies]
    tfidf = tfidf_similarities([_candidate_text(c) for c in dataset.candidates],
                               [_vacancy_text(v) for v in dataset.vacancies])
    v_index = {v.id: i for i, v in enumerate(dataset.vacancies)}
    c_index = {c.id: i for i, c in enumerate(dataset.candidates)}
    semantic: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    if embedder is not None:
        embedded = [embed_vacancy(v, embedder) for v in ordered_v]
        for c in dataset.candidates:
            vectors = embed_candidate(c.profile, embedder)
            for v, s in zip(ordered_v, semantic_scores(vectors, embedded, calibration),
                            strict=True):
                semantic[(c.id, v.id)] = (s.score, s.raw)
    truth = dataset.truth.candidates
    out = []
    for p in dataset.pairs:
        vacancy = vacancies[p.vacancy_id]
        sem, raw = semantic.get((p.candidate_id, p.vacancy_id), (None, None))
        out.append(PairData(
            candidate_id=p.candidate_id, vacancy_id=p.vacancy_id, grade=p.grade,
            expected_band=p.expected_band, trap=truth[p.candidate_id].trap,
            overlap=skill_overlap(candidates[p.candidate_id], vacancy),
            semantic=sem, semantic_raw=raw,
            tfidf=float(tfidf[c_index[p.candidate_id], v_index[p.vacancy_id]]),
            posted=_timestamp(vacancy.posted_at),
        ))
    return out


# --- variants ----------------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    strong: float = STRONG_COVERAGE
    partial: float = PARTIAL_COVERAGE
    sem_min: float = SEM_MIN


@dataclass(frozen=True)
class Variant:
    name: str
    w_skill: float | None  # None: the TF-IDF baseline
    use_semantic: bool = True
    use_bands: bool = True
    thresholds: Thresholds = field(default_factory=Thresholds)


def standard_variants() -> list[Variant]:
    return [
        Variant(CURRENT, W_SKILL),
        Variant("skills_only", 1.0, use_semantic=False),
        Variant("semantic_only", 0.0),
        *[Variant(f"w_skill={w:.1f}", w) for w in WEIGHT_GRID if not math.isclose(w, W_SKILL)],
        Variant("fit_order", W_SKILL, use_bands=False),
        Variant(TFIDF, None, use_semantic=False, use_bands=False),
    ]


def score_pair(pair: PairData, variant: Variant) -> tuple[float | None, Band | None]:
    """(fit, band) under a variant; band is None when the variant does not use bands."""
    if variant.w_skill is None:
        return pair.tfidf, None
    semantic = pair.semantic if variant.use_semantic else None
    components = FitComponents(skill=pair.overlap.score,
                               required_coverage=pair.overlap.required_coverage,
                               semantic=semantic, semantic_raw=None)
    fit = LinearScorer(variant.w_skill, 1 - variant.w_skill).score(components)
    t = variant.thresholds
    band, _ = assign_band(pair.overlap, semantic, strong_coverage=t.strong,
                          partial_coverage=t.partial, sem_min=t.sem_min)
    return fit, band


def rank_candidate(pairs: Sequence[PairData], variant: Variant) -> list[
        tuple[PairData, Band | None]]:
    """Product order: band, fit (None last), newest posting, id."""
    scored = [(p, *score_pair(p, variant)) for p in pairs]

    def key(item: tuple[PairData, float | None, Band | None]) -> tuple:
        p, fit, band = item
        return (BAND_ORDER[band] if variant.use_bands and band is not None else 0,
                fit is None, -(fit or 0.0), p.posted is None, -(p.posted or 0.0), p.vacancy_id)

    return [(p, band) for p, _, band in sorted(scored, key=key)]


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    trap: str | None
    spearman: float | None
    ndcg: float | None
    kendall: float | None
    expected_bands: tuple[str, ...]
    predicted_bands: tuple[str, ...]  # empty when the variant has no bands


def evaluate_candidate(pairs: Sequence[PairData], variant: Variant,
                       k: int = DEFAULT_K) -> CandidateResult:
    ranked = rank_candidate(pairs, variant)
    grades = [p.grade for p, _ in ranked]
    position = [-float(i) for i in range(len(ranked))]  # earlier = higher
    bands = tuple(b.value for _, b in ranked if b is not None) if variant.use_bands else ()
    return CandidateResult(
        candidate_id=pairs[0].candidate_id, trap=pairs[0].trap,
        spearman=spearman(position, [float(g) for g in grades]),
        ndcg=ndcg_graded(grades, k), kendall=kendall_tau_b(position, grades),
        expected_bands=tuple(p.expected_band for p, _ in ranked), predicted_bands=bands,
    )


def by_candidate(pairs: Iterable[PairData]) -> dict[str, list[PairData]]:
    groups: dict[str, list[PairData]] = defaultdict(list)
    for p in pairs:
        groups[p.candidate_id].append(p)
    return dict(sorted(groups.items()))


# --- summaries ---------------------------------------------------------------------


def _defined(values: Iterable[float | None]) -> list[float]:
    return [v for v in values if v is not None]


def _stat(values: Sequence[float | None], seed: int) -> dict[str, Any]:
    vals = _defined(values)
    ci = bootstrap_mean_ci(vals, seed=seed)
    return {"mean": round(float(np.mean(vals)), 4) if vals else None,
            "ci95": [round(ci[0], 4), round(ci[1], 4)] if ci else None, "n": len(vals)}


def paired(base: Sequence[CandidateResult], other: Sequence[CandidateResult],
           seed: int) -> dict[str, Any]:
    """Other minus base, per candidate, on Spearman; both lists in the same order."""
    diffs = [o.spearman - b.spearman for b, o in zip(base, other, strict=True)
             if b.spearman is not None and o.spearman is not None]
    out = _stat(diffs, seed)
    out["sign_test_p"] = None if (p := sign_test_p(diffs)) is None else round(p, 4)
    return out


def summarize(results: Sequence[CandidateResult], seed: int) -> dict[str, Any]:
    summary = {
        "spearman": _stat([r.spearman for r in results], seed),
        "ndcg": _stat([r.ndcg for r in results], seed),
        "kendall": _stat([r.kendall for r in results], seed),
    }
    expected = [e for r in results if r.predicted_bands for e in r.expected_bands]
    predicted = [p for r in results for p in r.predicted_bands]
    if predicted:
        summary["band_accuracy"] = round(
            sum(e == p for e, p in zip(expected, predicted, strict=True)) / len(predicted), 4)
        summary["band_macro_f1"] = round(macro_f1(expected, predicted) or 0.0, 4)
        summary["band_confusion"] = {e: {p: sum(1 for x, y in zip(expected, predicted,
                                                                   strict=True)
                                               if x == e and y == p) for p in BANDS}
                                     for e in BANDS}
    traps: dict[str, list[float | None]] = defaultdict(list)
    for r in results:
        traps[r.trap or "regular"].append(r.spearman)
    summary["spearman_by_case"] = {t: _stat(v, seed)["mean"] for t, v in sorted(traps.items())}
    return summary


def semantic_auc(pairs: Sequence[PairData]) -> float | None:
    pos = _defined(p.semantic_raw for p in pairs if p.grade == 3)
    neg = _defined(p.semantic_raw for p in pairs if p.grade == 0)
    auc = roc_auc(pos, neg)
    return None if auc is None else round(auc, 4)


# --- tuning ------------------------------------------------------------------------


def grouped_folds(candidate_ids: Sequence[str], folds: int = DEFAULT_FOLDS,
                  seed: int = DEFAULT_SEED) -> list[tuple[list[str], list[str]]]:
    """(train, test) candidate lists; every candidate is in exactly one test fold."""
    ids = sorted(candidate_ids)
    random.Random(seed).shuffle(ids)
    folds = max(2, min(folds, len(ids)))
    tests = [ids[i::folds] for i in range(folds)]
    return [([c for c in ids if c not in test], sorted(test)) for test in tests]


def tune(groups: Mapping[str, Sequence[PairData]], candidates: Sequence[Variant],
         k: int = DEFAULT_K, folds: int = DEFAULT_FOLDS, seed: int = DEFAULT_SEED,
         objective: Callable[[Sequence[CandidateResult]], float] | None = None,
         ) -> tuple[list[CandidateResult], list[str]]:
    """Held-out results: per fold, pick the variant with the best mean train Spearman
    (ties: the earlier one in ``candidates``), then score it on the test candidates.
    Returns results in candidate-id order and the chosen variant name per fold."""
    objective = objective or mean_spearman
    cache = {(v.name, cid): evaluate_candidate(groups[cid], v, k)
             for v in candidates for cid in groups}
    held_out: dict[str, CandidateResult] = {}
    chosen = []
    for train, test in grouped_folds(list(groups), folds, seed):
        best = max(candidates, key=lambda v: (objective([cache[(v.name, c)] for c in train]),
                                              -candidates.index(v)))
        chosen.append(best.name)
        for cid in test:
            held_out[cid] = cache[(best.name, cid)]
    return [held_out[c] for c in sorted(held_out)], chosen


def mean_spearman(results: Sequence[CandidateResult]) -> float:
    values = _defined(r.spearman for r in results)
    return float(np.mean(values)) if values else -math.inf


def weight_candidates() -> list[Variant]:
    return [Variant(f"w_skill={w:.1f}", w) for w in WEIGHT_GRID]


def threshold_candidates() -> list[Variant]:
    return [Variant(f"strong={s},partial={p},sem_min={m}", W_SKILL,
                    thresholds=Thresholds(s, p, m))
            for s in STRONG_GRID for p in PARTIAL_GRID for m in SEM_MIN_GRID if p < s]


def decide(comparison: Mapping[str, Any], chosen: Sequence[str]) -> dict[str, Any]:
    """Recommend the most chosen value only if the held-out gain is clearly positive."""
    ci = comparison.get("ci95")
    winner, votes = Counter(chosen).most_common(1)[0]
    change = bool(ci and ci[0] > 0)
    return {"recommendation": f"change to {winner}" if change else "keep current",
            "most_chosen": winner, "folds_choosing_it": votes, "folds": len(chosen),
            "reason": ("held-out Spearman gain with a 95% interval above zero" if change
                       else "held-out gain not clearly above zero")}


# --- whole run ---------------------------------------------------------------------


def run_evaluation(pairs: Sequence[PairData], *, k: int = DEFAULT_K,
                   folds: int = DEFAULT_FOLDS, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """The full report as a JSON-ready dict of numbers."""
    groups = by_candidate(pairs)
    has_semantic = any(p.semantic is not None for p in pairs)
    results = {v.name: [evaluate_candidate(groups[c], v, k) for c in groups]
               for v in standard_variants()}
    variants = {}
    for name, rs in results.items():
        variants[name] = summarize(rs, seed)
        if name != CURRENT:
            variants[name]["vs_current"] = paired(results[CURRENT], rs, seed)

    tuned_w, chosen_w = tune(groups, weight_candidates(), k, folds, seed)
    w_cmp = paired(results[CURRENT], tuned_w, seed)
    tuning = {"weight": {**summarize(tuned_w, seed), "vs_current": w_cmp,
                         "chosen_per_fold": chosen_w, **decide(w_cmp, chosen_w)}}
    if not has_semantic:
        tuning["weight"].update(recommendation="keep current",
                                reason="no semantic scores in this run, so the weight "
                                       "cannot be measured")
    tuned_t, chosen_t = tune(groups, threshold_candidates(), k, folds, seed)
    t_cmp = paired(results[CURRENT], tuned_t, seed)
    tuning["thresholds"] = {**summarize(tuned_t, seed), "vs_current": t_cmp,
                            "chosen_per_fold": chosen_t, **decide(t_cmp, chosen_t)}

    ci = variants[CURRENT]["spearman"]["ci95"]
    oracle = [spearman([-float(i) for i in range(len(ps))],
                       sorted((float(p.grade) for p in ps), reverse=True))
              for ps in groups.values()]
    return {
        "k": k, "folds": folds, "seed": seed, "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "n_candidates": len(groups), "n_pairs": len(pairs),
        "has_semantic": has_semantic,
        # The best Spearman any order can reach: ties in the grades cap it below 1.
        "oracle_spearman": _stat(oracle, seed)["mean"],
        "semantic_auc": semantic_auc(pairs),
        "noisy": bool(ci and ci[1] - ci[0] > NOISY_CI_WIDTH),
        "variants": variants,
        "tuning": tuning,
    }

