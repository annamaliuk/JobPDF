import pytest

from jobpdf.ranking import insights
from jobpdf.ranking.insights import (
    IN_DEMAND_LIMIT,
    LEARN_LIMIT,
    MAX_ADDED,
    TOP_N,
    WITHIN_REACH_LIMIT,
    InsightsResult,
    compute_insights,
    in_demand,
    skills_worth_learning,
    within_reach,
    working_set,
)
from jobpdf.ranking.models import CandidateSkills, SkillRef, VacancyFeatures
from jobpdf.ranking.score import (
    SEM_MIN,
    THRESHOLDS,
    Band,
    LinearScorer,
    RankingResult,
    VacancyAdjustment,
    _sort_key,
    score_vacancy,
)
from jobpdf.ranking.semantic import SemanticScore

B = Band


def ref(name: str) -> SkillRef:
    return SkillRef(concept_id=f"esco:{name}", raw=name, normalized=name.lower(), label=name)


def candidate(*names: str) -> CandidateSkills:
    refs = {r.key: r for r in map(ref, names)}
    return CandidateSkills(refs=refs, sources={key: ["skills[0]"] for key in refs})


def vacancy(vid: str, required: str, preferred: str = "") -> VacancyFeatures:
    return VacancyFeatures(id=vid, source="synthetic", title=vid,
                           required_skills=[ref(n) for n in required.split()],
                           preferred_skills=[ref(n) for n in preferred.split()])


def semantic(score: float | None) -> SemanticScore | None:
    if score is None:
        return None
    return SemanticScore(score=score, raw=score, item_scores=[score], best_item_index=0,
                         calibrated=True, warnings=[])


def ranked(cand: CandidateSkills, *specs, adjustments=None, scorer=None) -> RankingResult:
    """Rank (vacancy, semantic) pairs with the real scoring code, semantics given."""
    scorer = scorer or LinearScorer()
    adjustments = adjustments or {}
    visible, hidden = [], []
    for vac, sem in specs:
        adjustment = adjustments.get(vac.id, VacancyAdjustment())
        fit = score_vacancy(cand, vac, semantic(sem), adjustment, scorer)
        (hidden if adjustment.hidden else visible).append(fit)
    visible.sort(key=_sort_key)
    band_counts = dict.fromkeys(Band, 0)
    for fit in visible:
        band_counts[fit.band] += 1
    return RankingResult(results=visible, hidden=hidden, band_counts=band_counts,
                         hidden_count=len(hidden), scorer_name=scorer.name,
                         weights=scorer.weights, thresholds=dict(THRESHOLDS),
                         semantic_calibrated=True, ranking_version="1", warnings=[])


def names(refs) -> list[str]:
    return [r.raw for r in refs]


# --- the worked example from the plan ----------------------------------------------
# Weights 0.7 / 0.3, strong at 80% coverage with semantic >= 0.3. The candidate has
# A to E. v1 needs one skill, v2 two, v3 has low semantic, v4 needs three, v5 is strong.

CANDIDATE = candidate("A", "B", "C", "D", "E")
EXAMPLE = [
    (vacancy("v1", "A B C F G"), 0.5),
    (vacancy("v2", "A B C D E H I J"), 0.6),
    (vacancy("v3", "A B K L"), 0.2),
    (vacancy("v4", "A B C D E M N O P Q"), 0.5),
    (vacancy("v5", "A B"), 0.5),
]


@pytest.fixture
def example() -> RankingResult:
    return ranked(CANDIDATE, *EXAMPLE)


def test_the_example_ranks_as_the_plan_says(example) -> None:
    bands = {r.vacancy.id: r.band for r in example.results}
    assert bands == {"v1": B.PARTIAL, "v2": B.PARTIAL, "v3": B.PARTIAL, "v4": B.PARTIAL,
                     "v5": B.STRONG}
    assert SEM_MIN == 0.3  # the example's numbers assume the JM-24 constants


def test_within_reach_matches_the_hand_computed_values(example) -> None:
    items = within_reach(CANDIDATE, working_set(example))

    assert [i.vacancy_id for i in items] == ["v1", "v2"]
    v1, v2 = items
    assert names(v1.added) == ["F"]  # F and G both work; the vacancy's first wins
    assert v1.fit_now == pytest.approx(0.57)  # 0.7 x 3/5 + 0.3 x 0.5
    assert v1.fit_after == pytest.approx(0.71)  # 0.7 x 4/5 + 0.3 x 0.5
    assert (v1.required_matched_count, v1.required_count) == (3, 5)
    assert names(v2.added) == ["H", "I"]
    assert v2.fit_now == pytest.approx(0.6175)  # 0.7 x 5/8 + 0.3 x 0.6
    assert v2.fit_after == pytest.approx(0.7925)  # 0.7 x 7/8 + 0.3 x 0.6


def test_skills_worth_learning_matches_the_hand_computed_values(example) -> None:
    items = skills_worth_learning(CANDIDATE, working_set(example))

    # F and G tie on gain 1 and frequency 1, so the label decides. H to Q unlock nothing.
    assert [(i.skill.raw, i.gain, i.frequency, i.vacancy_ids) for i in items] == [
        ("F", 1, 1, ["v1"]),
        ("G", 1, 1, ["v1"]),
    ]


def test_in_demand_matches_the_hand_computed_values(example) -> None:
    items = in_demand(CANDIDATE, working_set(example))

    assert [(i.skill.raw, i.count) for i in items] == [
        ("A", 5), ("B", 5), ("C", 3), ("D", 2), ("E", 2)]


def test_compute_insights_puts_it_together(example) -> None:
    result = compute_insights(CANDIDATE, example)

    assert result.working_set_size == 5
    assert [i.vacancy_id for i in result.within_reach] == ["v1", "v2"]
    assert [i.skill.raw for i in result.skills_to_learn] == ["F", "G"]
    assert [i.skill.raw for i in result.in_demand] == ["A", "B", "C", "D", "E"]
    assert InsightsResult.model_validate_json(result.model_dump_json()) == result


# --- edge cases --------------------------------------------------------------------


def test_no_partial_vacancies_gives_nothing_within_reach() -> None:
    cand = candidate("A", "B")
    ranking = ranked(cand, (vacancy("strong", "A B"), 0.5), (vacancy("weak", "A X Y Z"), 0.5))

    result = compute_insights(cand, ranking)

    assert result.within_reach == []
    assert result.skills_to_learn == []  # one skill alone can't lift 1/4 to 80%
    assert [(i.skill.raw, i.count) for i in result.in_demand] == [("A", 2), ("B", 1)]


def test_a_skill_that_unlocks_nothing_is_left_out() -> None:
    cand = candidate("A", "B", "C")
    ranking = ranked(cand, (vacancy("v", "A B C X Y"), 0.5))  # 3/5: one skill gives 4/5

    learn = skills_worth_learning(cand, working_set(ranking))
    reach = within_reach(cand, working_set(ranking))

    assert [i.skill.raw for i in learn] == ["X", "Y"]
    assert names(reach[0].added) == ["X"]

    ranking = ranked(cand, (vacancy("v", "A B C W X Y Z"), 0.5))  # 3/7: needs 3
    assert skills_worth_learning(cand, working_set(ranking)) == []
    assert within_reach(cand, working_set(ranking)) == []


def test_equal_vacancies_sort_by_id() -> None:
    cand = candidate("A", "B", "C")
    ranking = ranked(cand, (vacancy("b", "A B C X"), 0.5), (vacancy("a", "A B C Y"), 0.5))

    items = within_reach(cand, working_set(ranking))

    assert [i.vacancy_id for i in items] == ["a", "b"]
    assert items[0].fit_now == items[1].fit_now


def test_fewer_skills_to_add_comes_before_a_higher_fit() -> None:
    cand = candidate("A", "B", "C", "D", "E")
    ranking = ranked(cand,
                     (vacancy("two", "A B C D E H I J"), 1.0),  # higher fit, needs two
                     (vacancy("one", "A B C F G"), 0.3))  # needs one

    items = within_reach(cand, working_set(ranking))

    assert items[0].fit_now < items[1].fit_now
    assert [i.vacancy_id for i in items] == ["one", "two"]


def test_skills_with_equal_gain_sort_by_frequency_then_label() -> None:
    cand = candidate("A", "B", "C")
    ranking = ranked(cand,
                     (vacancy("v1", "A B C Z"), 0.5),  # Z unlocks v1
                     (vacancy("v2", "A B C Y"), 0.5),  # Y unlocks v2
                     (vacancy("v3", "A Y W"), 0.5))  # Y also required here (weak)

    items = skills_worth_learning(cand, working_set(ranking))

    assert [(i.skill.raw, i.gain, i.frequency) for i in items] == [("Y", 1, 2), ("Z", 1, 1)]


def test_skills_that_unlock_more_come_first() -> None:
    cand = candidate("A", "B", "C")
    ranking = ranked(cand,
                     (vacancy("v1", "A B C K"), 0.5),
                     (vacancy("v2", "A B C K"), 0.5),
                     (vacancy("v3", "A B C J"), 0.5))

    items = skills_worth_learning(cand, working_set(ranking))

    assert [(i.skill.raw, i.gain, i.vacancy_ids) for i in items] == [
        ("K", 2, ["v1", "v2"]), ("J", 1, ["v3"])]


def test_the_semantic_score_stays_as_it_was() -> None:
    cand = candidate("A", "B", "C")
    low = SEM_MIN - 0.1
    ranking = ranked(cand, (vacancy("low", "A B C X"), low), (vacancy("ok", "A B C Y"), 0.5))

    result = compute_insights(cand, ranking)

    assert [i.vacancy_id for i in result.within_reach] == ["ok"]  # full coverage is not enough
    assert [i.skill.raw for i in result.skills_to_learn] == ["Y"]
    assert result.within_reach[0].fit_after == pytest.approx(0.7 * 1.0 + 0.3 * 0.5)


def test_a_missing_semantic_score_does_not_block_strong() -> None:
    cand = candidate("A", "B", "C")
    ranking = ranked(cand, (vacancy("v", "A B C X"), None))

    items = within_reach(cand, working_set(ranking))

    assert items[0].fit_after == pytest.approx(1.0)  # skills only when semantic is missing


def test_a_capped_vacancy_is_never_within_reach() -> None:
    cand = candidate("A", "B", "C", "D")
    adjustments = {"capped": VacancyAdjustment(band_cap=B.PARTIAL, reasons=["location"])}
    ranking = ranked(cand, (vacancy("capped", "A B C D X"), 0.5), adjustments=adjustments)
    assert (ranking.results[0].band, ranking.results[0].band_before_cap) == (B.PARTIAL,
                                                                            B.STRONG)

    # With the mapping, and with the cap read back from the result.
    assert within_reach(cand, working_set(ranking), adjustments=adjustments) == []
    assert compute_insights(cand, ranking).within_reach == []
    assert compute_insights(cand, ranking).skills_to_learn == []


def test_a_cap_given_only_in_the_mapping_is_respected() -> None:
    # The vacancy is partial on its own, so its result shows no cap; only the mapping knows.
    cand = candidate("A", "B", "C")
    adjustments = {"v": VacancyAdjustment(band_cap=B.PARTIAL)}
    ranking = ranked(cand, (vacancy("v", "A B C X"), 0.5), adjustments=adjustments)
    assert ranking.results[0].band is ranking.results[0].band_before_cap

    assert within_reach(cand, working_set(ranking), adjustments=adjustments) == []
    assert len(within_reach(cand, working_set(ranking))) == 1  # without it, no cap known


def test_hidden_and_unscored_vacancies_are_ignored() -> None:
    cand = candidate("A", "B", "C")
    adjustments = {"hidden": VacancyAdjustment(hidden=True)}
    ranking = ranked(cand,
                     (vacancy("hidden", "A B C X"), 0.5),
                     (VacancyFeatures(id="unscored", source="synthetic", title="u"), None),
                     (vacancy("seen", "A B C Y"), 0.5),
                     adjustments=adjustments)

    result = compute_insights(cand, ranking, adjustments=adjustments)

    assert result.working_set_size == 1
    assert [i.vacancy_id for i in result.within_reach] == ["seen"]
    assert [i.skill.raw for i in result.skills_to_learn] == ["Y"]


def test_a_vacancy_with_only_preferred_skills_has_nothing_to_add() -> None:
    cand = candidate("A")
    ranking = ranked(cand, (vacancy("v", "", preferred="A X"), 0.5))  # 1/1.5 -> partial

    assert ranking.results[0].band is B.PARTIAL
    assert compute_insights(cand, ranking).within_reach == []


def test_the_working_set_is_cut_at_top_n(monkeypatch) -> None:
    monkeypatch.setattr(insights, "TOP_N", 2)
    cand = candidate("A", "B", "C")
    ranking = ranked(cand,
                     (vacancy("a", "A B C"), 0.9),
                     (vacancy("b", "A B C"), 0.8),
                     (vacancy("c", "A B C X"), 0.5))  # third: outside the working set

    result = compute_insights(cand, ranking)

    assert result.working_set_size == 2
    assert result.within_reach == [] and result.skills_to_learn == []
    assert [(i.skill.raw, i.count) for i in result.in_demand][0] == ("A", 2)


def test_results_are_capped_at_their_limits() -> None:
    cand = candidate("A", "B", "C")
    specs = [(vacancy(f"v{i}", f"A B C X{i}"), 0.5) for i in range(10)]
    ranking = ranked(cand, *specs)

    result = compute_insights(cand, ranking)

    assert len(result.within_reach) == WITHIN_REACH_LIMIT
    assert len(result.skills_to_learn) == LEARN_LIMIT
    assert len(result.in_demand) == min(IN_DEMAND_LIMIT, 3)
    assert (TOP_N, MAX_ADDED) == (50, 2)


def test_no_candidate_skills_or_no_vacancies_is_not_an_error() -> None:
    empty = candidate()
    assert compute_insights(empty, ranked(empty, (vacancy("v", "A B"), 0.5))).in_demand == []
    result = compute_insights(CANDIDATE, ranked(CANDIDATE))
    assert result == InsightsResult(working_set_size=0)


def test_unrecognised_skills_count_by_their_text() -> None:
    raw = SkillRef(raw="Pipewright", normalized="pipewright")
    cand = candidate("A", "B", "C")
    vac = VacancyFeatures(id="v", source="synthetic", title="v",
                          required_skills=[ref("A"), ref("B"), ref("C"), raw])
    ranking = ranked(cand, (vac, 0.5))

    items = skills_worth_learning(cand, working_set(ranking))

    assert [(i.skill.key, i.skill.raw) for i in items] == [("raw:pipewright", "Pipewright")]


def test_the_candidate_is_not_changed(example) -> None:
    before = CANDIDATE.model_dump()

    compute_insights(CANDIDATE, example)

    assert CANDIDATE.model_dump() == before


def test_another_scorer_changes_fits_but_not_which_jobs() -> None:
    skills_only = LinearScorer(w_skill=1.0, w_semantic=0.0)
    default = compute_insights(CANDIDATE, ranked(CANDIDATE, *EXAMPLE))
    other = compute_insights(CANDIDATE, ranked(CANDIDATE, *EXAMPLE, scorer=skills_only),
                             scorer=skills_only)

    assert [i.vacancy_id for i in other.within_reach] == [i.vacancy_id
                                                          for i in default.within_reach]
    assert other.within_reach[0].fit_after == pytest.approx(0.8)  # 4/5, skills only
    assert other.skills_to_learn == default.skills_to_learn
