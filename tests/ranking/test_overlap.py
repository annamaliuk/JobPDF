import math

import pytest

from jobpdf.ranking.models import CandidateSkills, SkillRef, VacancyFeatures
from jobpdf.ranking.overlap import (
    DUPLICATE_REQUIREMENT,
    EMPTY_SKILL,
    NO_CANDIDATE_SKILLS,
    NO_VACANCY_SKILLS,
    SkillOverlap,
    default_weight,
    skill_overlap,
)

ESCO = "http://data.europa.eu/esco/skill/"
PYTHON = ESCO + "ccd0a1d9-afda-43d9-b901-96344886e14d"
SQL = ESCO + "598de5b0-5b58-4ea7-8058-a4bc4d18c742"
BIGQUERY = "jobpdf:skill/google-bigquery"
DOCKER = "jobpdf:skill/docker"
KUBERNETES = "jobpdf:skill/kubernetes"
AIRFLOW = "jobpdf:skill/apache-airflow"


def ref(key: str) -> SkillRef:
    """A concept ID, or "raw:<text>" for an unrecognised skill."""
    if key.startswith("raw:"):
        text = key.removeprefix("raw:")
        return SkillRef(raw=text.title(), normalized=text)
    return SkillRef(concept_id=key, raw=key.rsplit("/", 1)[-1], normalized=key.rsplit("/")[-1])


def vacancy(required=(), preferred=()) -> VacancyFeatures:
    return VacancyFeatures(
        id="synthetic-1", source="synthetic", title="Data Engineer",
        required_skills=[r if isinstance(r, SkillRef) else ref(r) for r in required],
        preferred_skills=[r if isinstance(r, SkillRef) else ref(r) for r in preferred],
    )


def candidate(*keys: str) -> CandidateSkills:
    return CandidateSkills().with_added(ref(k) for k in keys)


def keys(refs: list[SkillRef]) -> list[str]:
    return [r.key for r in refs]


def check(cand: CandidateSkills, vac: VacancyFeatures, **kwargs) -> SkillOverlap:
    """skill_overlap plus the invariants every result must satisfy."""
    result = skill_overlap(cand, vac, **kwargs)

    required, preferred = [], []
    for refs, out in ((vac.required_skills, required), (vac.preferred_skills, preferred)):
        for r in refs:
            if not r.is_empty and r.key not in out:
                out.append(r.key)
    preferred = [k for k in preferred if k not in required]
    for matched, missing, expected in (
        (result.required_matched, result.required_missing, required),
        (result.preferred_matched, result.preferred_missing, preferred),
    ):
        assert not set(keys(matched)) & set(keys(missing))
        assert sorted(keys(matched) + keys(missing), key=expected.index) == expected
        assert keys(matched) == [k for k in expected if k in keys(matched)]  # vacancy order
    assert result.required_matched_count == len(result.required_matched)
    assert result.required_count == len(required)
    assert result.preferred_matched_count == len(result.preferred_matched)
    assert result.preferred_count == len(preferred)
    assert result.score is None or 0.0 <= result.score <= 1.0
    assert skill_overlap(cand, vac, **kwargs) == result  # deterministic, order included
    return result


# --- hand-computed scores ----------------------------------------------------------


def test_full_match() -> None:
    result = check(candidate(PYTHON, SQL, DOCKER), vacancy([PYTHON, SQL], [DOCKER]))

    assert (result.score, result.required_coverage, result.warnings) == (1.0, 1.0, [])
    assert result.required_missing == [] and result.preferred_missing == []


def test_no_match() -> None:
    result = check(candidate(KUBERNETES), vacancy([PYTHON, SQL], [DOCKER]))

    assert (result.score, result.required_coverage) == (0.0, 0.0)
    assert keys(result.required_missing) == [PYTHON, SQL]


def test_preferred_only() -> None:
    result = check(candidate(DOCKER), vacancy(preferred=[DOCKER, KUBERNETES]))

    assert (result.score, result.required_coverage, result.warnings) == (0.5, None, [])


def test_two_of_three_required_and_one_of_two_preferred() -> None:
    result = check(candidate(PYTHON, BIGQUERY, DOCKER, "raw:excel"),
                   vacancy([PYTHON, SQL, BIGQUERY], [DOCKER, AIRFLOW]))

    assert result.score == (2 + 0.5) / (3 + 1) == 0.625
    assert result.required_coverage == pytest.approx(2 / 3)
    assert keys(result.required_matched) == [PYTHON, BIGQUERY]
    assert keys(result.required_missing) == [SQL]
    assert keys(result.preferred_matched) == [DOCKER]
    assert keys(result.preferred_missing) == [AIRFLOW]
    assert (result.required_matched_count, result.required_count,
            result.preferred_matched_count, result.preferred_count) == (2, 3, 1, 2)


# --- edge cases --------------------------------------------------------------------


def test_vacancy_without_skills() -> None:
    result = check(candidate(PYTHON), vacancy())

    assert (result.score, result.required_coverage) == (None, None)
    assert result.warnings == [NO_VACANCY_SKILLS]


def test_candidate_without_skills() -> None:
    result = check(CandidateSkills(), vacancy([PYTHON], [DOCKER]))

    assert (result.score, result.required_coverage) == (0.0, 0.0)
    assert result.warnings == [NO_CANDIDATE_SKILLS]


def test_candidate_without_skills_and_only_preferred() -> None:
    result = check(CandidateSkills(), vacancy(preferred=[DOCKER]))

    assert (result.score, result.required_coverage) == (0.0, None)
    assert result.warnings == [NO_CANDIDATE_SKILLS]


def test_neither_side_has_skills() -> None:
    result = check(CandidateSkills(), vacancy())

    assert result.score is None and result.warnings == [NO_VACANCY_SKILLS]


def test_same_key_required_and_preferred_counts_once_as_required() -> None:
    result = check(candidate(PYTHON), vacancy([PYTHON, SQL], [PYTHON, DOCKER]))

    assert keys(result.preferred_missing) == [DOCKER] and result.preferred_count == 1
    assert result.score == pytest.approx(1 / (2 + 0.5))
    assert result.warnings == [DUPLICATE_REQUIREMENT]


def test_duplicates_within_a_list_count_once_at_the_first_position() -> None:
    gbq = SkillRef(concept_id=BIGQUERY, raw="GBQ", normalized="gbq")
    bigquery = SkillRef(concept_id=BIGQUERY, raw="BigQuery", normalized="bigquery")

    result = check(candidate(BIGQUERY), vacancy([SQL, gbq, PYTHON, bigquery]))

    assert keys(result.required_missing) == [SQL, PYTHON]
    assert [r.raw for r in result.required_matched] == ["GBQ"]
    assert result.required_count == 3 and result.score == pytest.approx(1 / 3)
    assert result.warnings == []


def test_empty_vacancy_skill_is_ignored_with_a_warning() -> None:
    empty = SkillRef(raw="  ", normalized="")

    result = check(candidate(PYTHON), vacancy([PYTHON, empty], [empty]))

    assert result.required_count == 1 and result.score == 1.0
    assert result.warnings == [EMPTY_SKILL]


def test_unrecognised_skill_matches_through_its_raw_key() -> None:
    cand = CandidateSkills().with_added([SkillRef(raw="In-house ETL", normalized="in-house etl")])
    vac = vacancy([SkillRef(raw="in-house  ETL", normalized="in-house etl"), PYTHON])

    result = check(cand, vac)

    assert keys(result.required_matched) == ["raw:in-house etl"]
    assert keys(result.required_missing) == [PYTHON]
    assert result.score == 0.5


def test_warnings_come_in_a_fixed_order() -> None:
    empty = SkillRef(raw="", normalized="")

    result = check(CandidateSkills(), vacancy([PYTHON, empty], [PYTHON]))

    assert result.warnings == [EMPTY_SKILL, DUPLICATE_REQUIREMENT, NO_CANDIDATE_SKILLS]


# --- the weight hook ---------------------------------------------------------------


def test_custom_weight_changes_the_score() -> None:
    def rare_sql(key: str) -> float:
        return 2.0 if key == SQL else 1.0

    cand, vac = candidate(PYTHON, DOCKER), vacancy([PYTHON, SQL, BIGQUERY], [DOCKER, AIRFLOW])

    result = check(cand, vac, weight=rare_sql)

    # matched: PYTHON 1.0 + DOCKER 0.5; total: 1 + 2 + 1 + 0.5 + 0.5
    assert result.score == pytest.approx(1.5 / 5.0)
    assert result.required_coverage == pytest.approx(1 / 3)  # coverage is a plain count
    assert check(cand, vac).score == pytest.approx(1.5 / 4.0)
    assert default_weight(PYTHON) == 1.0


@pytest.mark.parametrize("bad", [0, 0.0, -1.0, math.nan, math.inf, "x", None, True])
def test_non_positive_or_non_numeric_weight_raises(bad) -> None:
    with pytest.raises(ValueError, match="weight"):
        skill_overlap(candidate(PYTHON), vacancy([PYTHON]), weight=lambda key: bad)


# --- with_added re-scoring (JM-42) -------------------------------------------------


def test_with_added_moves_exactly_one_skill_from_missing_to_matched() -> None:
    cand, vac = candidate(PYTHON), vacancy([PYTHON, SQL, BIGQUERY], [DOCKER])
    before = check(cand, vac)

    after = check(cand.with_added([ref(SQL)]), vac)

    assert keys(after.required_matched) == [PYTHON, SQL]
    assert keys(after.required_missing) == [BIGQUERY]
    assert after.preferred_matched == before.preferred_matched
    assert after.score == pytest.approx(2 / 3.5) and before.score == pytest.approx(1 / 3.5)
    assert cand.keys == {PYTHON}  # the original candidate is untouched
    assert check(cand, vac) == before
