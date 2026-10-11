from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from jobpdf.normalization.match_cache import MatchTier, SkillMatch
from jobpdf.ranking.models import (
    RANKING_VERSION,
    CandidateSkills,
    EducationRequirement,
    SkillRef,
    VacancyFeatures,
)

PYTHON = "http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d"
BIGQUERY = "jobpdf:skill/google-bigquery"
DOCKER = "jobpdf:skill/docker"


@dataclass(frozen=True)
class Match:
    """Any object with the SkillLike attributes."""

    raw: str
    normalized: str
    concept_id: str | None = None
    preferred_label: str | None = None


def test_ranking_version() -> None:
    assert RANKING_VERSION == "1"


def test_skill_ref_key_uses_the_concept_id_else_the_normalized_text() -> None:
    known = SkillRef(concept_id=BIGQUERY, raw="GBQ", normalized="gbq", label="Google BigQuery")
    unknown = SkillRef(raw="Pyspark-ish", normalized="pyspark-ish")

    assert known.key == BIGQUERY
    assert unknown.key == "raw:pyspark-ish"
    assert known.model_dump()["key"] == BIGQUERY  # visible to API and storage
    assert SkillRef(raw="  ", normalized="").is_empty


def test_skill_ref_json_round_trip_despite_the_computed_key() -> None:
    ref = SkillRef(concept_id=BIGQUERY, raw="GBQ", normalized="gbq", label="Google BigQuery")

    assert SkillRef.model_validate_json(ref.model_dump_json()) == ref


def test_from_match_reads_preferred_label() -> None:
    ref = SkillRef.from_match(Match(raw="GBQ", normalized="gbq", concept_id=BIGQUERY,
                                    preferred_label="Google BigQuery"))

    assert (ref.key, ref.raw, ref.normalized, ref.label) == (
        BIGQUERY, "GBQ", "gbq", "Google BigQuery")


def test_from_match_accepts_a_real_skill_match() -> None:
    match = SkillMatch(raw="Docker", normalized="docker", concept_id=DOCKER,
                       preferred_label="Docker", tier=MatchTier.EXACT)

    assert SkillRef.from_match(match).key == DOCKER


def test_from_skill_matches_merges_paths_and_keeps_unmatched_as_raw_keys() -> None:
    matches = {
        "skills[0]": Match("GBQ", "gbq", BIGQUERY, "Google BigQuery"),
        "skills[1]": Match("Pyspark-ish", "pyspark-ish"),
        "skills[2]": Match("", ""),
        "experience[1].skills_used[0]": Match("gbq", "gbq", BIGQUERY, "Google BigQuery"),
    }

    skills = CandidateSkills.from_skill_matches(matches)

    assert skills.keys == {BIGQUERY, "raw:pyspark-ish"}
    assert list(skills.refs) == [BIGQUERY, "raw:pyspark-ish"]
    assert skills.refs[BIGQUERY].raw == "GBQ"  # first occurrence is the display ref
    assert skills.sources == {
        BIGQUERY: ["skills[0]", "experience[1].skills_used[0]"],
        "raw:pyspark-ish": ["skills[1]"],
    }


def test_with_added_returns_a_new_instance_and_never_mutates() -> None:
    skills = CandidateSkills.from_skill_matches({"skills[0]": Match("GBQ", "gbq", BIGQUERY)})
    docker = SkillRef(concept_id=DOCKER, raw="Docker", normalized="docker")
    gbq = SkillRef(concept_id=BIGQUERY, raw="BigQuery", normalized="bigquery")

    grown = skills.with_added([docker, gbq, SkillRef(raw="", normalized="")])

    assert grown.keys == {BIGQUERY, DOCKER}
    assert grown.sources == {BIGQUERY: ["skills[0]", "added"], DOCKER: ["added"]}
    assert grown.refs[BIGQUERY].raw == "GBQ"
    assert skills.keys == {BIGQUERY}
    assert skills.sources == {BIGQUERY: ["skills[0]"]}


def test_models_are_frozen() -> None:
    ref = SkillRef(raw="Docker", normalized="docker")
    with pytest.raises(ValidationError):
        ref.raw = "Podman"
    with pytest.raises(ValidationError):
        CandidateSkills().refs = {}


def test_vacancy_with_only_required_fields_has_correct_defaults() -> None:
    vacancy = VacancyFeatures(id="remotive-123", source="remotive", title="Data Engineer")

    assert vacancy.model_dump(exclude={"id", "source", "title"}) == {
        "company": None, "url": None, "posted_at": None, "location_raw": None,
        "required_skills": [], "preferred_skills": [], "responsibilities": [],
        "min_years_experience": None, "education_requirement": None, "languages": None,
        "embedding": None, "embedding_version": None, "vacancy_text_version": None,
        "taxonomy_version": None, "index_version": None, "matcher_prompt_version": None,
    }


def test_vacancy_requires_id_source_and_title() -> None:
    with pytest.raises(ValidationError):
        VacancyFeatures(id="1", source="synthetic")


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="nice_to_have_skills"):
        VacancyFeatures(id="1", source="synthetic", title="X", nice_to_have_skills=[])
    with pytest.raises(ValidationError):
        SkillRef(raw="x", normalized="x", name="x")
    with pytest.raises(ValidationError):
        EducationRequirement(degree="BSc")


def test_full_vacancy_round_trips_through_json() -> None:
    vacancy = VacancyFeatures(
        id="remotive-123", source="remotive", title="Data Engineer", company="Acme Widgets",
        url="https://example.com/jobs/123",
        posted_at=datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc),
        location_raw="Europe only",
        required_skills=[SkillRef(concept_id=PYTHON, raw="Python", normalized="python")],
        preferred_skills=[SkillRef(raw="Pyspark-ish", normalized="pyspark-ish")],
        responsibilities=["Build pipelines"], min_years_experience=3,
        education_requirement=EducationRequirement(level="bachelor", field="CS",
                                                   quote="BSc in CS"),
        languages=["English"], embedding=[0.1, 0.2], embedding_version="e5-small@614241f",
        vacancy_text_version="1", taxonomy_version="v1.2.1", index_version="0123abcd",
        matcher_prompt_version="1",
    )

    assert VacancyFeatures.model_validate_json(vacancy.model_dump_json()) == vacancy
