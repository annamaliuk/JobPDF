"""The built service against the real index and model (skipped without them)."""

from collections.abc import Iterator

import pytest

from jobpdf.normalization.db import database_url
from jobpdf.normalization.embedder import MODEL_NAME, MODEL_REVISION
from jobpdf.normalization.match_cache import MatchTier
from jobpdf.normalization.service import (
    LLM_DISABLED,
    NormalizationService,
    build_normalization_service,
)


def model_is_cached() -> bool:
    # Checked up front so a test never downloads the model.
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(MODEL_NAME, "config.json", revision=MODEL_REVISION)
    return isinstance(path, str)


@pytest.fixture
def built_service(monkeypatch: pytest.MonkeyPatch) -> Iterator[NormalizationService]:
    if not database_url():
        pytest.skip("DATABASE_URL not set (environment or .env)")
    if not model_is_cached():
        pytest.skip("e5 model not in the local Hugging Face cache; run scripts/build_index.py")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # exact and vector tiers only
    service = build_normalization_service()
    try:
        if service.status()["index"] != "ok":
            pytest.skip("database not reachable")
        if service.versions.taxonomy is None:
            pytest.skip("index not built; run scripts/build_index.py")
        yield service
    finally:
        service.close()


@pytest.mark.db
@pytest.mark.model
def test_bigquery_spellings_resolve_to_one_concept_without_the_llm(
    built_service: NormalizationService,
) -> None:
    matches = built_service.normalize_skills(["GBQ", "BigQuery", "Google BigQuery"])

    assert built_service.status()["llm"] == LLM_DISABLED
    assert all(m.tier in (MatchTier.EXACT, MatchTier.VECTOR) for m in matches)
    assert len(NormalizationService.concept_ids(matches)) == 1


@pytest.mark.db
@pytest.mark.model
def test_skill_groups_on_a_vacancy_shaped_dict(built_service: NormalizationService) -> None:
    vacancy = {"required": ["GBQ", "Python"], "nice_to_have": [], "extra": ["Docker", ""]}

    result = built_service.normalize_skill_groups(vacancy, context="vacancy skills")

    assert list(result) == ["required", "nice_to_have", "extra"]
    assert [[m.raw for m in matches] for matches in result.values()] == [
        ["GBQ", "Python"], [], ["Docker", ""]]
    assert result["required"][0].concept_id == "jobpdf:skill/google-bigquery"
    assert result["extra"][1].tier is MatchTier.UNMATCHED
