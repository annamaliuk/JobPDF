"""The Blueprint requirement against the real built index (skipped without it)."""

from collections.abc import Iterator

import psycopg
import pytest

from jobpdf.normalization.db import database_url
from jobpdf.normalization.index import SkillIndex
from jobpdf.normalization.match_cache import MatchTier
from jobpdf.normalization.matcher import SkillMatcher


@pytest.fixture
def built_index() -> Iterator[SkillIndex]:
    url = database_url()
    if not url:
        pytest.skip("DATABASE_URL not set (environment or .env)")
    try:
        index = SkillIndex.connect(url)
    except psycopg.OperationalError as exc:
        pytest.skip(f"database not reachable: {type(exc).__name__}")
    if "build" not in index.meta:
        index.close()
        pytest.skip("index not built; run scripts/build_index.py")
    with index:
        yield index


@pytest.mark.db
def test_bigquery_spellings_resolve_to_one_concept_without_the_llm(
    built_index: SkillIndex,
) -> None:
    matches = SkillMatcher(built_index, tool_caller=None).match_many(
        ["GBQ", "BigQuery", "Google BigQuery"]
    )

    assert all(m.tier in (MatchTier.EXACT, MatchTier.VECTOR) for m in matches), [
        (m.tier, m.warnings) for m in matches
    ]
    assert len({m.concept_id for m in matches}) == 1
    assert matches[0].concept_id is not None
