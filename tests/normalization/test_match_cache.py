from jobpdf.normalization.match_cache import (
    InMemorySkillMatchCache,
    MatchCandidate,
    MatcherVersions,
    MatchTier,
    SkillMatch,
    cache_key,
    index_version,
)

META = {
    "model": {"name": "intfloat/multilingual-e5-small", "revision": "abc"},
    "build": {"built_at": "2026-10-01T10:00:00+00:00"},
    "taxonomy": {"manifest_sha256": "f00", "esco_version": "v1.2.1"},
    "thresholds": {"auto_accept_distance": 0.12},
}


def match() -> SkillMatch:
    return SkillMatch(
        raw="GBQ", normalized="gbq", concept_id="jobpdf:skill/google-bigquery",
        preferred_label="Google BigQuery", tier=MatchTier.EXACT, distance=0.0,
        candidates=[MatchCandidate(concept_id="jobpdf:skill/google-bigquery",
                                   preferred_label="Google BigQuery", matched_alias="gbq",
                                   distance=0.0)],
    )


def test_cache_key_puts_versions_before_the_normalized_skill() -> None:
    assert cache_key("c++", MatcherVersions(index="abc", prompt="1")) == "abc|1|c++"


def test_index_version_changes_with_any_versioned_meta_key() -> None:
    base = index_version(META)

    assert index_version(dict(META)) == base
    assert index_version({**META, "thresholds": {"auto_accept_distance": 0.13}}) != base
    assert index_version({**META, "build": {"built_at": "2026-10-02T10:00:00+00:00"}}) != base
    assert index_version({k: v for k, v in META.items() if k != "thresholds"}) != base
    assert index_version({**META, "unrelated": 1}) == base


def test_in_memory_cache_round_trip_returns_copies() -> None:
    cache = InMemorySkillMatchCache()
    cache.set("k", match())

    first = cache.get("k")
    first.warnings.append("mutated")

    assert cache.get("k") == match()
    assert cache.get("missing") is None
    assert len(cache) == 1


def test_skill_match_survives_json_round_trip() -> None:
    # JM-19 stores matches as JSON in Postgres.
    assert SkillMatch.model_validate_json(match().model_dump_json()) == match()
