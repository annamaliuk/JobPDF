from pathlib import Path

import pytest

from jobpdf.normalization.taxonomy import (
    Alias,
    load_esco,
    load_esco_with_stats,
    make_alias,
    strip_parentheticals,
)

ESCO = Path(__file__).parent / "fixtures" / "esco"
URI = "http://example.org/esco/skill/"


@pytest.fixture(scope="module")
def loaded() -> tuple:
    return load_esco_with_stats(ESCO)


def aliases_of(aliases: list[Alias], concept: str) -> dict[str, Alias]:
    return {a.alias_norm: a for a in aliases if a.canonical_id == URI + concept}


def test_only_released_individual_skills_become_concepts(loaded: tuple) -> None:
    concepts, _, stats = loaded
    ids = {c.canonical_id for c in concepts}

    assert URI + "fax" not in ids  # obsolete
    assert "http://example.org/esco/group/programming" not in ids  # skill group
    assert len(ids) == len(concepts) == 10
    assert stats.dropped_rows == 2


def test_duplicate_uri_keeps_latest_row(loaded: tuple) -> None:
    _, aliases, stats = loaded
    sql = aliases_of(aliases, "sql")

    assert stats.duplicate_rows == 1
    assert "outdated label" not in sql
    assert "subquery" in sql


def test_multi_line_alt_and_hidden_labels_are_split(loaded: tuple) -> None:
    _, aliases, _ = loaded
    sql = aliases_of(aliases, "sql")

    assert {"structured query language", "sequel", "subquery"} <= set(sql)
    assert sql["sequel"].alias_kind == "alt"
    assert sql["sql queries"].alias_kind == "hidden"
    assert sql["sql"].alias_kind == "preferred"


def test_concept_fields(loaded: tuple) -> None:
    concepts, _, _ = loaded
    by_id = {c.canonical_id: c for c in concepts}

    python = by_id[URI + "python"]
    assert python.name_en == "Python (computer programming)"
    assert python.name_uk == "Python (комп’ютерне програмування)"
    assert python.description and "two lines" in python.description
    assert (python.skill_type, python.reuse_level, python.source) == (
        "knowledge",
        "cross-sector",
        "esco",
    )
    assert by_id[URI + "ruby"].name_uk is None
    ict = by_id[URI + "ict-safety"]
    assert (ict.skill_type, ict.reuse_level) == (None, None)


def test_is_digital_from_collection(loaded: tuple) -> None:
    concepts, _, _ = loaded
    digital = {c.canonical_id for c in concepts if c.is_digital}

    assert digital == {URI + "python", URI + "sql"}


def test_ukrainian_aliases_join_to_the_right_concept(loaded: tuple) -> None:
    _, aliases, stats = loaded
    uk = {(a.canonical_id, a.alias) for a in aliases if a.lang == "uk"}

    assert (URI + "teamwork", "працювати в команді") in uk
    assert (URI + "python", "Python (комп’ютерне програмування)") in uk
    # "SQL" is identical in both files; dedup keeps the English one.
    assert aliases_of(aliases, "sql")["sql"].lang == "en"
    assert stats.unknown_uris == {"uk": 1}
    assert not any(a.canonical_id.endswith("not-in-english") for a in aliases)


def test_short_aliases_are_exact_only(loaded: tuple) -> None:
    _, aliases, _ = loaded
    r = aliases_of(aliases, "r")

    assert r["r"].exact_only
    assert not r["r language"].exact_only
    assert make_alias("x", "Go", "en", "custom").exact_only
    assert not make_alias("x", "SQL", "en", "custom").exact_only


def test_aliases_are_unique_per_concept(loaded: tuple) -> None:
    _, aliases, _ = loaded
    keys = [(a.canonical_id, a.alias_norm) for a in aliases]

    assert len(keys) == len(set(keys))


def test_load_esco_requires_base_language() -> None:
    with pytest.raises(ValueError):
        load_esco(ESCO, languages=("uk",))


def test_stripping_adds_aliases_and_skips_collisions(loaded: tuple) -> None:
    _, aliases, _ = loaded
    stripped, collisions = strip_parentheticals(aliases)

    ruby = aliases_of(stripped, "ruby")
    assert ruby["ruby"].alias_kind == "stripped"
    # Python already has "Python" as an ESCO alt label: not re-added as stripped.
    assert aliases_of(stripped, "python")["python"].alias_kind == "alt"
    # Ukrainian label strips too.
    assert "python" in aliases_of(stripped, "python")

    by_norm = {c.alias_norm: c.canonical_ids for c in collisions}
    # Two concepts strip to "go"; "spark" already belongs to SPARK (Ada).
    assert by_norm["go"] == (URI + "go-game", URI + "go-lang")
    assert by_norm["spark"] == (URI + "spark-ada", URI + "spark-data")
    assert "go" not in aliases_of(stripped, "go-lang")
    assert "spark" not in aliases_of(stripped, "spark-data")
    assert "spark" in aliases_of(stripped, "spark-ada")


def test_stripping_is_order_independent(loaded: tuple) -> None:
    _, aliases, _ = loaded

    forward, fc = strip_parentheticals(aliases)
    backward, bc = strip_parentheticals(list(reversed(aliases)))

    assert forward == backward
    assert fc == bc
