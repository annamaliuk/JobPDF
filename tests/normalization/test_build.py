import csv
import json
from pathlib import Path

import pytest

from jobpdf.normalization.build import build_taxonomy, find_ambiguous
from jobpdf.normalization.taxonomy import make_alias

ESCO = Path(__file__).parent / "fixtures" / "esco"
URI = "http://example.org/esco/skill/"
CUSTOM = (
    "canonical_id,canonical_name,alias,lang\n"
    f"{URI}sql,SQL,postgres sql,en\n"
    "jobpdf:skill/apache-spark,Apache Spark,spark,en\n"
    "jobpdf:skill/apache-spark,,pyspark,en\n"
)


@pytest.fixture
def built(tmp_path: Path) -> tuple:
    custom = tmp_path / "custom_skills.csv"
    custom.write_text(CUSTOM, encoding="utf-8")
    out = tmp_path / "taxonomy"
    return build_taxonomy(ESCO, custom, out), out


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_files_written_and_match_result(built: tuple) -> None:
    result, out = built
    concepts, aliases = read(out / "concepts.csv"), read(out / "aliases.csv")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    assert len(concepts) == len(result.concepts) == 11  # 10 ESCO + 1 custom
    assert len(aliases) == len(result.aliases)
    assert manifest == result.manifest
    assert list(aliases[0]) == [
        "canonical_id", "alias", "alias_norm", "lang", "alias_kind", "exact_only"
    ]
    python = next(c for c in concepts if c["canonical_id"] == URI + "python")
    assert python["is_digital"] == "true"
    assert "\n" in python["description"]  # multi-line cell round-trips
    ict = next(c for c in concepts if c["canonical_id"] == URI + "ict-safety")
    assert ict["skill_type"] == ""


def test_manifest_counts(built: tuple) -> None:
    result, _ = built
    counts = result.manifest["counts"]

    assert counts["concepts"] == {"custom": 1, "esco": 10}
    assert counts["aliases_total"] == len(result.aliases)
    assert sum(counts["aliases_by_lang"].values()) == len(result.aliases)
    # "postgres sql", plus "apache spark" (the name), "spark" and "pyspark"
    assert counts["aliases_by_kind"]["custom"] == 4
    assert counts["aliases_by_kind"]["stripped"] >= 1  # "ruby"
    assert counts["exact_only"] == sum(a.exact_only for a in result.aliases)
    rows = result.manifest["esco_rows"]
    assert (rows["duplicate_rows"], rows["dropped_rows"]) == (1, 2)
    files = result.manifest["inputs"]["files"]
    assert set(files) == {
        "skills_en.csv", "skills_uk.csv", "digitalSkillsCollection_en.csv", "custom_skills.csv"
    }
    assert all(len(sha) == 64 for sha in files.values())


def test_ambiguity_and_collisions_reported(built: tuple) -> None:
    manifest = built[0].manifest
    examples = manifest["ambiguous_aliases"]["examples"]
    ambiguous = {e["alias_norm"]: e["canonical_ids"] for e in examples}

    assert ambiguous["spark"] == [URI + "spark-ada", "jobpdf:skill/apache-spark"]
    assert manifest["ambiguous_aliases"]["count"] == len(ambiguous)
    skipped = manifest["stripped_collisions_skipped"]["examples"]
    assert {e["alias_norm"] for e in skipped} == {"go", "spark"}


def test_find_ambiguous() -> None:
    aliases = [
        make_alias("a", "Spark", "en", "alt"),
        make_alias("b", "spark ", "en", "custom"),
        make_alias("a", "SPARK Ada", "en", "alt"),
    ]

    assert find_ambiguous(aliases) == {"spark": ["a", "b"]}
