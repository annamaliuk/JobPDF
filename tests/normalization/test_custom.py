import csv
from pathlib import Path

import pytest

from jobpdf.normalization.custom import CustomSkillsError, load_custom, merge_custom
from jobpdf.normalization.taxonomy import Concept, load_esco

ESCO = Path(__file__).parent / "fixtures" / "esco"
URI = "http://example.org/esco/skill/"
HEADER = "canonical_id,canonical_name,alias,lang\n"


@pytest.fixture(scope="module")
def esco() -> tuple:
    return load_esco(ESCO)


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "custom_skills.csv"
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def load(tmp_path: Path, body: str, esco: tuple) -> tuple:
    concepts, _ = esco
    return load_custom(write(tmp_path, body), {c.canonical_id: c for c in concepts})


def errors_for(tmp_path: Path, body: str, esco: tuple) -> list[str]:
    with pytest.raises(CustomSkillsError) as exc:
        load(tmp_path, body, esco)
    return exc.value.errors


def test_alias_attaches_to_esco_concept(tmp_path: Path, esco: tuple) -> None:
    concepts, aliases = load(tmp_path, f"{URI}sql,SQL,Postgres SQL,en\n", esco)

    assert concepts == []
    assert [(a.canonical_id, a.alias_norm, a.alias_kind) for a in aliases] == [
        (URI + "sql", "postgres sql", "custom")
    ]


def test_new_concept_with_several_aliases(tmp_path: Path, esco: tuple) -> None:
    body = "jobpdf:skill/kubernetes,Kubernetes,k8s,en\njobpdf:skill/kubernetes,,kube,en\n"
    concepts, aliases = load(tmp_path, body, esco)

    (kube,) = concepts
    assert (kube.name_en, kube.source, kube.is_digital) == ("Kubernetes", "custom", True)
    assert {a.alias_norm for a in aliases} == {"kubernetes", "k8s", "kube"}
    assert next(a for a in aliases if a.alias_norm == "k8s").exact_only is False


def test_merge_keeps_clash_with_esco_as_ambiguity(tmp_path: Path, esco: tuple) -> None:
    esco_concepts, esco_aliases = esco
    custom = load(tmp_path, "jobpdf:skill/apache-spark,Apache Spark,spark,en\n", esco)

    concepts, aliases = merge_custom(esco_concepts, esco_aliases, *custom)

    spark_owners = {a.canonical_id for a in aliases if a.alias_norm == "spark"}
    assert spark_owners == {URI + "spark-ada", "jobpdf:skill/apache-spark"}
    assert len(concepts) == len(esco_concepts) + 1


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (f"{URI}does-not-exist,,foo,en\n", "unknown ESCO URI"),
        ("jobpdf:skill/airflow,,airflow,en\n", "must have canonical_name"),
        ("jobpdf:skill/Apache_Airflow,Apache Airflow,airflow,en\n", "malformed slug"),
        ("jobpdf:skill/airflow-,Apache Airflow,airflow,en\n", "malformed slug"),
        ("airflow,Apache Airflow,airflow,en\n", "must be an ESCO URI"),
        (f"{URI}sql,,structured query,de\n", "lang must be"),
        (f"{URI}sql,,,en\n", "alias is required"),
        (f"{URI}sql,MySQL,mysql,en\n", "does not match ESCO label"),
        (
            "jobpdf:skill/airflow,Apache Airflow,airflow,en\n"
            "jobpdf:skill/airflow,Airflow 2,af,en\n",
            "conflicts with",
        ),
        (
            "jobpdf:skill/airflow,Apache Airflow,airflow,en\n"
            "jobpdf:skill/dbt,dbt,Airflow ,en\n",
            "points to several concepts",
        ),
    ],
)
def test_validation_errors(tmp_path: Path, esco: tuple, body: str, message: str) -> None:
    errors = errors_for(tmp_path, body, esco)

    assert any(message in e for e in errors), errors


def test_all_errors_reported_with_line_numbers(tmp_path: Path, esco: tuple) -> None:
    body = f"{URI}nope,,a,en\njobpdf:skill/ok,OK,ok,en\njobpdf:skill/BAD,Bad,b,en\n"
    errors = errors_for(tmp_path, body, esco)

    assert [e.split(":")[0] for e in errors] == ["line 2", "line 4"]


def test_missing_columns(tmp_path: Path, esco: tuple) -> None:
    path = tmp_path / "custom_skills.csv"
    path.write_text("canonical_id,alias\njobpdf:skill/x,x\n", encoding="utf-8")
    concepts, _ = esco

    with pytest.raises(CustomSkillsError, match="missing columns"):
        load_custom(path, {c.canonical_id: c for c in concepts})


def test_shipped_file_has_no_conflicts() -> None:
    """The committed custom_skills.csv loads without the ESCO export. ESCO concepts are
    stubbed with the name the file gives them, so ESCO labels are only checked by
    scripts/build_taxonomy.py; name clashes and aliases shared by two concepts fail here."""
    path = Path(__file__).parents[2] / "data" / "taxonomy" / "custom_skills.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    names: dict[str, str] = {}
    for row in rows:
        cid, name = row["canonical_id"], row["canonical_name"].strip()
        if not cid.startswith("jobpdf:") and not names.get(cid):
            names[cid] = name
    stubs = {
        cid: Concept(canonical_id=cid, name_en=name or cid, name_uk=None, description=None,
                     skill_type=None, reuse_level=None, is_digital=True, source="esco")
        for cid, name in names.items()
    }

    concepts, aliases = load_custom(path, stubs)

    assert concepts and aliases
