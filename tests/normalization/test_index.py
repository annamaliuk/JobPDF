import csv
import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from jobpdf.normalization import index_build
from jobpdf.normalization.db import apply_schema, connect, database_url
from jobpdf.normalization.index import (
    TAXONOMY_CHANGED,
    Candidate,
    SkillIndex,
    dedupe_candidates,
    manifest_warning,
)
from jobpdf.normalization.index_build import (
    build_index,
    create_table_sql,
    file_sha256,
    read_rows,
    standard_meta,
)
from jobpdf.normalization.index_eval import (
    Outcome,
    distance_percentiles,
    recall_at_k,
    select_threshold,
)
from tests.normalization.fakes import FakeEmbedder

# --- pure tests ----------------------------------------------------------------


def cand(cid: str, distance: float, alias: str = "a") -> Candidate:
    return Candidate(cid, cid.upper(), alias, distance)


def test_dedupe_keeps_best_distance_per_concept_sorted() -> None:
    ranked = dedupe_candidates(
        [cand("b", 0.30, "b1"), cand("a", 0.20, "a1"), cand("b", 0.10, "b2"), cand("c", 0.25)],
        k=2,
    )

    assert [(c.canonical_id, c.matched_alias, c.distance) for c in ranked] == [
        ("b", "b2", 0.10),
        ("a", "a1", 0.20),
    ]


def test_manifest_warning(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"esco_version": "v1.2.1"}')
    meta = {"taxonomy": {"manifest_sha256": file_sha256(manifest)}}

    assert manifest_warning(meta, manifest) is None
    manifest.write_text('{"esco_version": "v1.2.2"}')
    assert manifest_warning(meta, manifest) == TAXONOMY_CHANGED
    warning = manifest_warning(meta, tmp_path / "missing.json")
    assert warning and "not found" in warning


def outcome(distance: float, correct: bool) -> Outcome:
    return Outcome("x", ("x",) if correct else ("y", "x"), distance)


def test_select_threshold_takes_largest_qualifying_distance() -> None:
    outcomes = [outcome(0.01 * i, True) for i in range(1, 101)]  # 100 correct, 0.01..1.00
    outcomes += [outcome(0.705, False), outcome(0.905, False), outcome(0.915, False)]

    threshold = select_threshold(outcomes, min_precision=0.98, min_n=50)

    # Up to 0.90: 90 correct + 1 wrong = 0.989; adding 0.905 -> 90/92 = 0.978 fails.
    assert threshold is not None
    assert threshold.auto_accept_distance == 0.9
    assert threshold.n == 91
    assert threshold.precision == round(90 / 91, 4)
    assert threshold.coverage == round(91 / 103, 4)


def test_select_threshold_none_when_nothing_qualifies() -> None:
    few = [outcome(0.1, True)] * 10
    noisy = [outcome(0.01 * i, i % 2 == 0) for i in range(1, 200)]

    assert select_threshold(few, min_n=50) is None
    assert select_threshold(noisy) is None


def test_select_threshold_never_splits_ties() -> None:
    outcomes = [outcome(0.1, True)] * 60 + [outcome(0.2, True)] * 10 + [outcome(0.2, False)] * 5

    threshold = select_threshold(outcomes, min_precision=0.98, min_n=50)

    assert threshold is not None and threshold.auto_accept_distance == 0.1


def test_recall_and_percentiles() -> None:
    outcomes = [
        Outcome("a", ("a", "b"), 0.1),
        Outcome("a", ("b", "a"), 0.3),
        Outcome("a", ("b", "c", "d", "e", "f", "a"), 0.5),
        Outcome("a", (), None),
    ]

    assert recall_at_k(outcomes, 1) == 0.25
    assert recall_at_k(outcomes, 5) == 0.5
    percentiles = distance_percentiles(outcomes)
    assert percentiles["correct"] == {"p10": 0.1, "p50": 0.1, "p90": 0.1}
    assert percentiles["incorrect"]["p50"] == 0.4


# --- database tests --------------------------------------------------------------

AWS = "jobpdf:skill/amazon-web-services"
ALIASES = [
    # canonical_id, alias, lang, alias_kind, exact_only
    ("esco:python", "Python (computer programming)", "en", "preferred", False),
    ("esco:python", "Python", "en", "alt", False),
    ("esco:python", "Python (комп’ютерне програмування)", "uk", "preferred", False),
    ("esco:sql", "SQL", "en", "preferred", False),
    ("esco:sql", "Structured Query Language", "en", "alt", False),
    ("esco:r", "R", "en", "preferred", True),
    ("esco:go", "Go", "en", "custom", True),
    ("esco:cloud", "cloud technologies", "en", "preferred", False),
    ("esco:cloud", "aws", "en", "alt", False),
    (AWS, "Amazon Web Services", "en", "custom", False),
    (AWS, "aws", "en", "custom", False),
    ("esco:teamwork", "work in teams", "en", "preferred", False),
    ("esco:teamwork", "teamwork", "en", "alt", False),
    ("esco:teamwork", "працювати в команді", "uk", "preferred", False),
    ("esco:docker", "Docker", "en", "custom", False),
]
NAMES = {
    "esco:python": ("Python (computer programming)", "Python (комп’ютерне програмування)"),
    "esco:sql": ("SQL", ""),
    "esco:r": ("R", ""),
    "esco:go": ("Go", ""),
    "esco:cloud": ("cloud technologies", ""),
    AWS: ("Amazon Web Services", ""),
    "esco:teamwork": ("work in teams", "працювати в команді"),
    "esco:docker": ("", "Докер"),  # no English name: falls back to Ukrainian
}


@pytest.fixture
def taxonomy_dir(tmp_path: Path) -> Path:
    with (tmp_path / "aliases.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["canonical_id", "alias", "alias_norm", "lang", "alias_kind", "exact_only"])
        for cid, alias, lang, kind, exact in ALIASES:
            norm = " ".join(alias.lower().split())
            writer.writerow([cid, alias, norm, lang, kind, "true" if exact else "false"])
    with (tmp_path / "concepts.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["canonical_id", "name_en", "name_uk"])
        for cid, (en, uk) in NAMES.items():
            writer.writerow([cid, en, uk])
    (tmp_path / "manifest.json").write_text(json.dumps({"esco_version": "v1.2.1"}))
    return tmp_path


@pytest.fixture
def db_url() -> str:
    url = database_url()
    if not url:
        pytest.skip("DATABASE_URL not set (environment or .env)")
    return url


@pytest.fixture
def schema(db_url: str) -> Iterator[str]:
    """A throwaway schema per test; the vector extension itself lives in public."""
    name = f"test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(db_url, autocommit=True, connect_timeout=10) as admin:
        admin.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        admin.execute(f'CREATE SCHEMA "{name}"')
    try:
        yield name
    finally:
        with psycopg.connect(db_url, autocommit=True, connect_timeout=10) as admin:
            admin.execute(f'DROP SCHEMA "{name}" CASCADE')


def options(schema: str) -> str:
    return f"-c search_path={schema},public"


def build(db_url: str, schema: str, taxonomy_dir: Path) -> index_build.BuildSummary:
    meta = standard_meta(model_name="fake", revision="0", prefix="query: ",
                         languages=("en", "uk"), taxonomy_dir=taxonomy_dir)
    with connect(db_url, options=options(schema)) as conn:
        return build_index(conn, read_rows(taxonomy_dir), FakeEmbedder(), meta=meta,
                           progress=lambda _: None)


def open_index(db_url: str, schema: str, taxonomy_dir: Path) -> SkillIndex:
    return SkillIndex.connect(db_url, FakeEmbedder(), options=options(schema),
                              manifest_path=taxonomy_dir / "manifest.json")


def tables(db_url: str, schema: str) -> set[str]:
    with psycopg.connect(db_url, connect_timeout=10) as conn:
        rows = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = %s", [schema]
        ).fetchall()
    return {r[0] for r in rows}


@pytest.mark.db
def test_build_stores_rows_and_nulls_for_exact_only(
    db_url: str, schema: str, taxonomy_dir: Path
) -> None:
    summary = build(db_url, schema, taxonomy_dir)

    assert (summary.rows_total, summary.rows_embedded) == (15, 13)
    with connect(db_url, options=options(schema)) as conn:
        nulls = conn.execute(
            "SELECT alias FROM taxonomy_alias WHERE embedding IS NULL ORDER BY alias"
        ).fetchall()
        names = dict(conn.execute(
            "SELECT DISTINCT canonical_id, canonical_name FROM taxonomy_alias").fetchall())
        meta = dict(conn.execute("SELECT key, value FROM index_meta").fetchall())
    assert [r[0] for r in nulls] == ["Go", "R"]
    assert names["esco:docker"] == "Докер"
    assert meta["build"]["rows_embedded"] == 13
    assert meta["model"]["prefix"] == "query: "
    assert meta["taxonomy"]["manifest_sha256"] == file_sha256(taxonomy_dir / "manifest.json")


@pytest.mark.db
def test_rebuild_swaps_cleanly(db_url: str, schema: str, taxonomy_dir: Path) -> None:
    build(db_url, schema, taxonomy_dir)
    with connect(db_url, options=options(schema)) as conn:
        conn.execute("INSERT INTO index_meta VALUES ('thresholds', '{\"x\": 1}')")
        conn.commit()

    build(db_url, schema, taxonomy_dir)

    assert tables(db_url, schema) == {"taxonomy_alias", "index_meta"}
    with connect(db_url, options=options(schema)) as conn:
        assert conn.execute("SELECT count(*) FROM taxonomy_alias").fetchone()[0] == 15
        indexes = {r[0] for r in conn.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = %s", [schema]).fetchall()}
        assert {"taxonomy_alias_norm_idx", "taxonomy_alias_hnsw_idx",
                "taxonomy_alias_pkey"} <= indexes
        assert not any(name.endswith(("_new_norm_idx", "_old_hnsw_idx")) for name in indexes)
        keys = {r[0] for r in conn.execute("SELECT key FROM index_meta").fetchall()}
        assert "thresholds" not in keys  # stale threshold dropped with the old vectors


@pytest.mark.db
def test_failed_build_leaves_previous_index_intact(
    db_url: str, schema: str, taxonomy_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build(db_url, schema, taxonomy_dir)

    def explode(conn: psycopg.Connection, table: str) -> None:
        raise RuntimeError("interrupted after COPY")

    monkeypatch.setattr(index_build, "_create_indexes", explode)
    with pytest.raises(RuntimeError, match="interrupted"):
        build(db_url, schema, taxonomy_dir)

    assert tables(db_url, schema) == {"taxonomy_alias", "index_meta"}
    with open_index(db_url, schema, taxonomy_dir) as index:
        assert [c.canonical_id for c in index.lookup_exact("teamwork")] == ["esco:teamwork"]


@pytest.mark.db
def test_lookup_exact_includes_exact_only_and_ambiguity(
    db_url: str, schema: str, taxonomy_dir: Path
) -> None:
    build(db_url, schema, taxonomy_dir)

    with open_index(db_url, schema, taxonomy_dir) as index:
        aws = index.lookup_exact("  AWS ")
        assert {c.canonical_id for c in aws} == {AWS, "esco:cloud"}
        assert all(c.distance == 0.0 for c in aws)
        assert [c.canonical_id for c in index.lookup_exact("r")] == ["esco:r"]  # exact_only
        assert index.lookup_exact("   ") == []
        assert index.warnings == []


@pytest.mark.db
def test_search_dedups_and_sorts(db_url: str, schema: str, taxonomy_dir: Path) -> None:
    build(db_url, schema, taxonomy_dir)

    with open_index(db_url, schema, taxonomy_dir) as index:
        results = index.search(["  AWS ", "", "teamwork"], k=5)
        assert index.search([]) == []

    aws, blank, team = results
    assert blank == []
    assert aws[0].distance == pytest.approx(0.0, abs=1e-5)  # same text, same fake vector
    assert aws[0].canonical_id in {AWS, "esco:cloud"}
    for ranked in (aws, team):
        ids = [c.canonical_id for c in ranked]
        assert len(ids) == len(set(ids))
        assert [c.distance for c in ranked] == sorted(c.distance for c in ranked)
    assert team[0].canonical_id == "esco:teamwork"
    assert "Go" not in {c.matched_alias for r in results for c in r}  # exact_only not searched


@pytest.mark.db
def test_index_warns_when_taxonomy_changed(db_url: str, schema: str, taxonomy_dir: Path) -> None:
    build(db_url, schema, taxonomy_dir)
    (taxonomy_dir / "manifest.json").write_text('{"esco_version": "changed"}')

    with open_index(db_url, schema, taxonomy_dir) as index:
        assert index.warnings == [TAXONOMY_CHANGED]


@pytest.mark.db
def test_staging_ddl_matches_schema_file(db_url: str, schema: str) -> None:
    def columns(conn: psycopg.Connection, table: str) -> list[tuple[str, str, bool]]:
        return conn.execute(
            "SELECT attname, format_type(atttypid, atttypmod), attnotnull FROM pg_attribute "
            "WHERE attrelid = %s::regclass AND attnum > 0 AND NOT attisdropped ORDER BY attnum",
            [table],
        ).fetchall()

    with connect(db_url, options=options(schema)) as conn:
        apply_schema(conn)
        conn.execute(create_table_sql("staging_probe"))
        assert columns(conn, "staging_probe") == columns(conn, "taxonomy_alias")
        conn.rollback()


# --- real model + real index -------------------------------------------------------


@pytest.fixture
def real_index() -> Iterator[SkillIndex]:
    url = database_url()
    if not url:
        pytest.skip("DATABASE_URL not set")
    try:
        index = SkillIndex.connect(url)
    except psycopg.OperationalError as exc:
        pytest.skip(f"database not reachable: {exc}")
    if "build" not in index.meta:
        index.close()
        pytest.skip("real index not built; run scripts/build_index.py")
    with index:
        yield index


@pytest.mark.model
def test_real_index_finds_obvious_skills(real_index: SkillIndex) -> None:
    python, docker = real_index.search(["python programming", "Docker containers"], k=5)

    assert "python" in python[0].canonical_name.lower()
    assert "jobpdf:skill/docker" in [c.canonical_id for c in docker[:3]]
