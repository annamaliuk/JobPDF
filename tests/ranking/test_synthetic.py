"""The synthetic ranking dataset (JM-25): rubric, determinism, hard cases, files."""

import csv
import importlib.util
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import pytest

from jobpdf.ranking.score import rank_vacancies
from jobpdf.ranking.semantic import embed_candidate, embed_vacancy
from jobpdf.ranking.synthetic import (
    DATA_FILES,
    GRADES,
    MANIFEST_FILE,
    MIN_PAIRS_PER_GRADE,
    TRAPS,
    DatasetError,
    calibration_pairs,
    generate,
    grade,
    load_dataset,
    relation,
    required_share,
    serialize,
    truth_candidate_skills,
    truth_key,
    truth_vacancy,
)
from jobpdf.ranking.synthetic_catalog import ADJACENT, FAMILIES, SKILLS
from tests.normalization.fakes import FakeEmbedder

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "synthetic" / "ranking"
CUSTOM_SKILLS = ROOT / "data" / "taxonomy" / "custom_skills.csv"


@pytest.fixture(scope="module")
def dataset():
    return generate()


def pairs_of(dataset, candidate_id):
    return [p for p in dataset.pairs if p.candidate_id == candidate_id]


# --- rubric ------------------------------------------------------------------------


@pytest.mark.parametrize(("rel", "share", "expected"), [
    ("same", 1.0, 3), ("same", 0.8, 3), ("same", 0.79, 2), ("same", 0.5, 2),
    ("same", 0.49, 1), ("same", 0.25, 1), ("same", 0.24, 0), ("same", 0.0, 0),
    ("adjacent", 1.0, 2), ("adjacent", 0.6, 1), ("adjacent", 0.3, 0), ("adjacent", 0.0, 0),
    ("unrelated", 1.0, 1), ("unrelated", 0.8, 1), ("unrelated", 0.79, 0),
])
def test_grade_follows_the_rubric_table(rel, share, expected) -> None:
    assert grade(rel, share) == expected


def test_relation_is_symmetric_and_uses_the_adjacency_map() -> None:
    assert relation("backend", "backend") == "same"
    assert relation("backend", "devops") == relation("devops", "backend") == "adjacent"
    assert relation("product", "devops") == "unrelated"
    assert all(len(pair) == 2 and pair <= set(FAMILIES) for pair in ADJACENT)


def test_required_share_counts_required_skills_only() -> None:
    assert required_share({"python", "sql"}, ["python", "sql", "docker", "go"]) == 0.5
    assert required_share({"python"}, []) == 0.0


def test_every_grade_is_recomputed_from_ground_truth(dataset) -> None:
    for pair in dataset.pairs:
        ct = dataset.truth.candidates[pair.candidate_id]
        vt = dataset.truth.vacancies[pair.vacancy_id]
        have = {slug for slug in ct.skills.values() if slug}
        share = required_share(have, vt.required)
        rel = relation(ct.home_family, vt.family)
        assert (pair.relation, pair.required_coverage) == (rel, round(share, 4))
        assert pair.grade == grade(rel, share)
        assert pair.expected_band == {3: "strong", 2: "partial"}.get(pair.grade, "weak")


# --- shape -------------------------------------------------------------------------


def test_default_size_is_40_vacancies_20_candidates_800_pairs(dataset) -> None:
    assert (len(dataset.vacancies), len(dataset.candidates), len(dataset.pairs)) == (40, 20, 800)
    assert Counter(dataset.truth.vacancies[v.id].family for v in dataset.vacancies) == {
        f: 5 for f in FAMILIES}
    assert len({(p.candidate_id, p.vacancy_id) for p in dataset.pairs}) == 800


def test_size_is_a_setting() -> None:
    large = generate(vacancies_per_family=10, candidates=40)
    assert (len(large.vacancies), len(large.candidates), len(large.pairs)) == (80, 40, 3200)
    assert large.vacancies[:40] == generate().vacancies  # growing keeps existing vacancies
    with pytest.raises(ValueError, match="at least 5"):
        generate(vacancies_per_family=4)


def test_every_regular_candidate_has_enough_pairs_at_every_grade(dataset) -> None:
    for c in dataset.candidates:
        counts = Counter(p.grade for p in pairs_of(dataset, c.id))
        if dataset.truth.candidates[c.id].trap in (None, "alias_only", "career_changer"):
            assert all(counts[g] >= MIN_PAIRS_PER_GRADE for g in GRADES), c.id
        else:
            assert len(counts) >= 2, c.id  # Spearman needs more than one grade


def test_vacancies_use_canonical_names_and_fill_later_fields(dataset) -> None:
    for v in dataset.vacancies:
        vt = dataset.truth.vacancies[v.id]
        assert [r.raw for r in v.required_skills] == [SKILLS[s].name for s in vt.required]
        assert 5 <= len(vt.required) <= 8 and 2 <= len(vt.preferred) <= 4
        assert not set(vt.required) & set(vt.preferred)
        assert v.source == "synthetic" and v.location_raw and v.min_years_experience
        assert 3 <= len(v.responsibilities) <= 5


def test_every_skill_name_on_a_cv_is_in_the_truth_map(dataset) -> None:
    for c in dataset.candidates:
        assert [s.name for s in c.profile.skills] == list(dataset.truth.candidates[c.id].skills)


def test_some_candidates_write_job_lines_in_ukrainian(dataset) -> None:
    uk = [c for c in dataset.candidates if dataset.truth.candidates[c.id].lang == "uk"]
    assert len(uk) >= 2
    assert all(re.search("[а-яіїєґ]", c.profile.experience[0].description) for c in uk)


# --- hard cases --------------------------------------------------------------------


def trap_candidate(dataset, trap):
    [cid] = [cid for cid, t in dataset.truth.candidates.items() if t.trap == trap]
    return cid


def test_each_hard_case_appears_once_in_the_default_set(dataset) -> None:
    traps = Counter(t.trap for t in dataset.truth.candidates.values() if t.trap)
    assert traps == {t: 1 for t in TRAPS}


def test_keyword_stuffer_gets_grade_1_on_the_unrelated_vacancy_it_copies(dataset) -> None:
    cid = trap_candidate(dataset, "keyword_stuffer")
    [target] = [p for p in pairs_of(dataset, cid) if p.trap_target]
    assert (target.relation, target.required_coverage, target.grade) == ("unrelated", 1.0, 1)


def test_title_only_copies_the_title_but_gets_grade_0(dataset) -> None:
    cid = trap_candidate(dataset, "title_only")
    [target] = [p for p in pairs_of(dataset, cid) if p.trap_target]
    vacancy = next(v for v in dataset.vacancies if v.id == target.vacancy_id)
    candidate = next(c for c in dataset.candidates if c.id == cid)
    assert candidate.profile.experience[0].title == vacancy.title
    assert target.relation == "same" and target.required_coverage < 0.25 and target.grade == 0


def test_alias_only_writes_every_skill_with_an_alias_as_its_alias(dataset) -> None:
    cid = trap_candidate(dataset, "alias_only")
    written = dataset.truth.candidates[cid].skills
    with_alias = {name: slug for name, slug in written.items() if slug and SKILLS[slug].aliases}
    assert with_alias
    assert all(name in SKILLS[slug].aliases for name, slug in with_alias.items())


def test_career_changer_is_graded_against_the_recent_family(dataset) -> None:
    cid = trap_candidate(dataset, "career_changer")
    home = dataset.truth.candidates[cid].home_family
    jobs = next(c for c in dataset.candidates if c.id == cid).profile.experience
    assert len(jobs) == 4
    assert all(j.title in FAMILIES[home].titles for j in jobs[:2])
    assert not any(j.title in FAMILIES[home].titles for j in jobs[2:])


# --- taxonomy and privacy ----------------------------------------------------------


def test_every_alias_resolves_to_the_same_concept_as_its_name() -> None:
    """So JM-20 normalization can resolve the aliases CVs use; an alias that pointed
    elsewhere would make the dataset test the taxonomy instead of the ranking."""
    alias_ids: dict[str, set[str]] = {}
    with CUSTOM_SKILLS.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            alias_ids.setdefault(row["alias"].strip().lower(), set()).add(row["canonical_id"])
            if row["canonical_name"].strip():
                alias_ids.setdefault(row["canonical_name"].strip().lower(), set()).add(
                    row["canonical_id"])
    for skill in SKILLS.values():
        for alias in skill.aliases:
            assert alias in alias_ids, f"{alias!r} is not in custom_skills.csv"
            if skill.name.lower() in alias_ids:
                assert alias_ids[alias] == alias_ids[skill.name.lower()], alias


def test_no_contact_data_anywhere_in_the_files(dataset) -> None:
    text = b"".join(serialize(dataset, 25, 5).values()).decode("utf-8")
    assert not re.search(r"[\w.+-]+@[\w-]+\.\w+", text)
    assert not re.search(r"\+\d[\d ()-]{7,}\d|\(\d{3}\) ?\d{3}-\d{4}", text)
    assert "http" not in text and "www." not in text


# --- files -------------------------------------------------------------------------


def test_generation_is_byte_identical_across_runs() -> None:
    assert serialize(generate(), 25, 5) == serialize(generate(), 25, 5)


def test_committed_files_are_what_the_generator_writes(dataset) -> None:
    """Fails when the generator changed but the dataset was not regenerated:
    run scripts/make_ranking_pairs.py and commit the result."""
    expected = serialize(dataset, 25, 5)
    for name, raw in expected.items():
        assert (DATA_DIR / name).read_bytes().replace(b"\r\n", b"\n") == raw, name


def test_load_round_trips_the_committed_dataset(dataset) -> None:
    assert load_dataset(DATA_DIR) == dataset


def test_load_refuses_a_file_that_no_longer_matches_the_manifest(tmp_path: Path) -> None:
    shutil.copytree(DATA_DIR, tmp_path / "d")
    pairs = tmp_path / "d" / "pairs.jsonl"
    pairs.write_text(pairs.read_text(encoding="utf-8").replace('"grade": 0', '"grade": 3', 1),
                     encoding="utf-8")
    with pytest.raises(DatasetError, match="pairs.jsonl"):
        load_dataset(tmp_path / "d")


def test_load_names_a_missing_file(tmp_path: Path) -> None:
    shutil.copytree(DATA_DIR, tmp_path / "d")
    (tmp_path / "d" / "truth.json").unlink()
    with pytest.raises(DatasetError, match="truth.json"):
        load_dataset(tmp_path / "d")


def test_manifest_records_seed_sizes_and_hashes() -> None:
    manifest = json.loads((DATA_DIR / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["seed"] == 25 and manifest["vacancies_per_family"] == 5
    assert manifest["counts"] == {"vacancies": 40, "candidates": 20, "pairs": 800}
    assert set(manifest["files"]) == set(DATA_FILES)


# --- evaluation variants -----------------------------------------------------------


def test_truth_variant_gives_both_sides_the_same_keys(dataset) -> None:
    cid = trap_candidate(dataset, "alias_only")
    skills = truth_candidate_skills(dataset, cid)
    have = {slug for slug in dataset.truth.candidates[cid].skills.values() if slug}
    assert {truth_key(s) for s in have} <= skills.keys
    vacancy = truth_vacancy(dataset, dataset.vacancies[0])
    assert all(r.key.startswith("synthetic:") for r in vacancy.required_skills)


def test_rank_vacancies_runs_over_the_whole_set_without_network(dataset) -> None:
    embedder = FakeEmbedder()
    embedder.model_name, embedder.revision = "fake/e5", "rev1"
    vacancies = [embed_vacancy(truth_vacancy(dataset, v), embedder) for v in dataset.vacancies]
    for c in dataset.candidates:
        result = rank_vacancies(truth_candidate_skills(dataset, c.id),
                                embed_candidate(c.profile, embedder), vacancies)
        assert len(result.results) == len(vacancies)
        assert all(r.fit is not None for r in result.results)


def test_calibration_export_has_both_classes_in_the_calibration_format(dataset) -> None:
    pairs = calibration_pairs(dataset)["pairs"]
    assert {p["related"] for p in pairs} == {True, False}
    assert all(p["cv_text"] and p["vacancy_text"] and isinstance(p["related"], bool)
               for p in pairs)
    assert len({p["id"] for p in pairs}) == len(pairs)


def test_script_writes_the_dataset_and_prints_counts_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = importlib.util.spec_from_file_location(
        "make_ranking_pairs", ROOT / "scripts" / "make_ranking_pairs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    code = module.main(["--out", str(tmp_path / "out"),
                        "--export-calibration", str(tmp_path / "cal.json")])

    printed = capsys.readouterr().out
    assert code == 0
    assert "40 vacancies, 20 candidates, 800 pairs" in printed
    assert (tmp_path / "out" / "pairs.jsonl").read_bytes() == (
        DATA_DIR / "pairs.jsonl").read_bytes().replace(b"\r\n", b"\n")
    assert json.loads((tmp_path / "cal.json").read_text(encoding="utf-8"))["pairs"]
    assert "ETL" not in printed and "Engineer" not in printed
