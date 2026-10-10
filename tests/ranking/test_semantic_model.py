"""JM-23 against the real e5 model (skipped unless it is in the local cache)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from jobpdf.normalization.embedder import EMBEDDING_DIM, MODEL_NAME, MODEL_REVISION, Embedder
from jobpdf.ranking.models import VacancyFeatures
from jobpdf.ranking.semantic import (
    TRUNCATED,
    Calibration,
    CandidateVectors,
    embed_candidate,
    embed_vacancy,
    embedding_version,
    load_calibration,
    semantic_scores,
)

ROOT = Path(__file__).resolve().parents[2]
PAIRS_PATH = Path(__file__).parent / "fixtures" / "semantic_calibration_pairs.json"
PAIRS = {p["id"]: p for p in json.loads(PAIRS_PATH.read_text(encoding="utf-8"))["pairs"]}


def model_is_cached() -> bool:
    # Checked up front so a test never downloads the model.
    from huggingface_hub import try_to_load_from_cache

    return isinstance(
        try_to_load_from_cache(MODEL_NAME, "config.json", revision=MODEL_REVISION), str)


pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not model_is_cached(), reason="e5 model not in the local HF cache"),
]


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


def cv_vectors(embedder: Embedder, text: str) -> CandidateVectors:
    vector = embedder.embed([text])[0]
    return CandidateVectors(vectors=[[float(x) for x in vector]], paths=["experience[0]"],
                            text_version="1", embedding_version=embedding_version(embedder),
                            warnings=[])


def vacancy_from_text(embedder: Embedder, text: str) -> VacancyFeatures:
    title, *responsibilities = text.split("\n")
    return embed_vacancy(VacancyFeatures(id=title, source="synthetic", title=title,
                                         responsibilities=responsibilities), embedder)


class _Job:
    def __init__(self, title: str, company: str | None, description: str | None) -> None:
        self.title, self.company, self.description, self.quote = title, company, description, ""


class _Profile:
    def __init__(self, jobs: list[_Job]) -> None:
        self.experience, self.skills = jobs, []


def test_same_text_gives_the_same_vector_as_jm17(embedder: Embedder) -> None:
    profile = _Profile([_Job("Data Engineer", "Acme Widgets", "Built ETL pipelines")])

    vectors = embed_candidate(profile, embedder)
    expected = embedder.embed(["Data Engineer\nAcme Widgets\nBuilt ETL pipelines"])[0]

    assert np.allclose(vectors.vectors[0], expected, atol=1e-6)
    assert len(vectors.vectors[0]) == EMBEDDING_DIM


def test_vacancy_embedding_has_the_index_dimension(embedder: Embedder) -> None:
    vacancy = vacancy_from_text(embedder, PAIRS["r01-en-data-engineer"]["vacancy_text"])

    assert len(vacancy.embedding) == EMBEDDING_DIM
    assert vacancy.embedding_version == embedding_version(embedder)


@pytest.mark.parametrize(
    ("related_id", "unrelated_id"),
    [("r01-en-data-engineer", "u01-en-data-vs-chef"),
     ("r07-uk-data-engineer", "u07-uk-data-vs-housekeeper"),
     ("r08-uk-accountant", "u08-uk-accountant-vs-welder")],
)
def test_related_scores_higher_than_unrelated(
    embedder: Embedder, related_id: str, unrelated_id: str,
) -> None:
    related, unrelated = PAIRS[related_id], PAIRS[unrelated_id]
    assert related["cv_text"] == unrelated["cv_text"]  # same CV, two vacancies
    candidate = cv_vectors(embedder, related["cv_text"])
    vacancies = [vacancy_from_text(embedder, related["vacancy_text"]),
                 vacancy_from_text(embedder, unrelated["vacancy_text"])]

    good, bad = semantic_scores(candidate, vacancies, Calibration())

    assert good.raw > bad.raw


def test_long_text_is_flagged_by_the_real_tokenizer(embedder: Embedder) -> None:
    profile = _Profile([_Job("Engineer", None, "pipelines " * 600)])

    assert TRUNCATED in embed_candidate(profile, embedder).warnings


def test_calibration_script_writes_a_valid_file(embedder: Embedder, tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "calibrate_semantic", ROOT / "scripts" / "calibrate_semantic.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    out = tmp_path / "calibration.json"

    assert script.main(["--pairs", str(PAIRS_PATH), "--out", str(out), "--provisional"],
                       embedder=embedder) == 0

    calibration = load_calibration(out)
    assert calibration.is_valid and calibration.provisional
    assert calibration.embedding_version == embedding_version(embedder)
    assert (calibration.n_related, calibration.n_unrelated) == (10, 10)
