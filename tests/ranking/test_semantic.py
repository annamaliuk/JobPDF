import json
import math
from dataclasses import dataclass, field

import numpy as np
import pytest

from jobpdf.extraction.schema import ExtractedJob, ExtractedProfile, SkillMention
from jobpdf.normalization.embedder import shared_embedder
from jobpdf.ranking import semantic
from jobpdf.ranking.models import VacancyFeatures
from jobpdf.ranking.semantic import (
    CANDIDATE_TEXT_VERSION,
    DIMENSION_MISMATCH,
    EMPTY_EXPERIENCE_ITEM,
    FALLBACK_SKILLS_PATH,
    MAX_TEXT_CHARS,
    NO_CANDIDATE_TEXT,
    NO_EXPERIENCE_TEXT,
    NO_RESPONSIBILITIES,
    NO_VACANCY_EMBEDDING,
    STALE_VACANCY_EMBEDDING,
    STALE_VACANCY_TEXT,
    TRUNCATED,
    UNCALIBRATED,
    VACANCY_TEXT_VERSION,
    Calibration,
    CandidateVectors,
    build_candidate_texts,
    build_vacancy_texts,
    embed_candidate,
    embed_vacancy,
    embedding_version,
    load_calibration,
    semantic_scores,
    vacancy_text_warnings,
)

SQRT2 = math.sqrt(2)


@dataclass
class Job:
    title: str
    company: str | None = None
    description: str | None = None
    quote: str = ""


@dataclass
class Skill:
    name: str


@dataclass
class Profile:
    experience: list[Job] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)


class FakeEmbedder:
    """Known texts -> hand-made unit vectors; the same identity attributes as Embedder."""

    model_name = "fake/e5"
    revision = "rev1"
    prefix = "query: "

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.asarray([self.vectors[t] for t in texts], dtype=np.float32)


JOB_A = "Data Engineer\nAcme Widgets\nBuilt nightly ETL pipelines"
JOB_B = "Barista\nCorner Cafe\nMade coffee"
JOB_C = "Analytics Engineer\nGlobex\nModelled data in dbt"
VAC_TEXT = "Senior Data Engineer\nBuild data pipelines\nOwn the warehouse"
VECTORS = {
    JOB_A: [1.0, 0.0, 0.0, 0.0],
    JOB_B: [0.0, 1.0, 0.0, 0.0],
    JOB_C: [1 / SQRT2, 1 / SQRT2, 0.0, 0.0],
    VAC_TEXT: [0.8, 0.6, 0.0, 0.0],
}
VERSION = embedding_version(FakeEmbedder({}))
CAL = Calibration(floor=0.2, ceil=0.8, provisional=True, embedding_version=VERSION)


def profile_abc() -> Profile:
    return Profile(experience=[
        Job("Data Engineer", "Acme Widgets", "Built nightly ETL pipelines"),
        Job("Barista", "Corner Cafe", "Made coffee"),
        Job("Analytics Engineer", "Globex", "Modelled data in dbt"),
    ])


def vacancy(vector=None, *, id="v1", version=VERSION, text_version=VACANCY_TEXT_VERSION,
            responsibilities=("Build data pipelines", "Own the warehouse")) -> VacancyFeatures:
    return VacancyFeatures(
        id=id, source="synthetic", title="Senior Data Engineer",
        responsibilities=list(responsibilities), embedding=vector,
        embedding_version=version if vector is not None else None,
        vacancy_text_version=text_version if vector is not None else None,
    )


def candidate(*rows: list[float], version: str = VERSION) -> CandidateVectors:
    return CandidateVectors(vectors=[list(r) for r in rows],
                            paths=[f"experience[{i}]" for i in range(len(rows))],
                            text_version=CANDIDATE_TEXT_VERSION, embedding_version=version,
                            warnings=[])


# --- text builders -----------------------------------------------------------------


def test_one_text_per_job_with_paths_and_quote_fallback() -> None:
    profile = Profile(experience=[
        Job("Data Engineer", "Acme Widgets", "Built nightly ETL pipelines"),
        Job("Analyst", None, None, quote="Analyst, Globex 2019-2021"),
    ])

    texts, warnings = build_candidate_texts(profile)

    assert [(t.path, t.text) for t in texts] == [
        ("experience[0]", "Data Engineer\nAcme Widgets\nBuilt nightly ETL pipelines"),
        ("experience[1]", "Analyst\nAnalyst, Globex 2019-2021"),
    ]
    assert warnings == []


def test_empty_jobs_are_skipped_with_a_warning() -> None:
    profile = Profile(experience=[Job("  ", None, "  ", quote=""), Job("Data Engineer")])

    texts, warnings = build_candidate_texts(profile)

    assert [t.path for t in texts] == ["experience[1]"]
    assert warnings == [EMPTY_EXPERIENCE_ITEM]


def test_no_experience_falls_back_to_the_skills_list() -> None:
    texts, warnings = build_candidate_texts(Profile(skills=[Skill("Python"), Skill(" "),
                                                            Skill("SQL")]))

    assert [(t.path, t.text) for t in texts] == [(FALLBACK_SKILLS_PATH, "Python, SQL")]
    assert warnings == [NO_EXPERIENCE_TEXT]


def test_nothing_at_all_gives_no_texts() -> None:
    texts, warnings = build_candidate_texts(Profile(experience=[Job("")]))

    assert texts == [] and warnings == [EMPTY_EXPERIENCE_ITEM, NO_EXPERIENCE_TEXT]


def test_builders_accept_real_jm11_profiles() -> None:
    profile = ExtractedProfile(
        summary=None,
        skills=[SkillMention(name="Python", kind="hard", found_in="skills_list", quote="Python")],
        experience=[ExtractedJob(title="Data Engineer", company="Acme Widgets", location=None,
                                 start="2021", end="present", description=None,
                                 skills_used=[], quote="Data Engineer, Acme Widgets")],
        education=[], languages=[], certifications=[],
    )

    texts, _ = build_candidate_texts(profile)

    assert texts[0].text == "Data Engineer\nAcme Widgets\nData Engineer, Acme Widgets"


def test_vacancy_text_is_title_plus_responsibilities() -> None:
    assert build_vacancy_texts(vacancy()) == [VAC_TEXT]
    assert vacancy_text_warnings(vacancy()) == []


def test_vacancy_without_responsibilities_is_title_only() -> None:
    v = vacancy(responsibilities=())

    assert build_vacancy_texts(v) == ["Senior Data Engineer"]
    assert vacancy_text_warnings(v) == [NO_RESPONSIBILITIES]


def test_truncation_with_a_tokenizer_and_with_the_char_fallback() -> None:
    long_job = Profile(experience=[Job("Engineer", None, "x" * (MAX_TEXT_CHARS + 1))])

    texts, warnings = build_candidate_texts(long_job)
    assert texts[0].truncated and warnings == [TRUNCATED]

    # len() as a stand-in tokenizer: job texts are 54, 31 and 46 "tokens" long.
    texts, warnings = build_candidate_texts(profile_abc(), count_tokens=len, token_limit=40)
    assert [t.truncated for t in texts] == [True, False, True]
    assert warnings == [TRUNCATED]

    v = vacancy(responsibilities=["y" * MAX_TEXT_CHARS])
    assert vacancy_text_warnings(v) == [TRUNCATED]


def test_version_constants() -> None:
    assert (CANDIDATE_TEXT_VERSION, VACANCY_TEXT_VERSION) == ("1", "1")
    assert semantic.AGGREGATION == "max"
    assert embedding_version() == (
        "intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3|query: ")
    assert VERSION == "fake/e5@rev1|query: "


# --- embedding ---------------------------------------------------------------------


def test_embed_vacancy_returns_a_filled_copy() -> None:
    original = vacancy()
    embedder = FakeEmbedder(VECTORS)

    embedded = embed_vacancy(original, embedder)

    assert embedded.embedding == pytest.approx([0.8, 0.6, 0.0, 0.0])
    assert (embedded.embedding_version, embedded.vacancy_text_version) == (
        VERSION, VACANCY_TEXT_VERSION)
    assert original.embedding is None and original.embedding_version is None
    assert embedder.calls == [[VAC_TEXT]]
    assert embedded.model_copy(update={"embedding": None, "embedding_version": None,
                                       "vacancy_text_version": None}) == original


def test_embed_candidate_uses_one_batch() -> None:
    embedder = FakeEmbedder(VECTORS)

    vectors = embed_candidate(profile_abc(), embedder)

    assert embedder.calls == [[JOB_A, JOB_B, JOB_C]]
    assert vectors.paths == ["experience[0]", "experience[1]", "experience[2]"]
    assert vectors.vectors[2] == pytest.approx([1 / SQRT2, 1 / SQRT2, 0.0, 0.0])
    assert (vectors.text_version, vectors.embedding_version, vectors.warnings) == (
        CANDIDATE_TEXT_VERSION, VERSION, [])


def test_embed_candidate_without_texts_makes_no_call() -> None:
    embedder = FakeEmbedder({})

    vectors = embed_candidate(Profile(), embedder)

    assert vectors.vectors == [] and embedder.calls == []
    assert vectors.warnings == [NO_EXPERIENCE_TEXT]


# --- aggregation and rescaling -----------------------------------------------------


def test_max_aggregation_and_item_scores_in_job_order() -> None:
    cand = embed_candidate(profile_abc(), FakeEmbedder(VECTORS))
    vac = embed_vacancy(vacancy(), FakeEmbedder(VECTORS))

    (s,) = semantic_scores(cand, [vac], CAL)

    assert s.item_scores == pytest.approx([0.8, 0.6, 1.4 / SQRT2])
    assert s.best_item_index == 2
    assert s.raw == pytest.approx(1.4 / SQRT2)
    assert s.score == pytest.approx(min(1.0, (1.4 / SQRT2 - 0.2) / 0.6))
    assert (s.calibrated, s.warnings) == (True, [])


def test_top2_mean_aggregation() -> None:
    cand = candidate([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0])

    (s,) = semantic_scores(cand, [vacancy([0.8, 0.6, 0, 0])], CAL, aggregation="top2_mean")
    (single,) = semantic_scores(candidate([0, 1, 0, 0]), [vacancy([0.8, 0.6, 0, 0])], CAL,
                                aggregation="top2_mean")

    assert s.raw == pytest.approx((0.8 + 0.6) / 2) and s.best_item_index == 0
    assert single.raw == pytest.approx(0.6)


def test_unknown_aggregation_raises() -> None:
    with pytest.raises(ValueError):
        semantic_scores(candidate([1, 0, 0, 0]), [], CAL, aggregation="mean")


@pytest.mark.parametrize(
    ("raw_vec", "expected"),
    [
        ([0.2, math.sqrt(1 - 0.04), 0, 0], 0.0),  # exactly at the floor
        ([0.8, 0.6, 0, 0], 1.0),  # exactly at the ceiling
        ([0.5, math.sqrt(0.75), 0, 0], 0.5),
        ([0.1, math.sqrt(0.99), 0, 0], 0.0),  # below the floor: clipped
        ([0.95, math.sqrt(1 - 0.9025), 0, 0], 1.0),  # above the ceiling: clipped
    ],
)
def test_rescaling_and_clipping(raw_vec: list[float], expected: float) -> None:
    (s,) = semantic_scores(candidate([1, 0, 0, 0]), [vacancy(raw_vec)], CAL)

    assert s.score == pytest.approx(expected, abs=1e-9)


# --- when there is no score ----------------------------------------------------------


@pytest.mark.parametrize(
    ("vac", "code"),
    [
        (vacancy(None), NO_VACANCY_EMBEDDING),
        (vacancy([1, 0, 0, 0], version="other/model@x|query: "), STALE_VACANCY_EMBEDDING),
        (vacancy([1, 0, 0, 0], text_version="0"), STALE_VACANCY_TEXT),
        (vacancy([1, 0, 0]), DIMENSION_MISMATCH),
    ],
)
def test_vacancy_problems_give_none_with_a_warning(vac: VacancyFeatures, code: str) -> None:
    (s,) = semantic_scores(candidate([1, 0, 0, 0]), [vac], CAL)

    assert (s.score, s.raw, s.item_scores, s.best_item_index, s.warnings) == (
        None, None, [], None, [code])


def test_candidate_without_vectors_gives_none_everywhere() -> None:
    results = semantic_scores(candidate(), [vacancy([1, 0, 0, 0]), vacancy(None, id="v2")], CAL)

    assert [(s.score, s.warnings) for s in results] == [(None, [NO_CANDIDATE_TEXT])] * 2


def test_batch_equals_one_by_one_in_order() -> None:
    cand = candidate([1, 0, 0, 0], [0, 1, 0, 0], [0.6, 0, 0.8, 0])
    vacs = [vacancy([0.8, 0.6, 0, 0], id="a"), vacancy(None, id="b"),
            vacancy([0, 0, 1, 0], id="c"), vacancy([0, 0.28, 0.96, 0], id="d")]

    batch = semantic_scores(cand, vacs, CAL)
    single = [semantic_scores(cand, [v], CAL)[0] for v in vacs]

    assert len(batch) == len(vacs)
    for b, s in zip(batch, single, strict=True):
        assert b.best_item_index == s.best_item_index and b.warnings == s.warnings
        assert b.item_scores == pytest.approx(s.item_scores)
        assert (b.score is None) == (s.score is None)
        assert b.score is None or b.score == pytest.approx(s.score)
    assert semantic_scores(cand, vacs, CAL) == batch  # identical every time
    assert [b.best_item_index for b in batch] == [0, None, 2, 2]


# --- calibration -------------------------------------------------------------------


def write(tmp_path, data) -> object:
    path = tmp_path / "cal.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def test_valid_calibration_file_is_used(tmp_path) -> None:
    cal = load_calibration(write(tmp_path, {
        "floor": 0.2, "ceil": 0.8, "provisional": True, "model": "fake/e5@rev1",
        "embedding_version": VERSION, "dataset": "pairs.json", "dataset_sha256": "ab",
        "measured_at": "2026-10-10T12:00:00+00:00", "n_related": 10, "n_unrelated": 10,
    }))

    assert cal.is_valid and cal.provisional
    (s,) = semantic_scores(candidate([1, 0, 0, 0]), [vacancy([0.5, math.sqrt(0.75), 0, 0])],
                           cal)
    assert s.calibrated and s.score == pytest.approx(0.5)


@pytest.mark.parametrize(
    "content",
    [None, "{not json", {"floor": None, "ceil": None, "embedding_version": VERSION},
     {"floor": 0.8, "ceil": 0.8, "embedding_version": VERSION},
     {"floor": 0.9, "ceil": 0.2, "embedding_version": VERSION},
     {"floor": 0.2, "ceil": 0.8, "embedding_version": "another/model@x|query: "}],
    ids=["missing", "invalid_json", "nulls", "ceil_equals_floor", "ceil_below_floor",
         "other_model"],
)
def test_unusable_calibration_falls_back_to_clipped_raw(tmp_path, content) -> None:
    path = tmp_path / "missing.json" if content is None else write(tmp_path, content)

    cal = load_calibration(path)
    (s,) = semantic_scores(candidate([1, 0, 0, 0]), [vacancy([0.5, math.sqrt(0.75), 0, 0])],
                           cal)

    assert (s.calibrated, s.warnings) == (False, [UNCALIBRATED])
    assert s.score == pytest.approx(0.5) == s.raw


def test_committed_calibration_is_valid_provisional_and_for_the_current_model() -> None:
    cal = load_calibration(semantic.CALIBRATION_PATH)

    assert cal.is_valid and cal.provisional
    assert cal.embedding_version == embedding_version()
    assert cal.n_related and cal.n_unrelated and cal.dataset_sha256
    assert semantic.CALIBRATION == cal  # loaded once at import


def test_default_embedder_is_the_one_shared_with_the_skill_index() -> None:
    # One e5 model per process: semantic scoring (JM-23) and the index (JM-20) share it.
    assert semantic.default_embedder() is shared_embedder()
    assert semantic.default_embedder() is semantic.default_embedder()
