import numpy as np

from jobpdf.normalization.embedder import EMBEDDING_DIM, Embedder, l2_normalize


class FakeModel:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.seen = list(texts)
        assert kwargs["normalize_embeddings"] is True
        return np.full((len(texts), EMBEDDING_DIM), 2.0, dtype=np.float64)


def make(prefix: str) -> tuple[Embedder, FakeModel]:
    embedder = Embedder(prefix=prefix)
    model = FakeModel()
    embedder._model = model  # type: ignore[assignment]  # skip the real download
    return embedder, model


def test_prefix_is_prepended_and_output_is_unit_float32() -> None:
    embedder, model = make("query: ")

    vectors = embedder.embed(["Python", "SQL"])

    assert model.seen == ["query: Python", "query: SQL"]
    assert vectors.dtype == np.float32
    assert vectors.shape == (2, EMBEDDING_DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_other_prefix() -> None:
    embedder, model = make("passage: ")

    embedder.embed(["Docker"])

    assert model.seen == ["passage: Docker"]


def test_empty_input_does_not_load_the_model() -> None:
    embedder = Embedder()

    assert embedder.embed([]).shape == (0, EMBEDDING_DIM)
    assert embedder._model is None


def test_l2_normalize_keeps_zero_rows_finite() -> None:
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))

    assert np.allclose(out, [[0.6, 0.8], [0.0, 0.0]])
