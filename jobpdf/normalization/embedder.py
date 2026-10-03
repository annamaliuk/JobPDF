"""Sentence embeddings with multilingual-e5-small (JM-17, reused by JM-23).

Deliberately free of taxonomy code: it turns texts into unit vectors, nothing else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"
# Pinned commit of the model repo, so a silent upstream update can't change
# vectors under an existing index. Read from the HF cache after the first download.
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
# e5 expects a prefix. We use "query: " on BOTH sides: aliases and CV skills are
# short phrases of the same kind (symmetric matching), not query vs document.
QUERY_PREFIX = "query: "
EMBEDDING_DIM = 384
BATCH_SIZE = 64


class TextEmbedder(Protocol):
    """What SkillIndex and the build need; tests substitute a fake."""

    prefix: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class Embedder:
    """Loads the model once, on first use, and embeds batches of texts."""

    def __init__(
        self,
        prefix: str = QUERY_PREFIX,
        model_name: str = MODEL_NAME,
        revision: str = MODEL_REVISION,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.prefix = prefix
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            # Imported here so importing this module (e.g. for exact lookups) stays cheap.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name, revision=self.revision, device="cpu"
            )
        return self._model

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """float32 array of shape (len(texts), EMBEDDING_DIM), each row L2-normalized."""
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        vectors = self.model.encode(
            [self.prefix + text for text in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return l2_normalize(np.asarray(vectors, dtype=np.float32))


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Unit-length rows; cosine distance in pgvector then equals 1 - dot product."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.where(norms == 0, 1.0, norms)).astype(np.float32)
