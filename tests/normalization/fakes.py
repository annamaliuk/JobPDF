"""Test doubles shared by the index tests: no model download, deterministic vectors."""

from __future__ import annotations

import hashlib

import numpy as np

from jobpdf.normalization.embedder import EMBEDDING_DIM, QUERY_PREFIX, l2_normalize


class FakeEmbedder:
    """Same interface as Embedder; unit vectors seeded by a hash of the text.

    Texts that normalise the same (case, spaces) get the same vector, so a query
    like "  AWS " lands exactly on the alias "aws", like a real model would roughly.
    """

    def __init__(self, prefix: str = QUERY_PREFIX) -> None:
        self.prefix = prefix
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        self.calls.append(list(texts))
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        return l2_normalize(np.stack([_vector(self.prefix + t) for t in texts]))


def _vector(text: str) -> np.ndarray:
    key = " ".join(text.lower().split()).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    return np.random.default_rng(seed).standard_normal(EMBEDDING_DIM).astype(np.float32)
