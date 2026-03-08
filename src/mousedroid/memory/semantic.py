"""Semantic index — FAISS-backed vector similarity search."""

from __future__ import annotations

from typing import Any

import faiss
import numpy as np
from numpy.typing import NDArray

from mousedroid.config.schema import MemoryConfig
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class SemanticIndex:
    """L2 nearest-neighbour index using FAISS ``IndexFlatL2``.

    Args:
        cfg: Memory configuration with ``semantic_dim``.
    """

    def __init__(self, cfg: MemoryConfig) -> None:
        self._dim = cfg.semantic_dim
        self._index: faiss.IndexFlatL2 = faiss.IndexFlatL2(self._dim)
        self._metadata: list[Any] = []

        _log.info("semantic_index_init", dim=self._dim)

    def store(self, key: str, value: NDArray[np.float32]) -> None:
        """Add a vector with associated metadata key.

        Args:
            key: Metadata identifier for the vector.
            value: Embedding vector, shape ``(semantic_dim,)``.
        """
        vec = np.ascontiguousarray(value.reshape(1, -1).astype(np.float32))
        self._index.add(vec)
        self._metadata.append(key)

    def retrieve(
        self,
        query: NDArray[np.float32],
        k: int = 1,
    ) -> list[tuple[str, float]]:
        """Find the *k* nearest neighbours to *query*.

        Args:
            query: Query vector, shape ``(semantic_dim,)``.
            k: Number of neighbours to return.

        Returns:
            List of ``(key, distance)`` tuples, nearest first.
        """
        if self._index.ntotal == 0:
            return []

        effective_k = min(k, self._index.ntotal)
        q = np.ascontiguousarray(query.reshape(1, -1).astype(np.float32))
        distances, indices = self._index.search(q, effective_k)

        results: list[tuple[str, float]] = []
        for i in range(effective_k):
            idx = int(indices[0, i])
            dist = float(distances[0, i])
            if idx < len(self._metadata):
                results.append((self._metadata[idx], dist))
        return results

    @property
    def size(self) -> int:
        """Number of stored vectors."""
        return int(self._index.ntotal)
