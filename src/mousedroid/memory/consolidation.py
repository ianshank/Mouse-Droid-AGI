"""Memory consolidation — offline hippocampal replay between memory tiers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mousedroid.config.schema import MemoryConfig
from mousedroid.logging.setup import get_logger
from mousedroid.memory.episodic import EpisodicReplay
from mousedroid.memory.semantic import SemanticIndex

_log = get_logger(__name__)


class MemoryConsolidation:
    """Offline consolidation: replay episodic memories into the semantic index.

    Periodically samples a batch of episodic experiences, extracts their
    embeddings, and upserts them into the semantic long-term store.

    Args:
        cfg: Memory configuration with ``consolidation_batch_size``.
        episodic: Episodic replay buffer (source).
        semantic: Semantic index (target).
    """

    def __init__(
        self,
        cfg: MemoryConfig,
        episodic: EpisodicReplay,
        semantic: SemanticIndex,
    ) -> None:
        self._batch_size = cfg.consolidation_batch_size
        self._episodic = episodic
        self._semantic = semantic
        self._consolidation_count = 0

        _log.info("consolidation_init", batch_size=self._batch_size)

    def consolidate(self) -> int:
        """Run one consolidation cycle.

        Samples a batch from episodic memory and stores embeddings in
        the semantic index.

        Returns:
            Number of experiences consolidated in this cycle.
        """
        batch = self._episodic.sample(self._batch_size)
        consolidated = 0

        for experience in batch:
            embedding = self._extract_embedding(experience)
            if embedding is not None:
                key = f"consolidation_{self._consolidation_count}"
                self._semantic.store(key, embedding)
                self._consolidation_count += 1
                consolidated += 1

        _log.debug("consolidation_cycle", consolidated=consolidated)
        return consolidated

    @staticmethod
    def _extract_embedding(experience: object) -> NDArray[np.float32] | None:
        """Extract a float32 embedding from an experience object.

        Args:
            experience: Experience data; must expose an ``embedding`` attribute
                        that is convertible to a numpy float32 array.

        Returns:
            The embedding array or ``None`` if extraction fails.
        """
        embedding_attr = getattr(experience, "embedding", None)
        if embedding_attr is None:
            return None
        return np.asarray(embedding_attr, dtype=np.float32)
