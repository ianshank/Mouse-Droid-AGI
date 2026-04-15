"""Memory tier — aggregates all four memory subsystems.

Provides a single injectable object that holds references to all memory
components.  Built by ``build_memory_tier()`` in the factory module and
passed to the orchestrator via constructor DI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mousedroid.memory.consolidation import MemoryConsolidation
    from mousedroid.memory.episodic import EpisodicReplay
    from mousedroid.memory.semantic import SemanticIndex
    from mousedroid.memory.working import WorkingMemory


@dataclass(frozen=True)
class MemoryTier:
    """Aggregates all four memory subsystems.

    Attributes:
        episodic: Priority-weighted episodic replay buffer.
        semantic: FAISS-backed semantic vector index.
        working: FIFO working memory with attention-based retrieval.
        consolidation: Offline consolidation (episodic -> semantic).
    """

    episodic: EpisodicReplay
    semantic: SemanticIndex
    working: WorkingMemory
    consolidation: MemoryConsolidation
