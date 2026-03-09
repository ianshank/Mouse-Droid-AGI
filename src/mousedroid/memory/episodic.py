"""Episodic replay — prioritized experience buffer."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.config.schema import MemoryConfig
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class EpisodicReplay:
    """Prioritized replay buffer backed by a bounded deque.

    Priorities are normalized into sampling probabilities with
    ``np.isfinite`` guards to prevent NaN propagation.

    Args:
        cfg: Memory configuration with ``episodic_capacity``.
    """

    def __init__(self, cfg: MemoryConfig) -> None:
        self._capacity = cfg.episodic_capacity
        self._buffer: deque[tuple[Any, float]] = deque(maxlen=cfg.episodic_capacity)

        _log.info("episodic_replay_init", capacity=cfg.episodic_capacity)

    def push(self, experience: Any, priority: float = 1.0) -> None:
        """Add an experience with an associated priority.

        Args:
            experience: Arbitrary experience data.
            priority: Sampling priority (must be non-negative).
        """
        safe_priority = float(np.clip(priority, 0.0, None))
        if not np.isfinite(safe_priority):
            safe_priority = 1.0
        self._buffer.append((experience, safe_priority))

    def sample(self, batch_size: int) -> list[Any]:
        """Sample a batch of experiences weighted by priority.

        Args:
            batch_size: Number of experiences to sample.

        Returns:
            List of sampled experiences.
        """
        if len(self._buffer) == 0:
            return []

        n = min(batch_size, len(self._buffer))
        priorities: NDArray[np.float64] = np.array(
            [p for _, p in self._buffer],
            dtype=np.float64,
        )

        # Safe normalization: replace non-finite with uniform weight.
        finite_mask = np.isfinite(priorities)
        if not finite_mask.any():
            priorities = np.ones_like(priorities)
        else:
            priorities[~finite_mask] = 0.0

        total = priorities.sum()
        if total <= 0.0:
            probabilities = np.ones(len(priorities)) / len(priorities)
        else:
            probabilities = priorities / total

        rng = np.random.default_rng()
        indices = rng.choice(len(self._buffer), size=n, replace=False, p=probabilities)
        return [self._buffer[int(i)][0] for i in indices]

    def __len__(self) -> int:
        """Current buffer size."""
        return len(self._buffer)
