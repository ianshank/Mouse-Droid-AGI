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

    def __init__(self, cfg: MemoryConfig, seed: int | None = None) -> None:
        self._capacity = cfg.episodic_capacity
        self._buffer: deque[tuple[Any, float, int]] = deque(maxlen=cfg.episodic_capacity)
        self._rng = np.random.default_rng(seed)
        self._seq_counter = 0

        _log.info("episodic_replay_init", capacity=cfg.episodic_capacity, seed=seed)

    def push(self, experience: Any, priority: float = 1.0) -> None:
        """Add an experience with an associated priority.

        Args:
            experience: Arbitrary experience data.
            priority: Sampling priority (must be non-negative).
        """
        safe_priority = float(np.clip(priority, 0.0, None))
        if not np.isfinite(safe_priority):
            safe_priority = 1.0
        self._buffer.append((experience, safe_priority, self._seq_counter))
        self._seq_counter += 1

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
            [p for _, p, _ in self._buffer],
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

        indices = self._rng.choice(len(self._buffer), size=n, replace=False, p=probabilities)
        return [self._buffer[int(i)][0] for i in indices]

    def cursor_query(self, cursor: int | None, limit: int) -> tuple[list[Any], int | None]:
        """Fetch items sequentially using a cursor (sequence ID).

        Iterates from newest to oldest. If a cursor is provided, returns
        items strictly OLDER than the cursor.

        Args:
            cursor: Sequence ID to fetch items older than, or None for newest.
            limit: Maximum items to return.

        Returns:
            Tuple of (experiences, next_cursor). next_cursor is the sequence
            ID of the last item returned, or None if no more items.
        """
        if not self._buffer:
            return [], None

        results = []
        next_cursor = None

        # Iterate reverse (newest first)
        for i in range(len(self._buffer) - 1, -1, -1):
            exp, _, seq = self._buffer[i]
            if cursor is not None and seq >= cursor:
                continue

            results.append(exp)
            next_cursor = seq

            if len(results) >= limit:
                break

        return results, next_cursor

    def __len__(self) -> int:
        """Current buffer size."""
        return len(self._buffer)
