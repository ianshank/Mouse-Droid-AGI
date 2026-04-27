"""Deterministic sim/real episode mixer with optional alpha ramp.

The mixer answers a single question on each ``draw()`` call: should the next
training step pull from the synthetic (sim) pool, or the real-replay pool?

It is deliberately **stateless across processes** — caller owns the seed and
step counter, so the same ``(seed, step)`` pair always produces the same draw.

Design invariants
-----------------
* All randomness flows through a caller-injected :class:`numpy.random.Generator`
  (or its seed). No global RNG state is touched.
* When ``alpha_ramp_steps == 0``, alpha is the constant ``alpha_target`` from
  step 0 onwards (instant ramp).
* When ``alpha_target == 0.0`` (or no real episodes), the mixer always draws
  from the sim pool — preserving byte-identical behavior with the
  pre-Phase-2 dataset loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import structlog

_log = structlog.get_logger(__name__)

Source = Literal["sim", "real"]


@dataclass
class MixerStats:
    """Realized counters for one mixer's lifetime."""

    sim_draws: int = 0
    real_draws: int = 0

    @property
    def total(self) -> int:
        """Return the total number of draws (sim + real) recorded so far."""
        return self.sim_draws + self.real_draws

    @property
    def realized_ratio(self) -> float:
        """Empirical fraction of real draws (returns 0.0 before any draw)."""
        if self.total == 0:
            return 0.0
        return self.real_draws / self.total


class EpisodeMixer:
    """Stateless ratio-controlled mixer over sim/real episode pools.

    Args:
        alpha_target: Target probability of drawing from the real-replay pool.
            Must lie in ``[0.0, 1.0]``.
        alpha_ramp_steps: Number of training steps to linearly ramp ``alpha``
            from 0.0 to ``alpha_target``. ``0`` => no ramp (alpha is constant
            at ``alpha_target``).
        rng: Caller-provided NumPy generator. Use :func:`numpy.random.default_rng`
            with an explicit seed for reproducibility.
        has_real_pool: If False (no real episodes available), the mixer always
            returns ``"sim"`` regardless of alpha. This guarantees byte-identical
            behavior when the real pool is empty.

    Raises:
        ValueError: If ``alpha_target`` is outside ``[0.0, 1.0]`` or
            ``alpha_ramp_steps`` is negative.
    """

    def __init__(
        self,
        *,
        alpha_target: float,
        alpha_ramp_steps: int,
        rng: np.random.Generator,
        has_real_pool: bool,
    ) -> None:
        if not 0.0 <= alpha_target <= 1.0:
            msg = f"alpha_target must be in [0.0, 1.0], got {alpha_target}"
            raise ValueError(msg)
        if alpha_ramp_steps < 0:
            msg = f"alpha_ramp_steps must be >= 0, got {alpha_ramp_steps}"
            raise ValueError(msg)

        self._alpha_target = alpha_target
        self._ramp_steps = alpha_ramp_steps
        self._rng = rng
        self._has_real_pool = has_real_pool
        self.stats = MixerStats()

    def alpha_at(self, step: int) -> float:
        """Compute the alpha value at a given training step.

        The ramp is linear: ``alpha(step) = alpha_target * min(1, step / ramp_steps)``.

        Args:
            step: 0-based training step.

        Returns:
            Effective alpha in ``[0.0, alpha_target]``. When the real pool is
            empty, returns 0.0 unconditionally.
        """
        if not self._has_real_pool:
            return 0.0
        if step < 0:
            msg = f"step must be >= 0, got {step}"
            raise ValueError(msg)
        if self._ramp_steps == 0:
            return self._alpha_target
        progress = min(1.0, step / float(self._ramp_steps))
        return self._alpha_target * progress

    def draw(self, step: int) -> Source:
        """Sample one source ("sim" or "real") for the given training step.

        Args:
            step: Current training step (drives the ramp).

        Returns:
            ``"real"`` with probability ``alpha_at(step)``, else ``"sim"``.
        """
        alpha = self.alpha_at(step)
        if alpha <= 0.0:
            self.stats.sim_draws += 1
            return "sim"
        if alpha >= 1.0:
            self.stats.real_draws += 1
            return "real"
        u = float(self._rng.random())
        if u < alpha:
            self.stats.real_draws += 1
            return "real"
        self.stats.sim_draws += 1
        return "sim"

    def draw_batch(self, step: int, batch_size: int) -> list[Source]:
        """Vectorized batch draw.

        Args:
            step: Current training step (one ramp value applied to all draws
                in the batch — matches typical PPO/BC mini-batch semantics).
            batch_size: Number of independent draws.

        Returns:
            List of ``Source`` of length ``batch_size``.
        """
        if batch_size <= 0:
            msg = f"batch_size must be positive, got {batch_size}"
            raise ValueError(msg)
        alpha = self.alpha_at(step)
        if alpha <= 0.0:
            self.stats.sim_draws += batch_size
            return ["sim"] * batch_size
        if alpha >= 1.0:
            self.stats.real_draws += batch_size
            return ["real"] * batch_size
        draws = self._rng.random(batch_size) < alpha
        n_real = int(draws.sum())
        self.stats.real_draws += n_real
        self.stats.sim_draws += batch_size - n_real
        return ["real" if d else "sim" for d in draws.tolist()]
