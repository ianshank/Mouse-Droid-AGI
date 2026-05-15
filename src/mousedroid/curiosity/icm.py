"""Intrinsic Curiosity Module (ICM) for exploration with optional novelty decay."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from mousedroid.config.schema import CuriosityConfig, ModelConfig
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class NoveltyDecay:
    """Tracks state visitation and decays curiosity for familiar regions.

    Maintains a hash-based visit counter that exponentially reduces
    the intrinsic reward scale for frequently visited state regions.
    States are discretised into bins for counting.

    Args:
        decay_rate: Exponential decay rate per visit.
        min_scale: Minimum scale factor (prevents total suppression).
        n_bins: Number of discretisation bins per dimension.
    """

    def __init__(
        self,
        decay_rate: float = 0.01,
        min_scale: float = 0.01,
        n_bins: int = 32,
    ) -> None:
        self._decay_rate = decay_rate
        self._min_scale = min_scale
        self._n_bins = n_bins
        self._visit_counts: dict[int, int] = {}

    @property
    def decay_rate(self) -> float:
        """Exponential decay rate per visit."""
        return self._decay_rate

    @property
    def min_scale(self) -> float:
        """Minimum curiosity scale after decay."""
        return self._min_scale

    @property
    def total_visits(self) -> int:
        """Total number of state visits recorded."""
        return sum(self._visit_counts.values())

    @property
    def unique_states(self) -> int:
        """Number of unique discretised states visited."""
        return len(self._visit_counts)

    def _discretise(self, state: Tensor) -> int:
        """Hash a state tensor into a bucket index.

        Args:
            state: 1-D state tensor.

        Returns:
            Integer hash key.
        """
        # Clamp to [-1, 1] and bin
        clamped = state.detach().clamp(-1.0, 1.0)
        bins = ((clamped + 1.0) * 0.5 * (self._n_bins - 1)).long()
        return hash(tuple(bins.cpu().tolist()))

    def get_scale(self, state: Tensor) -> float:
        """Get the novelty scale factor for a state (without incrementing visits).

        Args:
            state: 1-D state tensor.

        Returns:
            Scale factor in ``[min_scale, 1.0]``.
        """
        key = self._discretise(state)
        count = self._visit_counts.get(key, 0)
        if count == 0:
            return 1.0
        decayed = math.exp(-self._decay_rate * count)
        return max(decayed, self._min_scale)

    def record_visit(self, state: Tensor) -> float:
        """Record a visit and return the resulting scale factor.

        Args:
            state: 1-D state tensor.

        Returns:
            Scale factor in ``[min_scale, 1.0]`` after recording the visit.
        """
        key = self._discretise(state)
        self._visit_counts[key] = self._visit_counts.get(key, 0) + 1
        decayed = math.exp(-self._decay_rate * self._visit_counts[key])
        return max(decayed, self._min_scale)

    def reset(self) -> None:
        """Clear all visit counts."""
        self._visit_counts.clear()


class IntrinsicCuriosityModule(nn.Module):
    """ICM: forward model predicts next state, inverse model predicts action.

    Intrinsic reward is the forward model's prediction error, scaled by
    a configurable factor and optionally decayed by novelty tracking.

    Args:
        model_cfg: Model dimensions (``obs_dim``, ``action_dim``).
        curiosity_cfg: Curiosity hyper-parameters (scale, hidden dims, decay).
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        curiosity_cfg: CuriosityConfig,
    ) -> None:
        super().__init__()
        obs_dim = model_cfg.obs_dim
        action_dim = model_cfg.action_dim
        fwd_hidden = curiosity_cfg.forward_model_hidden
        inv_hidden = curiosity_cfg.inverse_model_hidden
        self._scale = curiosity_cfg.intrinsic_reward_scale

        # Forward model: (s, a) -> s'
        self.forward_model = nn.Sequential(
            nn.Linear(obs_dim + action_dim, fwd_hidden),
            nn.ReLU(),
            nn.Linear(fwd_hidden, obs_dim),
        )

        # Inverse model: (s, s') -> a
        self.inverse_model = nn.Sequential(
            nn.Linear(obs_dim + obs_dim, inv_hidden),
            nn.ReLU(),
            nn.Linear(inv_hidden, action_dim),
        )

        # Novelty decay (optional)
        self._novelty_decay: NoveltyDecay | None = None
        if curiosity_cfg.novelty_decay_enabled:
            self._novelty_decay = NoveltyDecay(
                decay_rate=curiosity_cfg.novelty_decay_rate,
                min_scale=curiosity_cfg.novelty_min_scale,
                n_bins=curiosity_cfg.novelty_n_bins,
            )

        _log.info(
            "icm_init",
            obs_dim=obs_dim,
            action_dim=action_dim,
            scale=self._scale,
            novelty_decay_enabled=self._novelty_decay is not None,
        )

    @property
    def novelty_decay(self) -> NoveltyDecay | None:
        """Access the novelty decay tracker, if enabled."""
        return self._novelty_decay

    def forward(
        self,
        s: Tensor,
        a: Tensor,
        s_next: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute forward/inverse predictions and losses.

        Args:
            s: Current state embedding, shape ``(batch, obs_dim)``.
            a: Action taken, shape ``(batch, action_dim)``.
            s_next: Next state embedding, shape ``(batch, obs_dim)``.

        Returns:
            Tuple of ``(forward_loss, inverse_loss, predicted_next_state)``.
        """
        # Forward model
        pred_s_next: Tensor = self.forward_model(torch.cat([s, a], dim=-1))
        forward_loss = F.mse_loss(pred_s_next, s_next)

        # Inverse model
        pred_action: Tensor = self.inverse_model(torch.cat([s, s_next], dim=-1))
        inverse_loss = F.mse_loss(pred_action, a)

        return forward_loss, inverse_loss, pred_s_next

    @torch.no_grad()
    def intrinsic_reward(self, s: Tensor, a: Tensor, s_next: Tensor) -> Tensor:
        """Compute intrinsic curiosity reward with optional novelty decay.

        Args:
            s: Current state embedding, shape ``(batch, obs_dim)``.
            a: Action taken, shape ``(batch, action_dim)``.
            s_next: Next state embedding, shape ``(batch, obs_dim)``.

        Returns:
            Intrinsic reward, shape ``(batch,)``.
        """
        pred_s_next = self.forward_model(torch.cat([s, a], dim=-1))
        error = F.mse_loss(pred_s_next, s_next, reduction="none").mean(dim=-1)
        reward = self._scale * error

        if self._novelty_decay is not None:
            scales = torch.tensor(
                [self._novelty_decay.record_visit(s[i]) for i in range(s.shape[0])],
                device=reward.device,
                dtype=reward.dtype,
            )
            reward = reward * scales

        return reward

    def reset_episode(self) -> None:
        """Reset per-episode accumulators.

        Clears novelty-decay visit counts so curiosity scores start fresh
        at the next episode boundary.
        """
        if self._novelty_decay is not None:
            self._novelty_decay.reset()
        _log.info("icm_episode_reset")
