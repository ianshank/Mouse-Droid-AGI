"""Intrinsic Curiosity Module (ICM) for exploration."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from mousedroid.config.schema import CuriosityConfig, ModelConfig
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class IntrinsicCuriosityModule(nn.Module):
    """ICM: forward model predicts next state, inverse model predicts action.

    Intrinsic reward is the forward model's prediction error, scaled by
    a configurable factor.

    Args:
        model_cfg: Model dimensions (``obs_dim``, ``action_dim``).
        curiosity_cfg: Curiosity hyper-parameters (scale, hidden dims).
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

        _log.info(
            "icm_init",
            obs_dim=obs_dim,
            action_dim=action_dim,
            scale=self._scale,
        )

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

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def intrinsic_reward(self, s: Tensor, a: Tensor, s_next: Tensor) -> Tensor:
        """Compute intrinsic curiosity reward.

        Args:
            s: Current state embedding, shape ``(batch, obs_dim)``.
            a: Action taken, shape ``(batch, action_dim)``.
            s_next: Next state embedding, shape ``(batch, obs_dim)``.

        Returns:
            Intrinsic reward, shape ``(batch,)``.
        """
        pred_s_next = self.forward_model(torch.cat([s, a], dim=-1))
        error = F.mse_loss(pred_s_next, s_next, reduction="none").mean(dim=-1)
        return self._scale * error
