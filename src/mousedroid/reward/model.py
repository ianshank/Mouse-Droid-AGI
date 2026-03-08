"""Multi-objective reward model with per-dimension heads."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.config.schema import ModelConfig, RewardConfig
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class MultiObjectiveRewardModel(nn.Module):
    """Reward model with four objective heads and weighted aggregation.

    Heads: truthfulness, helpfulness, safety, engagement.

    Args:
        model_cfg: Model dimensions (uses ``obs_dim`` as input).
        reward_cfg: Per-objective weight configuration.
    """

    _HEAD_NAMES = ("truthfulness", "helpfulness", "safety", "engagement")

    def __init__(self, model_cfg: ModelConfig, reward_cfg: RewardConfig) -> None:
        super().__init__()
        input_dim = model_cfg.obs_dim

        self.heads = nn.ModuleDict(
            {name: nn.Linear(input_dim, 1) for name in self._HEAD_NAMES}
        )

        self._weights: dict[str, float] = {
            "truthfulness": reward_cfg.weight_truthfulness,
            "helpfulness": reward_cfg.weight_helpfulness,
            "safety": reward_cfg.weight_safety,
            "engagement": reward_cfg.weight_engagement,
        }

        _log.info("reward_model_init", weights=self._weights)

    def compute_reward(self, state: Tensor) -> dict[str, Tensor]:
        """Compute per-objective reward scores.

        Args:
            state: State embedding, shape ``(batch, obs_dim)``.

        Returns:
            Dictionary mapping head name to scalar reward tensor.
        """
        return {name: head(state) for name, head in self.heads.items()}

    def aggregate(self, scores: dict[str, Tensor]) -> Tensor:
        """Aggregate per-objective scores into a scalar reward.

        Args:
            scores: Per-head score tensors from ``compute_reward``.

        Returns:
            Weighted scalar reward, shape ``(batch, 1)``.
        """
        result = torch.zeros_like(next(iter(scores.values())))
        for name, score in scores.items():
            result = result + self._weights[name] * score
        return result

    def forward(self, state: Tensor) -> Tensor:
        """Compute and aggregate reward in one call.

        Args:
            state: State embedding, shape ``(batch, obs_dim)``.

        Returns:
            Scalar reward, shape ``(batch, 1)``.
        """
        scores = self.compute_reward(state)
        return self.aggregate(scores)
