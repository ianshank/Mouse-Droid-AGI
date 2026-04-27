"""Multi-objective reward model with per-dimension heads and Three Laws integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.config.schema import ModelConfig, RewardConfig
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ThreeLawsConfig
    from mousedroid.reward.vlm_progress import VLMProgressHead

_log = get_logger(__name__)

# Module-level constants — single source of truth so other classes/tests don't
# reach into the private attributes of :class:`ThreeLawsRewardHead`.
LAW_HEAD_NAMES: tuple[str, ...] = ("law1_harm", "law2_obedience", "law3_preservation")
LAW1_HARM_KEY = "law1_harm"


class ThreeLawsRewardHead(nn.Module):
    """Reward heads encoding the Three Laws of Robotics.

    Three separate heads:
      - ``law1_harm``: Negative penalty for potential harm (Law 1)
      - ``law2_obedience``: Positive bonus for command compliance (Law 2)
      - ``law3_preservation``: Positive bonus for self-preservation (Law 3)

    Law 1 penalties are **multiplicative** — a Law 1 violation zeroes total reward.

    Args:
        input_dim: State embedding dimensionality.
        law_cfg: Three Laws configuration with reward weights.
    """

    _LAW_HEAD_NAMES = LAW_HEAD_NAMES

    def __init__(self, input_dim: int, law_cfg: ThreeLawsConfig) -> None:
        super().__init__()
        self.heads = nn.ModuleDict({name: nn.Linear(input_dim, 1) for name in LAW_HEAD_NAMES})
        self._weights = {
            "law1_harm": law_cfg.law1_reward_weight,
            "law2_obedience": law_cfg.law2_reward_weight,
            "law3_preservation": law_cfg.law3_reward_weight,
        }

    def compute_scores(self, state: Tensor) -> dict[str, Tensor]:
        """Compute per-law reward scores.

        Args:
            state: State embedding, shape ``(batch, obs_dim)``.

        Returns:
            Dict mapping law head name to scalar tensor.
        """
        return {name: head(state) for name, head in self.heads.items()}

    def aggregate(self, scores: dict[str, Tensor]) -> Tensor:
        """Aggregate law scores with hierarchical weighting.

        Law 1 (harm) is applied as a sigmoid gate — if the harm score
        is negative, it suppresses the total reward multiplicatively.
        Laws 2 and 3 are additive bonuses.

        Args:
            scores: Per-law score tensors.

        Returns:
            Scalar law reward modifier, shape ``(batch, 1)``.
        """
        # Law 1: sigmoid gate (negative harm score → near-zero multiplier)
        harm_gate = torch.sigmoid(scores[LAW1_HARM_KEY])

        # Laws 2 & 3: additive bonus
        bonus = (
            self._weights["law2_obedience"] * scores["law2_obedience"]
            + self._weights["law3_preservation"] * scores["law3_preservation"]
        )

        return harm_gate * bonus


class MultiObjectiveRewardModel(nn.Module):
    """Reward model with four objective heads and optional Three Laws integration.

    Heads: truthfulness, helpfulness, safety, engagement.
    Optional: law1_harm, law2_obedience, law3_preservation.
    Optional: vlm_progress (Phase 4) — VLM-derived dense progress reward
    that is gated by the Law-1 multiplicative sigmoid when the law head is
    present, so a contrived high progress value cannot override harm.

    Args:
        model_cfg: Model dimensions (uses ``obs_dim`` as input).
        reward_cfg: Per-objective weight configuration.
        law_cfg: Optional Three Laws configuration for law reward heads.
        vlm_head: Optional pre-built :class:`VLMProgressHead`. When ``None``,
            no VLM progress term is contributed even if
            ``reward_cfg.weight_vlm_progress > 0``.
    """

    _HEAD_NAMES = ("truthfulness", "helpfulness", "safety", "engagement")
    _VLM_HEAD_KEY = "vlm_progress"

    def __init__(
        self,
        model_cfg: ModelConfig,
        reward_cfg: RewardConfig,
        law_cfg: ThreeLawsConfig | None = None,
        vlm_head: VLMProgressHead | None = None,
    ) -> None:
        super().__init__()
        input_dim = model_cfg.obs_dim

        self.heads = nn.ModuleDict({name: nn.Linear(input_dim, 1) for name in self._HEAD_NAMES})

        self._weights: dict[str, float] = {
            "truthfulness": reward_cfg.weight_truthfulness,
            "helpfulness": reward_cfg.weight_helpfulness,
            "safety": reward_cfg.weight_safety,
            "engagement": reward_cfg.weight_engagement,
        }
        self._weight_vlm_progress = reward_cfg.weight_vlm_progress

        self.law_head: ThreeLawsRewardHead | None = None
        if law_cfg is not None and law_cfg.enabled:
            self.law_head = ThreeLawsRewardHead(input_dim, law_cfg)

        self.vlm_head: VLMProgressHead | None = vlm_head

        _log.info(
            "reward_model_init",
            weights=self._weights,
            three_laws_enabled=self.law_head is not None,
            vlm_progress_enabled=self.vlm_head is not None,
            weight_vlm_progress=self._weight_vlm_progress,
        )

    def compute_reward(
        self,
        state: Tensor,
        *,
        prev_obs: Tensor | None = None,
        curr_obs: Tensor | None = None,
        instruction: str | None = None,
    ) -> dict[str, Tensor]:
        """Compute per-objective reward scores.

        Args:
            state: State embedding, shape ``(batch, obs_dim)``.
            prev_obs: Optional previous observation for the VLM progress head.
            curr_obs: Optional current observation for the VLM progress head.
                When both are provided and ``vlm_head`` is set, a
                ``vlm_progress`` entry is added to the returned dict.
            instruction: Optional instruction override for the VLM head.

        Returns:
            Dictionary mapping head name to scalar reward tensor.
        """
        scores = {name: head(state) for name, head in self.heads.items()}
        if self.law_head is not None:
            scores.update(self.law_head.compute_scores(state))
        if self.vlm_head is not None and prev_obs is not None and curr_obs is not None:
            scores[self._VLM_HEAD_KEY] = self.vlm_head.score(
                prev_obs, curr_obs, instruction=instruction
            )
        return scores

    def aggregate(self, scores: dict[str, Tensor]) -> Tensor:
        """Aggregate per-objective scores into a scalar reward.

        The base reward sums the four standard heads. When Three Laws heads
        are present, the law-2/3 bonus and the (optional) VLM progress term
        are summed and multiplied by the Law-1 ``sigmoid(harm)`` gate, then
        added to the base. When the law head is absent, the VLM term is
        added directly.

        Args:
            scores: Per-head score tensors from :meth:`compute_reward`.

        Returns:
            Weighted scalar reward, shape ``(batch, 1)``.
        """
        # Base reward from standard heads (unchanged from pre-Phase 4 behavior).
        base_scores = {k: v for k, v in scores.items() if k in self._HEAD_NAMES}
        result = torch.zeros_like(next(iter(base_scores.values())))
        for name, score in base_scores.items():
            result = result + self._weights[name] * score

        # Gated extras: law-2/3 bonus + VLM progress, all under sigmoid(law1_harm).
        vlm_score = scores.get(self._VLM_HEAD_KEY)
        if self.law_head is not None:
            law_scores = {k: v for k, v in scores.items() if k in LAW_HEAD_NAMES}
            if law_scores:
                law_modifier = self.law_head.aggregate(law_scores)
                result = result + law_modifier
                if vlm_score is not None:
                    harm_gate = torch.sigmoid(law_scores[LAW1_HARM_KEY])
                    result = result + harm_gate * (self._weight_vlm_progress * vlm_score)
        elif vlm_score is not None:
            result = result + self._weight_vlm_progress * vlm_score

        return result

    def forward(
        self,
        state: Tensor,
        *,
        prev_obs: Tensor | None = None,
        curr_obs: Tensor | None = None,
        instruction: str | None = None,
    ) -> Tensor:
        """Compute and aggregate reward in one call.

        Args:
            state: State embedding, shape ``(batch, obs_dim)``.
            prev_obs: Optional previous observation for the VLM progress head.
            curr_obs: Optional current observation for the VLM progress head.
            instruction: Optional instruction override for the VLM head.

        Returns:
            Scalar reward, shape ``(batch, 1)``.
        """
        scores = self.compute_reward(
            state, prev_obs=prev_obs, curr_obs=curr_obs, instruction=instruction
        )
        return self.aggregate(scores)
