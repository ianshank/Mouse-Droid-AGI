"""Mouse Droid navigation agent — MCTS-based action selection with safety override."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mousedroid.logging.setup import get_logger
from mousedroid.safety.context import SafetyContext

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings
    from mousedroid.world_model.protocol import WorldModelProtocol

_log = get_logger(__name__)


class MouseDroidNavigationAgent:
    """MCTS-based navigation agent for the Mouse Droid.

    Uses world model for planning and safety context for action override.
    Implements AgentProtocol.
    """

    def __init__(self, world_model: WorldModelProtocol, cfg: Settings) -> None:
        """Initialise navigation agent.

        Args:
            world_model: World model for latent-space planning.
            cfg: Root settings.
        """
        self._world_model = world_model
        self._cfg = cfg
        self._action_dim = cfg.model.action_dim
        self._name = "mouse_droid_navigator"

    @property
    def name(self) -> str:
        """Agent identifier."""
        return self._name

    def act(
        self,
        h: Tensor,
        z: Tensor,
        safety_ctx: SafetyContext,
    ) -> Tensor:
        """Select action via MCTS planning with safety override.

        Args:
            h: RNN hidden state.
            z: Latent state.
            safety_ctx: Current safety context.

        Returns:
            Action tensor, shape ``(action_dim,)``, values in ``[-1, 1]``.
        """
        if safety_ctx.is_emergency:
            _log.warning("emergency_stop", reason="safety_emergency")
            return torch.zeros(self._action_dim)

        # Law 1: Human proximity — full stop
        if safety_ctx.human_detected and safety_ctx.human_dist_m < 0.5:
            _log.warning(
                "law1_human_proximity_stop",
                human_dist_m=safety_ctx.human_dist_m,
            )
            return torch.zeros(self._action_dim)

        if not safety_ctx.forward_clearance_ok:
            _log.info("reverse_action", distance_m=safety_ctx.ultrasonic_dist_m)
            action = torch.zeros(self._action_dim)
            action[0] = -0.5  # Reverse
            return action

        from mousedroid.agents._planning import compute_mcts_budget

        budget = compute_mcts_budget(
            surprise=safety_ctx.surprise,
            base=self._cfg.mcts.n_simulations_base,
            maximum=self._cfg.mcts.n_simulations_max,
        )
        _log.debug("mcts_planning", budget=budget, surprise=safety_ctx.surprise)

        action = self._plan_action(h, z, budget)
        return torch.clamp(action, -1.0, 1.0)

    def _plan_action(self, h: Tensor, z: Tensor, budget: int) -> Tensor:
        """Plan action using world model imagination.

        Args:
            h: RNN hidden state.
            z: Latent state.
            budget: Number of MCTS simulations.

        Returns:
            Planned action tensor.
        """
        best_action = torch.zeros(self._action_dim)
        best_reward = float("-inf")

        n_candidates = self._cfg.mcts.n_action_candidates
        gamma = self._cfg.mcts.gamma
        depth = self._cfg.mcts.rollout_depth

        for _ in range(min(budget, n_candidates)):
            candidate = torch.rand(self._action_dim) * 2.0 - 1.0
            total_reward = 0.0
            h_sim, z_sim = h.clone(), z.clone()

            with torch.no_grad():
                for step in range(depth):
                    h_sim, z_sim, reward = self._world_model.imagine_step(
                        candidate,
                        h_sim,
                        z_sim,
                    )
                    total_reward += (gamma**step) * reward.item()

            if total_reward > best_reward:
                best_reward = total_reward
                best_action = candidate

        return best_action

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        _log.info("agent_reset", agent=self._name)
