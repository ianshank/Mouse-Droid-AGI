"""Mouse Droid navigation agent — MCTS-based action selection with safety override."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mousedroid.agents._planning import compute_mcts_budget
from mousedroid.logging.setup import get_logger
from mousedroid.safety.context import SafetyContext

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings
    from mousedroid.world_model.mcts import MCTSPlanner

_log = get_logger(__name__)


class MouseDroidNavigationAgent:
    """MCTS-based navigation agent for the Mouse Droid.

    Uses MCTSPlanner for latent-space planning and safety context for
    action override. Implements AgentProtocol.
    """

    def __init__(self, planner: MCTSPlanner, cfg: Settings) -> None:
        """Initialise navigation agent.

        Args:
            planner: MCTS planner for latent-space action selection.
            cfg: Root settings.
        """
        self._planner = planner
        self._cfg = cfg
        self._action_dim = cfg.model.action_dim
        self._human_safety_radius_m = cfg.three_laws.human_safety_radius_m
        self._action_min = torch.tensor(cfg.safety.action_min, dtype=torch.float32)
        self._action_max = torch.tensor(cfg.safety.action_max, dtype=torch.float32)
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
        if safety_ctx.human_detected and safety_ctx.human_dist_m < self._human_safety_radius_m:
            _log.warning(
                "law1_human_proximity_stop",
                human_dist_m=safety_ctx.human_dist_m,
            )
            return torch.zeros(self._action_dim)

        if not safety_ctx.forward_clearance_ok:
            _log.info("reverse_action", distance_m=safety_ctx.ultrasonic_dist_m)
            action = torch.zeros(self._action_dim)
            action[0] = self._cfg.safety.reverse_velocity
            return action

        budget = compute_mcts_budget(
            safety_ctx.surprise,
            base=self._cfg.mcts.n_simulations_base,
            maximum=self._cfg.mcts.n_simulations_max,
        )
        _log.debug("mcts_planning", surprise=safety_ctx.surprise, budget=budget)
        action = self._planner.plan(h, z, n_simulations=budget)

        # MCTSPlanner returns (1, action_dim); squeeze to (action_dim,)
        if action.dim() == 2:
            action = action.squeeze(0)

        action_min = self._action_min.to(device=action.device, dtype=action.dtype)
        action_max = self._action_max.to(device=action.device, dtype=action.dtype)
        return torch.max(torch.min(action, action_max), action_min)

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        _log.info("agent_reset", agent=self._name)
