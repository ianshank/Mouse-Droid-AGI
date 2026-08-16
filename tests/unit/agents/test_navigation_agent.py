"""Tests for MouseDroidNavigationAgent — action selection and safety overrides."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from mousedroid.agents.navigation import MouseDroidNavigationAgent
from mousedroid.config.schema import Settings
from mousedroid.safety.context import SafetyContext


def _make_agent() -> tuple[MouseDroidNavigationAgent, MagicMock, Settings]:
    """Create agent with mock MCTSPlanner."""
    cfg = Settings(mock_hardware=True)
    mock_planner = MagicMock()
    mock_planner.plan.return_value = torch.tensor([[0.1, 0.0, 0.0]])
    agent = MouseDroidNavigationAgent(mock_planner, cfg)
    return agent, mock_planner, cfg


def _h_z(cfg: Settings) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(1, cfg.model.hidden_dim), torch.zeros(1, cfg.model.latent_dim)


class TestEmergencyStop:
    def test_emergency_returns_zeros(self) -> None:
        agent, _, cfg = _make_agent()
        h, z = _h_z(cfg)
        ctx = SafetyContext(is_emergency=True)
        action = agent.act(h, z, ctx)
        assert torch.allclose(action, torch.zeros(cfg.model.action_dim))

    def test_emergency_overrides_all(self) -> None:
        agent, _, cfg = _make_agent()
        h, z = _h_z(cfg)
        ctx = SafetyContext(is_emergency=True, forward_clearance_ok=True)
        action = agent.act(h, z, ctx)
        assert torch.allclose(action, torch.zeros(cfg.model.action_dim))


class TestLaw1HumanProximity:
    def test_human_close_stops(self) -> None:
        agent, _, cfg = _make_agent()
        h, z = _h_z(cfg)
        ctx = SafetyContext(human_detected=True, human_dist_m=0.2)
        action = agent.act(h, z, ctx)
        assert torch.allclose(action, torch.zeros(cfg.model.action_dim))

    def test_human_far_allows_action(self) -> None:
        agent, _, cfg = _make_agent()
        h, z = _h_z(cfg)
        ctx = SafetyContext(human_detected=True, human_dist_m=1.0)
        action = agent.act(h, z, ctx)
        assert action.shape == (cfg.model.action_dim,)

    def test_no_human_allows_action(self) -> None:
        agent, _, cfg = _make_agent()
        h, z = _h_z(cfg)
        ctx = SafetyContext()
        action = agent.act(h, z, ctx)
        assert action.shape == (cfg.model.action_dim,)


class TestForwardClearance:
    def test_no_clearance_reverses(self) -> None:
        agent, _, cfg = _make_agent()
        h, z = _h_z(cfg)
        ctx = SafetyContext(forward_clearance_ok=False)
        action = agent.act(h, z, ctx)
        assert float(action[0]) == -0.5


class TestActionBounds:
    def test_actions_within_bounds(self) -> None:
        agent, _, cfg = _make_agent()
        h, z = _h_z(cfg)
        ctx = SafetyContext()
        action = agent.act(h, z, ctx)
        action_min = torch.tensor(cfg.safety.action_min)
        action_max = torch.tensor(cfg.safety.action_max)
        assert torch.all(action >= action_min)
        assert torch.all(action <= action_max)

    def test_custom_action_bounds_are_applied(self) -> None:
        cfg = Settings(
            mock_hardware=True,
            safety={
                "action_min": [-0.2, -0.3, -0.4],
                "action_max": [0.2, 0.3, 0.4],
            },
        )
        planner = MagicMock()
        planner.plan.return_value = torch.tensor([[0.9, -0.9, 0.8]])
        agent = MouseDroidNavigationAgent(planner, cfg)
        h, z = _h_z(cfg)

        action = agent.act(h, z, SafetyContext())

        assert torch.allclose(action, torch.tensor([0.2, -0.3, 0.4]))


class TestSurpriseAdaptiveBudget:
    def test_zero_surprise_passes_base_budget(self) -> None:
        agent, planner, cfg = _make_agent()
        h, z = _h_z(cfg)
        ctx = SafetyContext(surprise=0.0)
        agent.act(h, z, ctx)
        planner.plan.assert_called_once()
        _, kwargs = planner.plan.call_args
        assert kwargs["n_simulations"] == cfg.mcts.n_simulations_base

    def test_high_surprise_increases_budget(self) -> None:
        agent, planner, cfg = _make_agent()
        h, z = _h_z(cfg)
        ctx = SafetyContext(surprise=5.0)
        agent.act(h, z, ctx)
        _, kwargs = planner.plan.call_args
        assert kwargs["n_simulations"] > cfg.mcts.n_simulations_base

    def test_budget_never_exceeds_maximum(self) -> None:
        agent, planner, cfg = _make_agent()
        h, z = _h_z(cfg)
        ctx = SafetyContext(surprise=1000.0)
        agent.act(h, z, ctx)
        _, kwargs = planner.plan.call_args
        assert kwargs["n_simulations"] <= cfg.mcts.n_simulations_max


class TestAgentMeta:
    def test_name(self) -> None:
        agent, _, _ = _make_agent()
        assert agent.name == "mouse_droid_navigator"

    def test_reset(self) -> None:
        agent, _, _ = _make_agent()
        agent.reset()  # Should not raise
