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
        assert (action >= -1.0).all()
        assert (action <= 1.0).all()


class TestAgentMeta:
    def test_name(self) -> None:
        agent, _, _ = _make_agent()
        assert agent.name == "mouse_droid_navigator"

    def test_reset(self) -> None:
        agent, _, _ = _make_agent()
        agent.reset()  # Should not raise
