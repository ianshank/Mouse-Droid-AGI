from __future__ import annotations

from unittest.mock import MagicMock

import torch

from mousedroid.agents.navigation import MouseDroidNavigationAgent
from mousedroid.config.schema import Settings
from mousedroid.safety.context import SafetyContext


def _make_agent() -> tuple[MouseDroidNavigationAgent, torch.Tensor, torch.Tensor]:
    cfg = Settings(mock_hardware=True)
    mock_wm = MagicMock()
    mock_wm.imagine_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.tensor([[0.1]]),
    )
    agent = MouseDroidNavigationAgent(mock_wm, cfg)
    h = torch.zeros(1, cfg.model.hidden_dim)
    z = torch.zeros(1, cfg.model.latent_dim)
    return agent, h, z


def test_actions_in_bounds_normal_context() -> None:
    agent, h, z = _make_agent()
    ctx = SafetyContext()
    action = agent.act(h, z, ctx)
    assert (action >= -1.0).all()
    assert (action <= 1.0).all()


def test_actions_in_bounds_low_clearance() -> None:
    agent, h, z = _make_agent()
    ctx = SafetyContext(forward_clearance_ok=False, ultrasonic_dist_m=0.1)
    action = agent.act(h, z, ctx)
    assert (action >= -1.0).all()
    assert (action <= 1.0).all()


def test_emergency_context_returns_zeros() -> None:
    agent, h, z = _make_agent()
    ctx = SafetyContext(is_emergency=True)
    action = agent.act(h, z, ctx)
    assert torch.allclose(action, torch.zeros_like(action))


def test_actions_with_high_surprise() -> None:
    agent, h, z = _make_agent()
    ctx = SafetyContext(surprise=10.0)
    action = agent.act(h, z, ctx)
    assert (action >= -1.0).all()
    assert (action <= 1.0).all()


def test_actions_with_low_battery_context() -> None:
    agent, h, z = _make_agent()
    ctx = SafetyContext(battery_voltage=9.0)
    action = agent.act(h, z, ctx)
    assert (action >= -1.0).all()
    assert (action <= 1.0).all()
