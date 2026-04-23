from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from mousedroid.agents.navigation import MouseDroidNavigationAgent
from mousedroid.common.actions import normalize_action_numpy
from mousedroid.config.schema import Settings
from mousedroid.safety.context import SafetyContext


def _make_agent() -> tuple[MouseDroidNavigationAgent, torch.Tensor, torch.Tensor]:
    cfg = Settings(mock_hardware=True)
    mock_planner = MagicMock()
    mock_planner.plan.return_value = torch.tensor([[0.1, 0.0, 0.0]])
    agent = MouseDroidNavigationAgent(mock_planner, cfg)
    h = torch.zeros(1, cfg.model.hidden_dim)
    z = torch.zeros(1, cfg.model.latent_dim)
    return agent, h, z


def test_actions_in_bounds_normal_context() -> None:
    agent, h, z = _make_agent()
    cfg = Settings(mock_hardware=True)
    ctx = SafetyContext()
    action = agent.act(h, z, ctx)
    assert (action >= torch.tensor(cfg.safety.action_min)).all()
    assert (action <= torch.tensor(cfg.safety.action_max)).all()


def test_actions_in_bounds_low_clearance() -> None:
    agent, h, z = _make_agent()
    cfg = Settings(mock_hardware=True)
    ctx = SafetyContext(forward_clearance_ok=False, ultrasonic_dist_m=0.1)
    action = agent.act(h, z, ctx)
    assert (action >= torch.tensor(cfg.safety.action_min)).all()
    assert (action <= torch.tensor(cfg.safety.action_max)).all()


def test_emergency_context_returns_zeros() -> None:
    agent, h, z = _make_agent()
    ctx = SafetyContext(is_emergency=True)
    action = agent.act(h, z, ctx)
    assert torch.allclose(action, torch.zeros_like(action))


def test_actions_with_high_surprise() -> None:
    agent, h, z = _make_agent()
    cfg = Settings(mock_hardware=True)
    ctx = SafetyContext(surprise=10.0)
    action = agent.act(h, z, ctx)
    assert (action >= torch.tensor(cfg.safety.action_min)).all()
    assert (action <= torch.tensor(cfg.safety.action_max)).all()


def test_actions_with_low_battery_context() -> None:
    agent, h, z = _make_agent()
    cfg = Settings(mock_hardware=True)
    ctx = SafetyContext(battery_voltage=9.0)
    action = agent.act(h, z, ctx)
    assert (action >= torch.tensor(cfg.safety.action_min)).all()
    assert (action <= torch.tensor(cfg.safety.action_max)).all()


def test_settings_expand_default_action_bounds_to_action_dim() -> None:
    cfg = Settings(mock_hardware=True, model={"action_dim": 4})

    assert cfg.safety.action_min == [-1.0, -1.0, -1.0, -1.0]
    assert cfg.safety.action_max == [1.0, 1.0, 1.0, 1.0]


def test_settings_reject_action_bound_length_mismatch() -> None:
    with pytest.raises(ValueError, match=r"action_min length"):
        Settings(
            mock_hardware=True,
            model={"action_dim": 2},
            safety={"action_min": [-1.0], "action_max": [1.0, 1.0]},
        )


def test_normalize_action_numpy_uses_configured_bounds() -> None:
    action = normalize_action_numpy(
        np.array([0.9, -0.9, 0.8], dtype=np.float32),
        expected_dim=3,
        action_min=[-0.2, -0.4, -0.5],
        action_max=[0.2, 0.3, 0.5],
    )

    assert torch.allclose(action, torch.tensor([0.2, -0.4, 0.5]))
