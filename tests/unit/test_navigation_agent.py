from __future__ import annotations

import pytest
import torch

from mousedroid.agents.navigation import MouseDroidNavigationAgent
from mousedroid.config.schema import Settings
from mousedroid.safety.context import SafetyContext


class MockWorldModel:
    def observe_step(self, observation, prev_action, h, z):
        return h, z, h, 0.0

    def imagine_step(self, action, h, z):
        new_h = h.clone() if hasattr(h, "clone") else h
        new_z = z.clone() if hasattr(z, "clone") else z
        reward = torch.zeros(1, 1)
        return new_h, new_z, reward


@pytest.fixture
def cfg() -> Settings:
    return Settings(mock_hardware=True)


@pytest.fixture
def agent(cfg: Settings) -> MouseDroidNavigationAgent:
    return MouseDroidNavigationAgent(MockWorldModel(), cfg)


def test_constructor(agent: MouseDroidNavigationAgent) -> None:
    assert agent._name == "mouse_droid_navigator"


def test_name_property(agent: MouseDroidNavigationAgent) -> None:
    assert agent.name == "mouse_droid_navigator"


def test_act_returns_tensor(agent: MouseDroidNavigationAgent) -> None:
    h = torch.zeros(1, 256)
    z = torch.zeros(1, 64)
    ctx = SafetyContext()
    action = agent.act(h, z, ctx)
    assert isinstance(action, torch.Tensor)


def test_act_values_in_range(agent: MouseDroidNavigationAgent) -> None:
    h = torch.zeros(1, 256)
    z = torch.zeros(1, 64)
    ctx = SafetyContext()
    action = agent.act(h, z, ctx)
    assert (action >= -1.0).all()
    assert (action <= 1.0).all()


def test_act_action_dim(agent: MouseDroidNavigationAgent, cfg: Settings) -> None:
    h = torch.zeros(1, 256)
    z = torch.zeros(1, 64)
    ctx = SafetyContext()
    action = agent.act(h, z, ctx)
    assert action.shape == (cfg.model.action_dim,)


def test_act_emergency_returns_zeros(agent: MouseDroidNavigationAgent, cfg: Settings) -> None:
    h = torch.zeros(1, 256)
    z = torch.zeros(1, 64)
    ctx = SafetyContext(is_emergency=True)
    action = agent.act(h, z, ctx)
    assert (action == 0.0).all()
    assert action.shape == (cfg.model.action_dim,)


def test_act_no_forward_clearance_returns_reverse(agent: MouseDroidNavigationAgent) -> None:
    h = torch.zeros(1, 256)
    z = torch.zeros(1, 64)
    ctx = SafetyContext(forward_clearance_ok=False)
    action = agent.act(h, z, ctx)
    assert action[0].item() == pytest.approx(-0.5)


def test_reset(agent: MouseDroidNavigationAgent) -> None:
    agent.reset()  # Should not raise


def test_act_with_surprise(agent: MouseDroidNavigationAgent) -> None:
    h = torch.zeros(1, 256)
    z = torch.zeros(1, 64)
    ctx = SafetyContext(surprise=5.0)
    action = agent.act(h, z, ctx)
    assert (action >= -1.0).all()
    assert (action <= 1.0).all()


def test_emergency_overrides_no_clearance(agent: MouseDroidNavigationAgent, cfg: Settings) -> None:
    h = torch.zeros(1, 256)
    z = torch.zeros(1, 64)
    ctx = SafetyContext(is_emergency=True, forward_clearance_ok=False)
    action = agent.act(h, z, ctx)
    # Emergency takes priority: zeros
    assert (action == 0.0).all()
