"""Tests for MockOpenClawGateway."""

from __future__ import annotations

import numpy as np
import pytest

from mousedroid.config.schema import OpenClawConfig
from mousedroid.openclaw.mock_gateway import MockOpenClawGateway


@pytest.fixture
def cfg() -> OpenClawConfig:
    return OpenClawConfig()


@pytest.fixture
def gateway(cfg: OpenClawConfig) -> MockOpenClawGateway:
    return MockOpenClawGateway(cfg)


# -- Lifecycle -----------------------------------------------------------------


async def test_start_sets_connected(gateway: MockOpenClawGateway):
    assert gateway.is_connected is False
    await gateway.start()
    assert gateway.is_connected is True
    assert gateway.start_calls == 1


async def test_stop_clears_connected(gateway: MockOpenClawGateway):
    await gateway.start()
    await gateway.stop()
    assert gateway.is_connected is False
    assert gateway.stop_calls == 1


# -- get_action ----------------------------------------------------------------


async def test_default_action_is_none(gateway: MockOpenClawGateway):
    result = await gateway.get_action({"state": "test"})
    assert result is None
    assert len(gateway.action_calls) == 1
    assert gateway.action_calls[0] == {"state": "test"}


async def test_set_action_returns_configured(gateway: MockOpenClawGateway):
    action_result = gateway.make_action(vx=0.5, vy=-0.1, omega=0.2)
    gateway.set_action(action_result)

    result = await gateway.get_action({})
    assert result is not None
    np.testing.assert_allclose(result.action, [0.5, -0.1, 0.2], atol=1e-6)
    assert result.goal_id == "test-goal"
    assert result.confidence == 1.0


async def test_set_action_none_returns_none(gateway: MockOpenClawGateway):
    gateway.set_action(gateway.make_action())
    gateway.set_action(None)

    result = await gateway.get_action({})
    assert result is None


# -- set_goal ------------------------------------------------------------------


async def test_set_goal_records_history(gateway: MockOpenClawGateway):
    await gateway.set_goal("navigate to room 101")
    await gateway.set_goal("stop and wait")
    assert gateway.goal_calls == ["navigate to room 101", "stop and wait"]


# -- Control methods -----------------------------------------------------------


def test_set_connected_override(gateway: MockOpenClawGateway):
    gateway.set_connected(True)
    assert gateway.is_connected is True
    gateway.set_connected(False)
    assert gateway.is_connected is False


def test_make_action_defaults(gateway: MockOpenClawGateway):
    result = gateway.make_action()
    np.testing.assert_array_equal(result.action, [0.0, 0.0, 0.0])
    assert result.goal_id == "test-goal"
    assert result.confidence == 1.0
    assert result.reasoning == "mock_action"


def test_make_action_custom(gateway: MockOpenClawGateway):
    result = gateway.make_action(vx=1.0, vy=-1.0, omega=0.5, goal_id="g2", confidence=0.7)
    np.testing.assert_allclose(result.action, [1.0, -1.0, 0.5], atol=1e-6)
    assert result.goal_id == "g2"
    assert result.confidence == 0.7


def test_reset_clears_all(gateway: MockOpenClawGateway):
    gateway.set_action(gateway.make_action())
    gateway.set_connected(True)
    gateway.start_calls = 5
    gateway.goal_calls.append("test")

    gateway.reset()

    assert gateway.start_calls == 0
    assert gateway.stop_calls == 0
    assert len(gateway.action_calls) == 0
    assert len(gateway.goal_calls) == 0
