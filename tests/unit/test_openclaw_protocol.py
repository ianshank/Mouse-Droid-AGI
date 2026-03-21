"""Tests for OpenClaw protocol and data types."""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np

from mousedroid.openclaw.mock_gateway import MockOpenClawGateway
from mousedroid.openclaw.protocol import OpenClawActionResult, OpenClawProtocol

# -- Protocol structural checks -----------------------------------------------


def test_openclaw_protocol_is_protocol():
    assert issubclass(OpenClawProtocol, Protocol)


def test_openclaw_protocol_is_runtime_checkable():
    from mousedroid.config.schema import OpenClawConfig

    mock = MockOpenClawGateway(OpenClawConfig())
    assert isinstance(mock, OpenClawProtocol)


# -- OpenClawActionResult frozen dataclass ------------------------------------


def test_action_result_fields():
    action = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    result = OpenClawActionResult(
        action=action,
        goal_id="g-1",
        reasoning="test",
        confidence=0.95,
        timestamp=100.0,
    )
    np.testing.assert_array_equal(result.action, action)
    assert result.goal_id == "g-1"
    assert result.reasoning == "test"
    assert result.confidence == 0.95
    assert result.timestamp == 100.0


def test_action_result_is_frozen():
    result = OpenClawActionResult(
        action=np.zeros(3, dtype=np.float32),
        goal_id="g-1",
        reasoning="",
        confidence=1.0,
        timestamp=0.0,
    )
    import dataclasses

    import pytest

    assert dataclasses.is_dataclass(result)
    with pytest.raises(AttributeError):
        result.goal_id = "new"  # type: ignore[misc]


def test_action_result_is_stale():
    # Very old timestamp → stale
    result = OpenClawActionResult(
        action=np.zeros(3, dtype=np.float32),
        goal_id="",
        reasoning="",
        confidence=1.0,
        timestamp=time.monotonic() - 10.0,  # 10 seconds ago
    )
    assert result.is_stale(max_age_ms=50.0) is True


def test_action_result_is_fresh():
    result = OpenClawActionResult(
        action=np.zeros(3, dtype=np.float32),
        goal_id="",
        reasoning="",
        confidence=1.0,
        timestamp=time.monotonic(),
    )
    assert result.is_stale(max_age_ms=5000.0) is False
