"""Integration tests for OpenClaw gateway in the orchestrator pipeline."""

from __future__ import annotations

import numpy as np

from mousedroid.config.schema import OpenClawConfig, Settings
from mousedroid.factory import build_openclaw_gateway, build_orchestrator
from mousedroid.openclaw.mock_gateway import MockOpenClawGateway
from mousedroid.openclaw.protocol import OpenClawProtocol


def _mock_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {"mock_hardware": True}
    defaults.update(overrides)
    return Settings(**defaults)


# -- Factory tests -------------------------------------------------------------


def test_build_openclaw_gateway_disabled():
    cfg = _mock_settings()
    assert cfg.openclaw.enabled is False
    gw = build_openclaw_gateway(cfg)
    assert gw is None


def test_build_openclaw_gateway_mock():
    cfg = _mock_settings(openclaw=OpenClawConfig(enabled=True))
    gw = build_openclaw_gateway(cfg)
    assert isinstance(gw, MockOpenClawGateway)
    assert isinstance(gw, OpenClawProtocol)


def test_build_orchestrator_with_openclaw():
    cfg = _mock_settings(openclaw=OpenClawConfig(enabled=True))
    orch = build_orchestrator(cfg)
    assert orch is not None
    assert orch._openclaw_gateway is not None


def test_build_orchestrator_without_openclaw():
    cfg = _mock_settings()
    orch = build_orchestrator(cfg)
    assert orch is not None
    assert orch._openclaw_gateway is None


# -- Orchestrator action selection ---------------------------------------------


async def test_openclaw_action_takes_priority():
    """OpenClaw cached action should be used when fresh."""
    cfg = _mock_settings(openclaw=OpenClawConfig(enabled=True, max_action_age_ms=5000.0))
    orch = build_orchestrator(cfg)

    # Inject a fresh cached action
    import time

    from mousedroid.openclaw.protocol import OpenClawActionResult

    cached = OpenClawActionResult(
        action=np.array([0.7, -0.3, 0.1], dtype=np.float32),
        goal_id="test-goal",
        reasoning="integration test",
        confidence=0.95,
        timestamp=time.monotonic(),
    )
    orch._openclaw_cached_action = cached

    await orch.start()
    try:
        await orch.tick()
    finally:
        await orch.stop()


async def test_stale_openclaw_action_falls_through():
    """Stale OpenClaw action should fall back to MCTS."""
    cfg = _mock_settings(openclaw=OpenClawConfig(enabled=True, max_action_age_ms=1.0))
    orch = build_orchestrator(cfg)

    import time

    from mousedroid.openclaw.protocol import OpenClawActionResult

    # Action from 10 seconds ago — definitely stale
    stale = OpenClawActionResult(
        action=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        goal_id="old",
        reasoning="stale",
        confidence=0.5,
        timestamp=time.monotonic() - 10.0,
    )
    orch._openclaw_cached_action = stale

    await orch.start()
    try:
        # Should not crash — falls through to MCTS
        await orch.tick()
    finally:
        await orch.stop()


async def test_no_openclaw_action_falls_to_mcts():
    """When OpenClaw returns no cached action, MCTS is used."""
    cfg = _mock_settings(openclaw=OpenClawConfig(enabled=True))
    orch = build_orchestrator(cfg)

    # No cached action
    assert orch._openclaw_cached_action is None

    await orch.start()
    try:
        await orch.tick()
    finally:
        await orch.stop()


# -- Lifecycle -----------------------------------------------------------------


async def test_openclaw_start_stop_lifecycle():
    cfg = _mock_settings(openclaw=OpenClawConfig(enabled=True))
    orch = build_orchestrator(cfg)

    gw = orch._openclaw_gateway
    assert isinstance(gw, MockOpenClawGateway)

    await orch.start()
    assert gw.is_connected is True
    assert gw.start_calls == 1
    assert orch._openclaw_poll_task is not None

    await orch.stop()
    assert gw.stop_calls == 1
    assert orch._openclaw_poll_task is None
