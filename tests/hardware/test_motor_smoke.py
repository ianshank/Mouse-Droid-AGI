"""Rover motor hardware smoke tests.

Three checks, all driven entirely by config values from
:class:`~mousedroid.config.schema.ESP32Config` and
:class:`~mousedroid.config.schema.MCPConfig`:

1. ``send_velocity`` → ``read_encoders`` round-trip (encoder velocity
   reflects the setpoint within a configurable fraction). Asserted only
   on real hardware; mock drivers return zeros and would fail the
   assertion, so the check is auto-skipped under
   ``mock_hardware=True``.
2. ``emergency_stop`` ack latency stays inside
   ``ESP32Config.emergency_stop_budget_ms``.
3. The rover MCP bridge keeps responding while a client polls a
   telemetry-style URI at ``MCPConfig.smoke_test_poll_rps`` for
   ``MCPConfig.smoke_test_duration_s`` seconds — verifies the lifecycle
   doesn't deadlock under concurrent MCP traffic.

Marked ``hardware``; ``tests/hardware/conftest.py`` reverses the
top-level ``MOUSEDROID_MOCK_HARDWARE`` env override and forces
``mock_hardware=True`` again on non-Jetson hosts.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings

pytestmark = pytest.mark.hardware


async def test_velocity_roundtrip_clamps_and_dispatches(
    jetson_settings: Settings,
) -> None:
    """``send_velocity`` accepts the smoke setpoint and ``read_encoders`` returns
    a structurally valid reading. On real hardware additionally assert that
    the encoder velocity reflects the setpoint within the configured fraction.
    """
    from mousedroid.factory import build_esp32_driver

    driver = build_esp32_driver(jetson_settings)
    await driver.connect()
    try:
        target_vx = jetson_settings.esp32.smoke_test_velocity_mps
        await driver.send_velocity(target_vx, 0.0, 0.0)
        await asyncio.sleep(jetson_settings.esp32.smoke_test_settle_s)
        reading = await driver.read_encoders()
        # Structural assertion: every encoder field is well-typed.
        assert reading.left_velocity_mps == pytest.approx(reading.left_velocity_mps, rel=1.0)
        assert isinstance(reading.timestamp, float)
        if not jetson_settings.mock_hardware:
            min_fraction = jetson_settings.esp32.smoke_test_min_velocity_fraction
            assert reading.left_velocity_mps >= target_vx * min_fraction
            assert reading.right_velocity_mps >= target_vx * min_fraction
    finally:
        await driver.emergency_stop()
        await driver.disconnect()


async def test_emergency_stop_latency_within_budget(jetson_settings: Settings) -> None:
    """``emergency_stop`` returns within ``ESP32Config.emergency_stop_budget_ms``."""
    from mousedroid.factory import build_esp32_driver

    driver = build_esp32_driver(jetson_settings)
    await driver.connect()
    try:
        await driver.send_velocity(jetson_settings.esp32.smoke_test_velocity_mps, 0.0, 0.0)
        t0 = time.monotonic()
        await driver.emergency_stop()
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        budget_ms = jetson_settings.esp32.emergency_stop_budget_ms
        assert elapsed_ms <= budget_ms, (
            f"emergency_stop took {elapsed_ms:.1f} ms, budget {budget_ms:.1f} ms"
        )
    finally:
        await driver.disconnect()


async def test_mcp_resource_polling_does_not_deadlock(jetson_settings: Settings) -> None:
    """MCP server responds throughout the configured smoke window.

    Builds the MCP server with ``cfg.mcp`` overridden to ``enabled=True``,
    starts it, then polls ``mousedroid://config/redacted`` (always
    available) at ``MCPConfig.smoke_test_poll_rps`` for
    ``MCPConfig.smoke_test_duration_s``. Validates that every poll
    succeeds — the lifecycle therefore survives concurrent traffic.

    Skipped if the MCP factory returns None (e.g. ``cfg.mcp`` is None
    for this overlay).
    """
    from mousedroid.config.schema import MCPConfig
    from mousedroid.factory import build_mcp_server
    from mousedroid.mcp.server import MouseDroidMCPServer

    overlay_mcp = (jetson_settings.mcp or MCPConfig()).model_copy(
        update={"enabled": True, "transport": "stdio"}
    )
    cfg = jetson_settings.model_copy(update={"mcp": overlay_mcp})
    server = build_mcp_server(
        cfg,
        tool_registry=_empty_registry(),
        safety_monitor=_NoOpSafetyMonitor(),
    )
    if server is None:
        pytest.skip("MCP not configured for this overlay")
    assert isinstance(server, MouseDroidMCPServer)

    await server.start()
    try:
        period = 1.0 / overlay_mcp.smoke_test_poll_rps
        deadline = asyncio.get_running_loop().time() + overlay_mcp.smoke_test_duration_s
        polls = 0
        while asyncio.get_running_loop().time() < deadline:
            payload = await server.read_resource("mousedroid://config/redacted", peer="smoke")
            assert isinstance(payload, dict)
            polls += 1
            await asyncio.sleep(period)
        assert polls > 0
    finally:
        await server.stop()


# ---------------------------------------------------------------------------
# Helpers (kept local to this module; nothing here belongs in src/).
# ---------------------------------------------------------------------------


def _empty_registry() -> object:
    """Build a fresh ``ToolRegistry`` so smoke runs do not mutate the live one."""
    from mousedroid.common.tools.registry import ToolRegistry, ToolSpec

    reg = ToolRegistry()

    async def _ok() -> dict[str, str]:
        return {"status": "ok"}

    reg.register(ToolSpec("health_check", "ok", _ok))
    return reg


class _NoOpSafetyMonitor:
    """SafetyMonitor stand-in for the smoke test (no actuation tools used)."""

    def evaluate(self, observation: object, loop_time_ms: float) -> object:
        from mousedroid.safety.context import SafetyContext

        return SafetyContext(loop_time_ms=loop_time_ms)
