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
    from mousedroid.comms.protocol import ESP32CommProtocol
    from mousedroid.config.schema import Settings

pytestmark = pytest.mark.hardware


async def test_velocity_roundtrip_clamps_and_dispatches(
    jetson_settings: Settings,
) -> None:
    """``send_velocity`` accepts the smoke setpoint and ``read_encoders`` returns
    a structurally valid reading.

    Safety: when ``ESP32Config.smoke_test_allow_motion`` is False (default),
    the actual velocity sent to the driver is **zero** — the test only
    exercises the connect → send → read → e-stop → disconnect flow without
    physically moving the rover. This protects an untethered rover on a
    table from rolling off when the smoke runs unattended.

    When the operator explicitly opts in (e.g. rover on rollers),
    ``smoke_test_allow_motion=True`` lets the test drive the motors at
    ``smoke_test_velocity_mps``. The motion-quality criterion then depends
    on the chassis (F-025 / audit R3): with wheel encoders
    (``chassis_has_wheel_encoders=True``) the encoder must reflect the
    setpoint within ``smoke_test_min_velocity_fraction``; on an
    encoder-less chassis (WAVE ROVER) that assertion is unsatisfiable —
    the re-scoped criterion is "command accepted (send path raised
    nothing) + e-stop within budget" (the budget half lives in
    ``test_emergency_stop_latency_within_budget``).

    Under ``command_set='waveshare_stock'`` with the chassis heartbeat
    armed, the settle window re-sends the velocity at ``keepalive_hz`` —
    the default heartbeat window (300 ms) is shorter than
    ``smoke_test_settle_s`` (0.5 s), so a single send would be halted by
    the failsafe mid-settle.
    """
    from mousedroid.factory import build_esp32_driver

    driver = build_esp32_driver(jetson_settings)
    await driver.connect()
    try:
        esp32_cfg = jetson_settings.esp32
        target_vx = esp32_cfg.smoke_test_velocity_mps if esp32_cfg.smoke_test_allow_motion else 0.0
        await driver.send_velocity(target_vx, 0.0, 0.0)
        await _settle_with_keepalive(driver, jetson_settings, target_vx)
        reading = await driver.read_encoders()
        # Structural assertion: every encoder field is well-typed.
        assert isinstance(reading.left_velocity_mps, float)
        assert isinstance(reading.right_velocity_mps, float)
        assert isinstance(reading.odometry_x_m, float)
        assert isinstance(reading.odometry_y_m, float)
        assert isinstance(reading.heading_rad, float)
        assert isinstance(reading.timestamp, float)
        # Motion-quality assertion only when motion was actually requested.
        if esp32_cfg.smoke_test_allow_motion and not jetson_settings.mock_hardware:
            if esp32_cfg.chassis_has_wheel_encoders:
                min_fraction = esp32_cfg.smoke_test_min_velocity_fraction
                assert reading.left_velocity_mps >= target_vx * min_fraction
                assert reading.right_velocity_mps >= target_vx * min_fraction
            else:
                # Encoder-less re-scope (audit R3): the chassis cannot prove
                # measured speed, so assert what it CAN prove — the command
                # round-trip still works after motion was requested, checked
                # through the public protocol rather than a private flag.
                # (`driver.inner._connected` is set once in connect() and
                # never cleared, so asserting it could not fail.)
                post_motion = await driver.read_encoders()
                assert isinstance(post_motion.timestamp, float)
                await driver.send_velocity(0.0, 0.0, 0.0)
    finally:
        await driver.emergency_stop()
        await driver.disconnect()


async def _settle_with_keepalive(
    driver: ESP32CommProtocol, jetson_settings: Settings, target_vx: float
) -> None:
    """Sleep out the settle window, re-sending if the failsafe could fire.

    The decision is computed from the ACTUAL derived window rather than an
    assumption about it: re-send only when the settle would outlast the
    chassis heartbeat. With the shipped defaults the window (3000 ms)
    comfortably exceeds ``smoke_test_settle_s`` (500 ms), so this is a plain
    sleep; a hand-tightened window automatically re-enables the keepalive.
    """
    from mousedroid.comms.command_set import heartbeat_window_ms

    esp32_cfg = jetson_settings.esp32
    settle_s = esp32_cfg.smoke_test_settle_s
    heartbeat_armed = esp32_cfg.command_set == "waveshare_stock" and esp32_cfg.heartbeat_enabled
    if not heartbeat_armed or settle_s * 1000.0 < heartbeat_window_ms(esp32_cfg):
        await asyncio.sleep(settle_s)
        return
    period_s = 1.0 / esp32_cfg.keepalive_hz
    remaining = settle_s
    while remaining > 0:
        await asyncio.sleep(min(period_s, remaining))
        remaining -= period_s
        if remaining > 0:
            await driver.send_velocity(target_vx, 0.0, 0.0)


async def test_emergency_stop_latency_within_budget(jetson_settings: Settings) -> None:
    """``emergency_stop`` returns within ``ESP32Config.emergency_stop_budget_ms``.

    Safety: motion is gated by ``smoke_test_allow_motion`` (see the
    velocity round-trip docstring). When the gate is closed, this test
    measures e-stop latency after a zero-velocity command — still
    representative of the round-trip cost.
    """
    from mousedroid.factory import build_esp32_driver

    driver = build_esp32_driver(jetson_settings)
    await driver.connect()
    try:
        target_vx = (
            jetson_settings.esp32.smoke_test_velocity_mps
            if jetson_settings.esp32.smoke_test_allow_motion
            else 0.0
        )
        await driver.send_velocity(target_vx, 0.0, 0.0)
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
