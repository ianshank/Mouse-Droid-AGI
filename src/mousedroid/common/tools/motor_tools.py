"""Motor control MCP tools for the rover platform.

Exposes three tools through the shared
:class:`~mousedroid.common.tools.registry.ToolRegistry`:

* ``set_velocity`` — clamp the requested velocity to
  ``cfg.esp32.max_velocity_mps`` / ``cfg.esp32.max_omega_rads`` and
  dispatch via :class:`~mousedroid.comms.protocol.ESP32CommProtocol`.
* ``read_encoders`` — read-only; returns the latest encoder reading.
* ``emergency_stop`` — issue an e-stop directly on the driver.

Bounds, tool names, and the driver instance are all driven by
:class:`~mousedroid.config.schema.Settings`; nothing is hardcoded.

Safety gating (refusing actuation when the safety monitor is in
emergency state) is enforced upstream in
:class:`~mousedroid.mcp.tool_bridge.MCPToolBridge` based on whether the
tool name appears in :attr:`MCPConfig.actuation_tools`. ``set_velocity``
is part of the default actuation list; ``emergency_stop`` is
intentionally NOT — refusing an e-stop call during an emergency would
be exactly the wrong behaviour. ``read_encoders`` is read-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mousedroid.common.tools.registry import ToolRegistry, ToolSpec
from mousedroid.comms.protocol import ESP32CommProtocol
from mousedroid.config.schema import Settings
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


@dataclass
class MotorToolDeps:
    """Dependencies required to register motor tools.

    Attributes:
        esp32: Concrete ESP32 driver implementing the comm protocol.
        cfg: Root :class:`Settings` (``cfg.esp32`` supplies bounds).
        name_set_velocity: Tool name override. Default ``set_velocity``.
        name_read_encoders: Tool name override. Default
            ``read_encoders``.
        name_emergency_stop: Tool name override. Default
            ``emergency_stop``.
    """

    esp32: ESP32CommProtocol
    cfg: Settings
    name_set_velocity: str = "set_velocity"
    name_read_encoders: str = "read_encoders"
    name_emergency_stop: str = "emergency_stop"


def _clamp(value: float, *, lower: float, upper: float) -> float:
    """Clamp ``value`` into ``[lower, upper]``."""
    return max(lower, min(upper, value))


def register_motor_tools(registry: ToolRegistry, deps: MotorToolDeps) -> None:
    """Register the three motor tools with the shared registry.

    Idempotent: re-registration overwrites previous specs (matching the
    registry's documented behaviour).

    Args:
        registry: Shared registry the bridge consults at startup.
        deps: Driver, settings, and optional name overrides.
    """
    max_v = deps.cfg.esp32.max_velocity_mps
    max_omega = deps.cfg.esp32.max_omega_rads

    async def _set_velocity(vx: float = 0.0, vy: float = 0.0, omega: float = 0.0) -> dict[str, Any]:
        clamped_vx = _clamp(vx, lower=-max_v, upper=max_v)
        clamped_vy = _clamp(vy, lower=-max_v, upper=max_v)
        clamped_omega = _clamp(omega, lower=-max_omega, upper=max_omega)
        await deps.esp32.send_velocity(clamped_vx, clamped_vy, clamped_omega)
        _log.info(
            "motor_tool_set_velocity",
            vx=clamped_vx,
            vy=clamped_vy,
            omega=clamped_omega,
        )
        return {
            "status": "ok",
            "vx": clamped_vx,
            "vy": clamped_vy,
            "omega": clamped_omega,
        }

    async def _read_encoders() -> dict[str, Any]:
        reading = await deps.esp32.read_encoders()
        return {
            "left_velocity_mps": float(reading.left_velocity_mps),
            "right_velocity_mps": float(reading.right_velocity_mps),
            "odometry_x_m": float(reading.odometry_x_m),
            "odometry_y_m": float(reading.odometry_y_m),
            "heading_rad": float(reading.heading_rad),
            "timestamp": float(reading.timestamp),
        }

    async def _emergency_stop() -> dict[str, str]:
        await deps.esp32.emergency_stop()
        _log.info("motor_tool_emergency_stop")
        return {"status": "ok"}

    registry.register(
        ToolSpec(
            deps.name_set_velocity,
            "Set rover velocity (vx, vy, omega) in robot frame, clamped to config bounds",
            _set_velocity,
        )
    )
    registry.register(
        ToolSpec(
            deps.name_read_encoders,
            "Read latest wheel encoder reading and odometry pose",
            _read_encoders,
        )
    )
    registry.register(
        ToolSpec(
            deps.name_emergency_stop,
            "Emergency stop — zero velocity command bypassing the actuation gate",
            _emergency_stop,
        )
    )
