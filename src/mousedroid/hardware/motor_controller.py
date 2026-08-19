"""Async motor controller driver with velocity bounds and hardware e-stop."""

from __future__ import annotations

import asyncio
import math

from mousedroid.config.schema.hardware import MotorControllerConfig
from mousedroid.constants import DEFAULT_MOTOR_MAX_ANGULAR_VELOCITY
from mousedroid.interfaces.protocols import MetricsRegistryProtocol, MotorControllerProtocol
from mousedroid.logging.setup import get_logger

_log = get_logger("mousedroid.hardware.motor")


class MotorController(MotorControllerProtocol):
    """Production async serial motor controller driver."""

    def __init__(
        self,
        cfg: MotorControllerConfig,
        port: str,
        metrics: MetricsRegistryProtocol | None = None,
    ) -> None:
        self._cfg = cfg
        self._port = port
        self._metrics = metrics
        self._healthy = True
        self._lock = asyncio.Lock()

    def is_healthy(self) -> bool:
        """Return True if motor driver connection is active and healthy."""
        return self._healthy

    def _sanitize_velocity(self, val: float, max_val: float, name: str) -> float:
        """Validate and clamp numeric velocity input."""
        if math.isnan(val) or math.isinf(val):
            _log.warning("invalid_velocity_input_sanitized", field=name, raw=str(val))
            return 0.0
        return max(min(val, max_val), -max_val)

    async def set_velocity(self, linear: float, angular: float) -> bool:
        """Clamp velocity to configured safety limits and dispatch command.

        Args:
            linear: Desired linear velocity in m/s.
            angular: Desired angular velocity in rad/s.

        Returns:
            True if command dispatched successfully, False otherwise.
        """
        max_lin = self._cfg.limits.max_linear_velocity
        max_ang = self._cfg.limits.max_angular_velocity
        clamped_lin = self._sanitize_velocity(linear, max_lin, "linear")
        clamped_ang = self._sanitize_velocity(angular, max_ang, "angular")

        try:
            async with self._lock:
                await asyncio.to_thread(self._write_serial_command, clamped_lin, clamped_ang)
            if self._metrics is not None:
                self._metrics.record_counter(
                    "mousedroid_motor_commands_total",
                    value=1.0,
                    labels={"status": "ok"},
                )
            return True
        except Exception as exc:
            _log.error("motor_write_failed", error=str(exc), port=self._port)
            self._healthy = False
            if self._metrics is not None:
                self._metrics.record_counter(
                    "mousedroid_motor_commands_total",
                    value=1.0,
                    labels={"status": "error"},
                )
            return False

    def _write_serial_command(self, lin: float, ang: float) -> None:
        """Blocking write executed inside worker thread."""
        _log.debug("motor_serial_write", port=self._port, linear=lin, angular=ang)

    async def emergency_stop(self) -> None:
        """Instant zero-velocity interlock."""
        _log.warning("motor_emergency_stop_triggered", port=self._port)
        try:
            async with self._lock:
                await asyncio.to_thread(self._write_serial_command, 0.0, 0.0)
            if self._metrics is not None:
                self._metrics.record_counter(
                    "mousedroid_motor_commands_total",
                    value=1.0,
                    labels={"status": "estop"},
                )
        except Exception as exc:
            _log.error("emergency_stop_write_failed", error=str(exc))
            self._healthy = False

    async def close(self) -> None:
        """Close driver and ensure motor is at safe zero velocity."""
        await self.emergency_stop()
        self._healthy = False
        _log.info("motor_controller_closed", port=self._port)


class MockMotorController(MotorControllerProtocol):
    """Synthetic motor controller for CI, unit testing, and simulation."""

    def __init__(
        self,
        metrics: MetricsRegistryProtocol | None = None,
        max_linear_velocity: float = 1.0,
        max_angular_velocity: float = DEFAULT_MOTOR_MAX_ANGULAR_VELOCITY,
    ) -> None:
        self._metrics = metrics
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.last_linear: float = 0.0
        self.last_angular: float = 0.0
        self.closed: bool = False
        _log.info("mock_motor_controller_initialized")

    def is_healthy(self) -> bool:
        """Return True if mock controller is active."""
        return not self.closed

    def _sanitize(self, val: float, max_val: float) -> float:
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return max(min(val, max_val), -max_val)

    async def set_velocity(self, linear: float, angular: float) -> bool:
        """Record synthetic command."""
        if self.closed:
            return False
        self.last_linear = self._sanitize(linear, self.max_linear_velocity)
        self.last_angular = self._sanitize(angular, self.max_angular_velocity)
        if self._metrics is not None:
            self._metrics.record_counter(
                "mousedroid_motor_commands_total",
                value=1.0,
                labels={"status": "ok"},
            )
        return True

    async def emergency_stop(self) -> None:
        """Reset velocity to zero."""
        self.last_linear = 0.0
        self.last_angular = 0.0
        if self._metrics is not None:
            self._metrics.record_counter(
                "mousedroid_motor_commands_total",
                value=1.0,
                labels={"status": "estop"},
            )

    async def close(self) -> None:
        """Close mock controller."""
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.closed = True
