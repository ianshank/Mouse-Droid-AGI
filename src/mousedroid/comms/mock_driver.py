"""Mock ESP32 communication driver for testing and simulation.

Implements ``ESP32CommProtocol`` with no real hardware dependencies.
``MockESP32Driver`` is a standalone class — it deliberately does NOT
inherit ``BaseESP32Driver`` (no transport layer to share), so we wire
the shared ``log_command_dispatch`` helper in explicitly to preserve the
"every driver emits the same ``command_dispatch`` event shape" guarantee
operators rely on when grepping smoke logs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.comms.base_driver import log_command_dispatch
from mousedroid.comms.protocol import EncoderReading
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ESP32Config

_log = get_logger(__name__)


class MockESP32Driver:
    """Mock driver implementing ``ESP32CommProtocol`` for offline testing.

    All commands are logged but have no side effects beyond internal state.
    """

    def __init__(self, cfg: ESP32Config) -> None:
        """Initialise mock driver from config.

        Args:
            cfg: ESP32 communication configuration.
        """
        self._cfg = cfg
        self._connected: bool = False
        self._last_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._battery_voltage: float = cfg.mock_battery_v

    async def connect(self) -> None:
        """Simulate establishing connection to ESP32."""
        self._connected = True
        _log.info("mock_esp32_connected", port=self._cfg.serial_port)

    async def disconnect(self) -> None:
        """Simulate closing connection to ESP32."""
        self._connected = False
        _log.info("mock_esp32_disconnected")

    async def send_velocity(self, vx: float, vy: float, omega: float) -> None:
        """Store and log velocity command.

        Emits the uniform ``command_dispatch`` INFO event used by smoke-
        time triage AND the legacy per-driver ``mock_velocity_sent`` DEBUG
        event for backwards compatibility with existing log-greps.

        Args:
            vx: Forward velocity in m/s.
            vy: Lateral velocity in m/s.
            omega: Angular velocity in rad/s.
        """
        self._last_velocity = (vx, vy, omega)
        log_command_dispatch(
            driver_name=type(self).__name__,
            vx=vx,
            vy=vy,
            omega=omega,
        )
        _log.debug("mock_velocity_sent", vx=vx, vy=vy, omega=omega)

    async def read_encoders(self) -> EncoderReading:
        """Return default encoder reading.

        Returns:
            ``EncoderReading`` with default zero values.
        """
        return EncoderReading()

    async def get_battery_voltage(self) -> float:
        """Return configurable battery voltage.

        Returns:
            Battery voltage in volts (default 12.0).
        """
        return self._battery_voltage

    async def emergency_stop(self) -> None:
        """Log emergency stop and zero velocity."""
        self._last_velocity = (0.0, 0.0, 0.0)
        _log.warning("mock_emergency_stop")
