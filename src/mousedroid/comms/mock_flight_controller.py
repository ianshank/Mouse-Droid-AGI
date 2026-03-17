"""Mock flight controller for testing and simulation.

Implements ``FlightControllerProtocol`` with no real hardware dependencies.
Follows the same pattern as ``MockESP32Driver``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import FlightControllerConfig

_log = get_logger(__name__)

_DEFAULT_MOCK_BATTERY_V: float = 16.8
"""Default mock battery voltage (nominal 4S LiPo)."""

_DEFAULT_MOCK_ALTITUDE_M: float = 10.0
"""Default mock hover altitude."""


class MockFlightController:
    """Mock flight controller for offline testing.

    All commands are logged but have no side effects beyond internal state.

    Args:
        cfg: Flight controller configuration.
    """

    def __init__(self, cfg: FlightControllerConfig) -> None:
        self._cfg = cfg
        self._connected: bool = False
        self._armed: bool = False
        self._flight_mode: str = "STABILIZE"
        self._altitude_m: float = 0.0
        self._last_velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._battery_voltage: float = _DEFAULT_MOCK_BATTERY_V
        self._gps_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
        _log.info("mock_flight_controller_init", system_id=cfg.system_id)

    # -- Lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        """Simulate connection to flight controller."""
        self._connected = True
        _log.info("mock_fc_connected", port=self._cfg.serial_port)

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        self._connected = False
        self._armed = False
        _log.info("mock_fc_disconnected")

    # -- Arming ------------------------------------------------------------

    async def arm(self) -> None:
        """Simulate motor arming."""
        self._armed = True
        _log.info("mock_fc_armed")

    async def disarm(self) -> None:
        """Simulate motor disarming."""
        self._armed = False
        self._last_velocity = (0.0, 0.0, 0.0, 0.0)
        _log.info("mock_fc_disarmed")

    # -- Flight commands ---------------------------------------------------

    async def takeoff(self, altitude_m: float) -> None:
        """Simulate takeoff to target altitude.

        Args:
            altitude_m: Target altitude AGL in metres.
        """
        self._altitude_m = altitude_m
        self._flight_mode = "GUIDED"
        _log.info("mock_fc_takeoff", target_altitude_m=altitude_m)

    async def land(self) -> None:
        """Simulate landing."""
        self._altitude_m = 0.0
        self._flight_mode = "LAND"
        _log.info("mock_fc_land")

    async def send_velocity_ned(
        self, vn: float, ve: float, vd: float, yaw_rate: float
    ) -> None:
        """Store and log velocity command in NED frame.

        Args:
            vn: North velocity (m/s).
            ve: East velocity (m/s).
            vd: Down velocity (m/s, negative = up).
            yaw_rate: Yaw rate (rad/s).
        """
        self._last_velocity = (vn, ve, vd, yaw_rate)
        _log.debug("mock_fc_velocity_sent", vn=vn, ve=ve, vd=vd, yaw_rate=yaw_rate)

    # -- Telemetry ---------------------------------------------------------

    async def get_altitude_m(self) -> float:
        """Return mock altitude AGL.

        Returns:
            Altitude in metres.
        """
        return self._altitude_m

    async def get_gps_position(self) -> tuple[float, float, float]:
        """Return mock GPS position.

        Returns:
            Tuple of ``(lat, lon, alt_msl)``.
        """
        return self._gps_position

    async def get_imu_data(self) -> NDArray[np.float32]:
        """Return mock IMU data (zeros = level, stationary).

        Returns:
            6-element array ``[ax, ay, az, gx, gy, gz]``.
        """
        return np.zeros(6, dtype=np.float32)

    async def get_battery_voltage(self) -> float:
        """Return mock battery voltage.

        Returns:
            Battery voltage in volts.
        """
        return self._battery_voltage

    # -- Mode control ------------------------------------------------------

    async def set_flight_mode(self, mode: str) -> None:
        """Set mock flight mode.

        Args:
            mode: Flight mode string.
        """
        self._flight_mode = mode
        _log.info("mock_fc_mode_set", mode=mode)

    async def return_to_launch(self) -> None:
        """Simulate RTL mode."""
        self._flight_mode = "RTL"
        _log.info("mock_fc_rtl")

    async def emergency_stop(self) -> None:
        """Simulate emergency motor kill."""
        self._armed = False
        self._last_velocity = (0.0, 0.0, 0.0, 0.0)
        _log.warning("mock_fc_emergency_stop")

    # -- Properties --------------------------------------------------------

    @property
    def armed(self) -> bool:
        """Whether the motors are currently armed."""
        return self._armed

    @property
    def flight_mode(self) -> str:
        """Current flight mode."""
        return self._flight_mode
