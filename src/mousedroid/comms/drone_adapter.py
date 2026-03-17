"""Drone motor adapter — wraps FlightControllerProtocol as MotorControlProtocol.

Translates the platform-agnostic ``send_command`` / ``read_state`` API
into flight-controller-specific ``send_velocity_ned`` / telemetry calls.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.comms.flight_protocol import FlightControllerProtocol

_log = get_logger(__name__)


class DroneMotorAdapter:
    """Adapt ``FlightControllerProtocol`` to ``MotorControlProtocol``.

    Args:
        flight_controller: Underlying flight controller driver.
    """

    def __init__(self, flight_controller: FlightControllerProtocol) -> None:
        self._fc = flight_controller
        _log.info("drone_motor_adapter_init")

    # -- MotorControlProtocol ----------------------------------------------

    async def connect(self) -> None:
        """Connect to the flight controller."""
        await self._fc.connect()

    async def send_command(self, command: NDArray[np.floating[Any]]) -> None:
        """Extract ``[vx, vy, vz, yaw_rate]`` and forward to FC as NED velocity.

        Args:
            command: Scaled command array. Elements map to
                     ``[vn, ve, vd, yaw_rate]``.
        """
        vn = float(command[0]) if command.size > 0 else 0.0
        ve = float(command[1]) if command.size > 1 else 0.0
        vd = float(command[2]) if command.size > 2 else 0.0
        yaw_rate = float(command[3]) if command.size > 3 else 0.0
        await self._fc.send_velocity_ned(vn, ve, vd, yaw_rate)
        _log.debug("drone_command_sent", vn=vn, ve=ve, vd=vd, yaw_rate=yaw_rate)

    async def read_state(self) -> NDArray[np.float32]:
        """Read IMU, altitude, battery, and arm state.

        Returns:
            7-element state array ``[vx, vy, vz, yaw_rate, altitude, battery_v, armed]``.
        """
        imu_data, altitude, battery_v = await asyncio.gather(
            self._fc.get_imu_data(),
            self._fc.get_altitude_m(),
            self._fc.get_battery_voltage(),
        )
        # IMU data: [ax, ay, az, gx, gy, gz] — use gyro rates as velocity proxy
        vx = float(imu_data[3]) if imu_data.size > 3 else 0.0
        vy = float(imu_data[4]) if imu_data.size > 4 else 0.0
        vz = float(imu_data[5]) if imu_data.size > 5 else 0.0
        armed_flag = 1.0 if self._fc.armed else 0.0

        state = np.array(
            [vx, vy, vz, 0.0, altitude, battery_v, armed_flag],
            dtype=np.float32,
        )
        return state

    async def get_battery_voltage(self) -> float:
        """Read battery voltage from flight controller.

        Returns:
            Battery voltage in volts.
        """
        return await self._fc.get_battery_voltage()

    async def emergency_stop(self) -> None:
        """Emergency stop — kill motors."""
        await self._fc.emergency_stop()
        _log.warning("drone_emergency_stop")

    async def disconnect(self) -> None:
        """Disconnect from flight controller."""
        await self._fc.disconnect()

    @property
    def platform_type(self) -> str:
        """Return ``'drone'``."""
        return "drone"
