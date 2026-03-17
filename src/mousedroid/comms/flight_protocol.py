"""Flight controller protocol — interface for drone autopilots.

Defines the ``FlightControllerProtocol`` for MAVLink-based flight
controllers (PX4, ArduPilot). Mock and real implementations conform
to this protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class FlightControllerProtocol(Protocol):
    """Interface for drone flight controller communication.

    Implementations include ``MockFlightController`` (testing) and
    future ``MAVLinkFlightController`` (real hardware).
    """

    async def connect(self) -> None:
        """Establish connection to the flight controller."""
        ...

    async def disconnect(self) -> None:
        """Close connection to the flight controller."""
        ...

    async def arm(self) -> None:
        """Arm the motors for flight."""
        ...

    async def disarm(self) -> None:
        """Disarm the motors."""
        ...

    async def takeoff(self, altitude_m: float) -> None:
        """Command autonomous takeoff to the specified altitude.

        Args:
            altitude_m: Target altitude above ground in metres.
        """
        ...

    async def land(self) -> None:
        """Command autonomous landing at current position."""
        ...

    async def send_velocity_ned(
        self, vn: float, ve: float, vd: float, yaw_rate: float
    ) -> None:
        """Send velocity setpoint in NED frame.

        Args:
            vn: North velocity (m/s).
            ve: East velocity (m/s).
            vd: Down velocity (m/s, negative = up).
            yaw_rate: Yaw rate (rad/s).
        """
        ...

    async def get_altitude_m(self) -> float:
        """Read current altitude above ground level.

        Returns:
            Altitude in metres AGL.
        """
        ...

    async def get_gps_position(self) -> tuple[float, float, float]:
        """Read current GPS position.

        Returns:
            Tuple of ``(latitude_deg, longitude_deg, altitude_msl_m)``.
        """
        ...

    async def get_imu_data(self) -> NDArray[np.float32]:
        """Read IMU data (accelerometer + gyroscope).

        Returns:
            6-element array ``[ax, ay, az, gx, gy, gz]``.
        """
        ...

    async def get_battery_voltage(self) -> float:
        """Read battery voltage.

        Returns:
            Battery voltage in volts.
        """
        ...

    async def set_flight_mode(self, mode: str) -> None:
        """Set the autopilot flight mode.

        Args:
            mode: Flight mode string (e.g. ``'GUIDED'``, ``'LOITER'``).
        """
        ...

    async def return_to_launch(self) -> None:
        """Command return-to-launch (RTL) mode."""
        ...

    async def emergency_stop(self) -> None:
        """Emergency motor kill — disarm immediately."""
        ...

    @property
    def armed(self) -> bool:
        """Whether the motors are currently armed."""
        ...

    @property
    def flight_mode(self) -> str:
        """Current autopilot flight mode."""
        ...
