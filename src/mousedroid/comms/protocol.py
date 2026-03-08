"""ESP32 communication protocol and data types.

Defines the ``ESP32CommProtocol`` interface and ``EncoderReading`` dataclass.
Both serial and WiFi drivers implement this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class EncoderReading:
    """Wheel encoder data from ESP32.

    All velocities in robot frame (m/s). Odometry in metres from session start.
    """

    left_velocity_mps: float = 0.0
    right_velocity_mps: float = 0.0
    odometry_x_m: float = 0.0
    odometry_y_m: float = 0.0
    heading_rad: float = 0.0
    timestamp: float = 0.0


@runtime_checkable
class ESP32CommProtocol(Protocol):
    """Interface for ESP32 Wave Rover communication drivers.

    Both ``SerialESP32Driver`` and ``WiFiESP32Driver`` implement this protocol.
    Factory selects based on ``cfg.esp32.protocol``.
    """

    async def connect(self) -> None:
        """Establish connection to ESP32."""
        ...

    async def send_velocity(self, vx: float, vy: float, omega: float) -> None:
        """Send velocity setpoint to motor controller.

        Args:
            vx: Forward velocity in m/s (robot frame).
            vy: Lateral velocity in m/s (robot frame, mecanum only).
            omega: Angular velocity in rad/s.
        """
        ...

    async def read_encoders(self) -> EncoderReading:
        """Read wheel encoder data from ESP32.

        Returns:
            ``EncoderReading`` with wheel velocities and odometry.
        """
        ...

    async def get_battery_voltage(self) -> float:
        """Read battery voltage from ESP32 ADC.

        Returns:
            Battery voltage in volts.
        """
        ...

    async def emergency_stop(self) -> None:
        """Send emergency stop command (zero velocity, high priority)."""
        ...

    async def disconnect(self) -> None:
        """Close connection to ESP32."""
        ...
