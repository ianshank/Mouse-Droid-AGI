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

    Attitude fields (``roll_rad`` / ``pitch_rad`` / ``yaw_rad``) are the
    stock-firmware IMU on ``FEEDBACK_BASE_INFO`` (keys ``r`` / ``p`` /
    ``y``). They stay 0.0 with ``imu_valid=False`` on the legacy protocol
    so existing YAML and the default ``command_set="legacy"`` path remain
    byte-identical. ``heading_rad`` is odometry heading and is *not* the
    same quantity as IMU yaw — :meth:`heading_for_motor` is the sensing
    seam that picks which one fills the 4-float motor observation.
    """

    left_velocity_mps: float = 0.0
    right_velocity_mps: float = 0.0
    odometry_x_m: float = 0.0
    odometry_y_m: float = 0.0
    heading_rad: float = 0.0
    timestamp: float = 0.0
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0
    imu_valid: bool = False

    def heading_for_motor(self) -> float:
        """Heading consumed by the 4-float motor observation vector.

        Stock WAVE ROVER frames carry IMU yaw and no wheel odometry.
        Legacy frames carry odometry heading and no IMU. Prefer IMU yaw
        when ``imu_valid`` so the encoder-less chassis is not stuck at
        heading 0.0; otherwise keep the historical ``heading_rad`` slot.

        Returns:
            Yaw in radians when a stock T=1001 frame was parsed, else
            odometry heading.
        """
        if self.imu_valid:
            return self.yaw_rad
        return self.heading_rad


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
