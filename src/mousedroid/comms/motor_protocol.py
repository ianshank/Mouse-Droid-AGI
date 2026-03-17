"""Motor control protocol — platform-agnostic abstraction over actuators.

Both ground (ESP32 Wave Rover) and aerial (MAVLink flight controller)
platforms implement this protocol. The orchestrator depends only on
``MotorControlProtocol``, never on platform-specific drivers directly.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class MotorControlProtocol(Protocol):
    """Platform-agnostic interface for all motor/actuator controllers.

    Ground robots send ``[vx, vy, omega]``; drones send
    ``[vx, vy, vz, yaw_rate]``.  The command array length matches
    ``cfg.model.action_dim``.
    """

    async def connect(self) -> None:
        """Establish connection to the motor controller."""
        ...

    async def send_command(self, command: NDArray[np.floating[Any]]) -> None:
        """Send a scaled command vector to the actuators.

        Args:
            command: Scaled command array whose length equals ``action_dim``.
                     Ground: ``[vx, vy, omega]``.
                     Drone: ``[vx, vy, vz, yaw_rate]``.
        """
        ...

    async def read_state(self) -> NDArray[np.float32]:
        """Read current actuator/platform state.

        Returns:
            State array. Ground: ``[left_vel, right_vel, heading, battery_v]``.
            Drone: ``[vx, vy, vz, yaw_rate, altitude, battery_v, armed]``.
        """
        ...

    async def get_battery_voltage(self) -> float:
        """Read battery voltage.

        Returns:
            Battery voltage in volts.
        """
        ...

    async def emergency_stop(self) -> None:
        """Send emergency stop — zero all actuators immediately."""
        ...

    async def disconnect(self) -> None:
        """Close connection to the motor controller."""
        ...

    @property
    def platform_type(self) -> str:
        """Identifier for the platform type (e.g. 'mouse_droid', 'drone')."""
        ...
