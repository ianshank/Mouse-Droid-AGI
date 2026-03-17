"""Ground motor adapter — wraps ESP32CommProtocol as MotorControlProtocol.

Translates the platform-agnostic ``send_command`` / ``read_state`` API
into ESP32-specific ``send_velocity`` / ``read_encoders`` calls.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.comms.protocol import ESP32CommProtocol

_log = get_logger(__name__)


class GroundMotorAdapter:
    """Adapt ``ESP32CommProtocol`` to ``MotorControlProtocol``.

    Args:
        esp32: Underlying ESP32 driver (serial, wifi, or mock).
    """

    def __init__(self, esp32: ESP32CommProtocol) -> None:
        self._esp32 = esp32
        _log.info("ground_motor_adapter_init")

    # -- MotorControlProtocol ----------------------------------------------

    async def connect(self) -> None:
        """Connect to ESP32."""
        await self._esp32.connect()

    async def send_command(self, command: NDArray[np.floating[Any]]) -> None:
        """Extract ``[vx, vy, omega]`` and forward to ESP32.

        Args:
            command: Scaled command array of length >= 1.
        """
        vx = float(command[0]) if command.size > 0 else 0.0
        vy = float(command[1]) if command.size > 1 else 0.0
        omega = float(command[2]) if command.size > 2 else 0.0
        await self._esp32.send_velocity(vx, vy, omega)
        _log.debug("ground_command_sent", vx=vx, vy=vy, omega=omega)

    async def read_state(self) -> NDArray[np.float32]:
        """Read encoders + battery and pack as ``[left_vel, right_vel, heading, battery_v]``.

        Returns:
            4-element motor state array.
        """
        encoders, battery_v = await asyncio.gather(
            self._esp32.read_encoders(),
            self._esp32.get_battery_voltage(),
        )
        state = np.array(
            [
                encoders.left_velocity_mps,
                encoders.right_velocity_mps,
                encoders.heading_rad,
                battery_v,
            ],
            dtype=np.float32,
        )
        return state

    async def get_battery_voltage(self) -> float:
        """Read battery voltage from ESP32 ADC.

        Returns:
            Battery voltage in volts.
        """
        return await self._esp32.get_battery_voltage()

    async def emergency_stop(self) -> None:
        """Send emergency stop to ESP32."""
        await self._esp32.emergency_stop()
        _log.warning("ground_emergency_stop")

    async def disconnect(self) -> None:
        """Disconnect from ESP32."""
        await self._esp32.disconnect()

    @property
    def platform_type(self) -> str:
        """Return ``'mouse_droid'``."""
        return "mouse_droid"
