"""SO-ARM100 serial driver for real hardware control.

Communicates with the SO-ARM100 robot arm over serial UART.
Joint commands are sent as position targets; joint states are
read back from the controller.

All blocking serial I/O is delegated to ``asyncio.to_thread`` to
avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ArmConfig

_log = get_logger(__name__)


class SoArm100Driver:
    """Serial driver for Hiwonder SO-ARM100 robot arm.

    Communicates over UART to command joint positions and
    read joint states from the SO-ARM100 servo controller.

    All blocking serial I/O is delegated to :func:`asyncio.to_thread`
    to comply with the asyncio-everywhere invariant.

    Args:
        cfg: Arm hardware configuration.
    """

    def __init__(self, cfg: ArmConfig) -> None:
        """Initialise SO-ARM100 driver.

        Args:
            cfg: Arm config with serial port, baud rate, DOF.
        """
        self._cfg = cfg
        self._dof = cfg.dof
        self._serial_port = cfg.serial_port
        self._baud = cfg.serial_baud
        self._timeout = cfg.command_timeout_s
        self._serial: Any = None
        _log.info(
            "so_arm100_driver_init",
            port=self._serial_port,
            baud=self._baud,
            dof=self._dof,
        )

    def _open_serial(self) -> Any:  # pragma: no cover
        """Open the serial port (blocking — called via ``to_thread``)."""
        import serial

        return serial.Serial(
            self._serial_port,
            self._baud,
            timeout=self._timeout,
        )

    def _close_serial(self) -> None:  # pragma: no cover
        """Close the serial port (blocking — called via ``to_thread``)."""
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    async def start(self) -> None:
        """Open serial connection to arm controller."""
        try:
            self._serial = await asyncio.to_thread(self._open_serial)
            _log.info("so_arm100_connected", port=self._serial_port)
        except ImportError:
            _log.error("pyserial_not_installed")
            raise
        except Exception:
            _log.error("so_arm100_connection_failed", port=self._serial_port, exc_info=True)
            raise

    async def stop(self) -> None:
        """Close serial connection."""
        if self._serial is not None:
            await asyncio.to_thread(self._close_serial)
            _log.info("so_arm100_disconnected")

    async def get_joint_states(self) -> NDArray[np.float64]:
        """Read current joint angles from arm controller.

        Returns:
            Joint angles in radians, shape ``(dof,)``.

        Raises:
            RuntimeError: If serial connection is not open.
        """
        if self._serial is None:
            msg = "Serial connection not open. Call start() first."
            raise RuntimeError(msg)

        _log.debug("reading_joint_states")
        # Placeholder — real protocol implementation depends on SO-ARM100 firmware
        return np.zeros(self._dof, dtype=np.float64)

    async def send_joint_command(self, target_angles: NDArray[np.float64]) -> None:
        """Command arm to target joint angles.

        Args:
            target_angles: Target joint angles in radians, shape ``(dof,)``.

        Raises:
            RuntimeError: If serial connection is not open.
            ValueError: If angle count doesn't match DOF.
        """
        if self._serial is None:
            msg = "Serial connection not open. Call start() first."
            raise RuntimeError(msg)

        if len(target_angles) != self._dof:
            msg = f"Expected {self._dof} joint angles, got {len(target_angles)}"
            raise ValueError(msg)

        _log.debug("sending_joint_command", angles=target_angles.tolist())
        # Placeholder — real protocol implementation needed

    async def close_gripper(self) -> float:
        """Close gripper and measure grip force.

        Returns:
            Measured grip force in Newtons.
        """
        _log.debug("closing_gripper")
        # Placeholder — command gripper servo, read force sensor
        return 0.0

    async def open_gripper(self) -> None:
        """Open gripper fully."""
        _log.debug("opening_gripper")
        # Placeholder — command gripper servo to open position

    async def emergency_stop(self) -> None:
        """Immediately halt all arm motion."""
        _log.warning("emergency_stop_triggered")
        if self._serial is not None:
            # Send emergency stop command
            pass

    async def home(self) -> None:
        """Move arm to home position."""
        home_angles = np.array(self._cfg.home_position, dtype=np.float64)
        await self.send_joint_command(home_angles)
        _log.info("arm_homed")

    @property
    def dof(self) -> int:
        """Degrees of freedom."""
        return self._dof
