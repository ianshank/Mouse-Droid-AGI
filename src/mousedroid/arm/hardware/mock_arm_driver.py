"""Mock robot arm driver for testing and simulation.

Implements ArmDriverProtocol with in-memory state tracking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ArmConfig

_log = get_logger(__name__)


class MockArmDriver:
    """Mock arm driver that tracks state in memory.

    Simulates arm joint states and gripper behaviour without
    real hardware. Useful for unit testing and sim-only training.

    Args:
        cfg: Arm hardware configuration.
    """

    def __init__(self, cfg: ArmConfig) -> None:
        """Initialise mock arm driver.

        Args:
            cfg: Arm config with DOF, home position, velocity limits.
        """
        self._cfg = cfg
        self._dof = cfg.dof
        self._joint_angles = np.array(cfg.home_position, dtype=np.float64)
        self._gripper_open = True
        self._grip_force = 0.0
        self._running = False
        _log.info("mock_arm_driver_init", dof=self._dof)

    async def get_joint_states(self) -> NDArray[np.float64]:
        """Read current joint angles.

        Returns:
            Joint angles in radians, shape ``(dof,)``.
        """
        return np.array(self._joint_angles, dtype=np.float64)

    async def send_joint_command(self, target_angles: NDArray[np.float64]) -> None:
        """Command arm to target joint angles.

        Instantly sets joints to target (no dynamics simulation).

        Args:
            target_angles: Target joint angles in radians, shape ``(dof,)``.
        """
        if len(target_angles) != self._dof:
            msg = f"Expected {self._dof} joint angles, got {len(target_angles)}"
            raise ValueError(msg)
        self._joint_angles = np.clip(target_angles, -np.pi, np.pi).astype(np.float64)
        _log.debug("mock_joint_command", angles=self._joint_angles.tolist())

    async def close_gripper(self) -> float:
        """Close gripper and return simulated grip force.

        Returns:
            Simulated grip force (1.0 N if gripper was open, 0.0 if already closed).
        """
        if self._gripper_open:
            self._gripper_open = False
            self._grip_force = 1.0
        else:
            self._grip_force = 0.0
        _log.debug("mock_gripper_close", force=self._grip_force)
        return self._grip_force

    async def open_gripper(self) -> None:
        """Open gripper fully."""
        self._gripper_open = True
        self._grip_force = 0.0
        _log.debug("mock_gripper_open")

    async def emergency_stop(self) -> None:
        """Halt all arm motion (no-op in mock)."""
        _log.warning("mock_emergency_stop")

    async def home(self) -> None:
        """Move arm to home position."""
        self._joint_angles = np.array(self._cfg.home_position, dtype=np.float64)
        self._gripper_open = True
        self._grip_force = 0.0
        _log.info("mock_arm_homed")

    async def start(self) -> None:
        """Initialise mock driver."""
        self._running = True
        _log.info("mock_arm_started")

    async def stop(self) -> None:
        """Shut down mock driver."""
        self._running = False
        _log.info("mock_arm_stopped")

    @property
    def dof(self) -> int:
        """Degrees of freedom."""
        return self._dof
