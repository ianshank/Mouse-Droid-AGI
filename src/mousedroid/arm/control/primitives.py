"""Pre-trained grasp/place action primitives for robot arm.

Modular action library providing high-level manipulation primitives
that the hierarchical policy can compose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.arm.protocols import ArmDriverProtocol
    from mousedroid.config.schema import ArmConfig

_log = get_logger(__name__)


class ActionPrimitives:
    """Library of pre-trained manipulation action primitives.

    Provides grasp, place, and transit primitives that abstract
    away low-level joint trajectory planning.

    Args:
        arm_cfg: Arm hardware configuration.
        driver: Arm driver protocol for execution.
    """

    def __init__(self, arm_cfg: ArmConfig, driver: ArmDriverProtocol) -> None:
        """Initialise action primitives.

        Args:
            arm_cfg: Arm config with joint limits and home position.
            driver: Arm driver for commanding joints.
        """
        self._cfg = arm_cfg
        self._driver = driver
        self._home = np.array(arm_cfg.home_position, dtype=np.float64)
        _log.info("action_primitives_init", dof=arm_cfg.dof)

    async def transit(self, target_pose: NDArray[np.float64]) -> bool:
        """Move end-effector to target pose without grasping.

        Args:
            target_pose: Target joint angles, shape ``(dof,)``.

        Returns:
            True if transit completed successfully.
        """
        _log.debug("primitive_transit", target=target_pose.tolist())
        try:
            await self._driver.send_joint_command(target_pose)
            return True
        except Exception:
            _log.error("transit_failed", exc_info=True)
            return False

    async def grasp(self, pre_grasp_pose: NDArray[np.float64]) -> float:
        """Execute grasp primitive: approach + close gripper.

        Args:
            pre_grasp_pose: Joint angles for pre-grasp position.

        Returns:
            Measured grip force (0.0 if grasp failed).
        """
        _log.debug("primitive_grasp", pre_grasp=pre_grasp_pose.tolist())
        try:
            await self._driver.send_joint_command(pre_grasp_pose)
            force = await self._driver.close_gripper()
            _log.info("grasp_complete", force=force)
            return force
        except Exception:
            _log.error("grasp_failed", exc_info=True)
            return 0.0

    async def place(self, place_pose: NDArray[np.float64]) -> bool:
        """Execute place primitive: move to target + open gripper.

        Args:
            place_pose: Joint angles for place position.

        Returns:
            True if place completed successfully.
        """
        _log.debug("primitive_place", target=place_pose.tolist())
        try:
            await self._driver.send_joint_command(place_pose)
            await self._driver.open_gripper()
            _log.info("place_complete")
            return True
        except Exception:
            _log.error("place_failed", exc_info=True)
            return False

    async def home(self) -> bool:
        """Return arm to home position.

        Returns:
            True if home position reached.
        """
        _log.debug("primitive_home")
        try:
            await self._driver.home()
            return True
        except Exception:
            _log.error("home_failed", exc_info=True)
            return False
