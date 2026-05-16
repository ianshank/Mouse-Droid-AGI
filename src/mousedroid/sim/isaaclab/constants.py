"""Isaac Lab Phase B constants — single source of truth for rover wiring.

URDF link/joint names mirror ``assets/rover/mse6_4wd.urdf``. Changing the
URDF requires updating this module in lockstep; the smoke test in
``tests/unit/sim/isaaclab/test_constants.py`` asserts the URDF still
defines these names.

Numeric dimension constants (``ROVER_NUM_WHEELS``, ``ROVER_IMU_DIM``,
``ROVER_CHASSIS_POSE_DIM``) are imported from
:mod:`mousedroid.sim.protocols` — single source of truth across the
mock and Isaac Lab backends.
"""

from __future__ import annotations

from typing import Final

from mousedroid.sim.protocols import (
    ROVER_CHASSIS_POSE_DIM,
    ROVER_IMU_DIM,
    ROVER_NUM_WHEELS,
)

# Re-export for ergonomic ``from .constants import ROVER_NUM_WHEELS``.
__all__ = [
    "ROVER_ACTION_DIM",
    "ROVER_CHASSIS_POSE_DIM",
    "ROVER_IMU_DIM",
    "ROVER_NUM_WHEELS",
    "ROVER_OBSERVATION_KEYS",
    "ROVER_SENSOR_LINK_NAMES",
    "ROVER_WHEEL_JOINT_NAMES",
]


# ---------------------------------------------------------------------------
# URDF-derived names — must match assets/rover/mse6_4wd.urdf verbatim
# ---------------------------------------------------------------------------

ROVER_WHEEL_JOINT_NAMES: Final[tuple[str, str, str, str]] = (
    "joint_wheel_fl",  # front-left
    "joint_wheel_fr",  # front-right
    "joint_wheel_rl",  # rear-left
    "joint_wheel_rr",  # rear-right
)
"""URDF joint names for the 4 continuous wheel revolutes. Order matches
the action-vector layout passed to ``RoverIsaacLabEnv.step()`` — that is,
``action[0]`` drives ``joint_wheel_fl``, ``action[1]`` drives
``joint_wheel_fr``, and so on.

Length is guaranteed equal to :data:`ROVER_NUM_WHEELS` (4) — both come
from the rover platform's single-source-of-truth and changing one
without the other will trip the test in ``test_constants.py``."""


ROVER_SENSOR_LINK_NAMES: Final[tuple[str, str, str]] = (
    "imu_link",
    "lidar_link",
    "camera_link",
)
"""URDF link names for the 3 fixed sensor frames. Isaac Lab attaches
``IMUSensorCfg``, ``RayCasterSensorCfg`` (LiDAR), and ``CameraSensorCfg``
to these links during ``RoverIsaacLabEnv.build()``."""


# ---------------------------------------------------------------------------
# Action / observation layout — kept in lockstep with MockRoverEnv contract
# ---------------------------------------------------------------------------

ROVER_ACTION_DIM: Final[int] = ROVER_NUM_WHEELS
"""One velocity command per wheel. Matches
:attr:`MockRoverEnv.action_dim` — cross-backend contract guarantee."""


ROVER_OBSERVATION_KEYS: Final[tuple[str, ...]] = (
    "imu",
    "chassis_pose",
    "wheel_vel",
    "lidar",
)
"""Default observation dict keys emitted by ``RoverIsaacLabEnv.reset()``
and ``.step()``. Mirrors :attr:`MockRoverEnv.observation_keys` exactly
under the default :class:`RoverObservationConfig` so both backends are
drop-in replacements at the orchestrator level.

Optional keys (``camera``, etc.) are gated by
:class:`RoverObservationConfig` toggles and surface only when the
corresponding flag is enabled — both backends honour the same toggles.

**Excludes ultrasonic / HC-SR04 / arm / gripper** — the active rover
production baseline is IMU + chassis pose + wheel velocity + LiDAR only.
The arm platform and the old ultrasonic distance sensor are parked per
project CLAUDE.md."""
