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
# Observation layout — kept in lockstep with MockRoverEnv contract
# ---------------------------------------------------------------------------
#
# Action-vector dimensionality is intentionally NOT a constant here. The
# policy action_dim is mode-dependent (2 for differential / body_velocity,
# could be 4 for a future independent-wheel mode) and lives on
# :attr:`RoverActionConfig.action_dim` (mapped via
# ``_ROVER_ACTION_DIM_BY_MODE`` in the schema). Both ``MockRoverEnv`` and
# ``RoverIsaacLabEnv`` read it from ``cfg.action.action_dim`` so the two
# backends cannot diverge.
#
# The 4-wheel articulation count remains :data:`ROVER_NUM_WHEELS` — that
# is a property of the physical chassis (URDF defines 4 wheel revolutes)
# and is unrelated to whatever shape the policy chooses to emit. The
# Isaac Lab ``step()`` body is responsible for fanning a 2-D
# differential-drive action across the 4 wheel actuators.


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

The list intentionally matches the default
:meth:`RoverObservationConfig.enabled_keys` output **exactly** (no
subset, no superset). Toggling a key on/off in the config flips it in
both backends; the cross-backend contract test in
``test_constants.py`` asserts that exact equality.

**Excludes ultrasonic / HC-SR04 / arm / gripper** — the active rover
production baseline is IMU + chassis pose + wheel velocity + LiDAR only.
The arm platform and the old ultrasonic distance sensor are parked per
project CLAUDE.md."""
