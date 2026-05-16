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
    "ROVER_CONTACT_SENSOR_NAME",
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
"""URDF joint names for the 4 continuous wheel revolutes.

**This is the actuator fan-out order**, not the policy action-vector
layout. The policy action vector is mode-dependent and lives on
:attr:`RoverActionConfig.action_dim` (2 for both supported modes —
``differential`` and ``body_velocity`` — per ``_ROVER_ACTION_DIM_BY_MODE``
in the schema). Phase B's ``RoverIsaacLabEnv.step()`` body fans out the
incoming 2-D action onto these 4 wheel actuators in the same way
:class:`MockRoverEnv` does: differential drive duplicates ``action[0]``
into the FL+RL pair and ``action[1]`` into the FR+RR pair.

The order ``(fl, fr, rl, rr)`` is the contract Phase B's fan-out builds
on; reordering this tuple without also updating the env step body
would route wheel commands to the wrong wheels. The test
``test_wheel_joint_order_matches_actuator_fan_out`` pins the order.

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


ROVER_CONTACT_SENSOR_NAME: Final[str] = "contact_sensor"
"""Key under which the chassis :class:`ContactSensor` lives in
``RoverIsaacLabEnv._sensors``. The sensor's USD prim path expression
covers every articulation body so the per-frame collision flag in
:meth:`RoverIsaacLabEnv._read_collision_flag` reflects any body-vs-world
contact, which feeds the ``-collision_weight * is_colliding`` term of
:class:`RoverRewardConfig`."""


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
"""Observation dict keys emitted under the **default**
:class:`RoverObservationConfig` (all four modality toggles enabled).
Operators flipping any ``include_*`` toggle in the config get a
different live observation set in both backends — the authoritative
runtime source of truth is :meth:`RoverObservationConfig.enabled_keys`,
which both ``MockRoverEnv`` and ``RoverIsaacLabEnv`` read at
``__init__``.

This constant exists primarily for:
  * Documentation in tests and ADRs (the "what does the rover emit
    out of the box?" question)
  * The cross-backend contract test
    (:func:`test_observation_keys_match_mock_rover_env`) — which
    confirms ``RoverIsaacLabEnv`` and ``MockRoverEnv`` agree under
    the default config

Non-default toggles (e.g. operator disables LiDAR) are NOT exercised
by this constant; the contract test only validates the default-config
case. Backend implementers should not branch on this tuple at runtime —
read ``cfg.observation.enabled_keys()`` instead.

**Excludes ultrasonic / HC-SR04 / arm / gripper** — the active rover
production baseline is IMU + chassis pose + wheel velocity + LiDAR only.
The arm platform and the old ultrasonic distance sensor are parked per
project CLAUDE.md."""
