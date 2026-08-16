"""4WD rover sim-to-real configuration models (Phase A scaffold).

Mass-property overrides, the MuJoCo backend, simulation-backend selection
and physics timing, and the action/observation/task/reward sub-blocks
composed by the top-level ``RoverConfig``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RoverInertialConfig(BaseModel):
    """Mass-property overrides for the MSE-6 shell + 4WD chassis URDF.

    Used by the Isaac Lab env stub (and future MuJoCo backend) to update
    the URDF's documentation-quality defaults with values derived from
    the actual 3D-print parameters of the physical droid. A top-heavy
    ``com_offset_xyz_m`` is intentional — the policy must experience the
    roll tendency in sim to generalise to hardware.
    """

    shell_mass_kg: float = Field(0.85, gt=0, description="MSE-6 shell mass (kg)")
    shell_thickness_m: float = Field(
        0.003, gt=0, description="Shell wall thickness for hollow inertia (m)"
    )
    shell_infill: float = Field(0.20, gt=0, le=1.0, description="Print infill fraction [0, 1]")
    com_offset_xyz_m: tuple[float, float, float] = Field(
        (0.0, 0.0, 0.04),
        description="COM offset from base_link origin (top-heavy: positive z)",
    )
    wheel_mass_kg: float = Field(0.06, gt=0, description="Per-wheel mass (kg)")


class MujocoSimConfig(BaseModel):
    """MuJoCo backend parameters (consumed only when ``rover.sim.backend == 'mujoco'``).

    Every physics knob is config-driven (invariant #3). ``wheel_slip_default``
    is a documented OBSERVATION-NOISE proxy — MuJoCo has no first-class slip
    parameter — applied as multiplicative noise on wheel_vel/pose, NOT a
    contact-solver field.
    """

    mjcf_path: str = Field(
        "assets/rover/mse6_4wd.xml",
        min_length=1,
        description="Repo-relative path to the skid-steer MJCF (resolved against repo root).",
    )
    arena_half_extent_m: float = Field(
        2.0, gt=0.0, description="Half-size of the walled arena (walls give the lidar a signal)."
    )
    lidar_num_sectors: int = Field(
        16, gt=0, description="Number of rangefinder sectors fanned around yaw."
    )
    lidar_max_range_m: float = Field(
        4.0, gt=0.0, description="Rangefinder clip; readings normalised to [0,1] by this."
    )
    lidar_ring_radius_m: float = Field(
        0.11, gt=0.0, description="Radius of the lidar-site ring on the chassis (matches MJCF)."
    )
    lidar_mount_z_m: float = Field(
        0.03, description="Z offset of the lidar sites above the chassis origin (matches MJCF)."
    )
    noise_rng_seed: int = Field(
        0, ge=0, description="Default seed for the slip observation-noise RNG (reset overrides)."
    )
    battery_voltage_const_v: float = Field(
        12.0, gt=0.0, description="Constant battery voltage stamped into motor_state[3]."
    )
    wheel_friction_default: float = Field(
        1.0, gt=0.0, description="Default tangential friction (geom_friction[:,0])."
    )
    wheel_slip_default: float = Field(
        0.0, ge=0.0, description="Observation-noise proxy magnitude (NOT a MuJoCo field)."
    )
    motor_gain_default: float = Field(
        1.0, gt=0.0, description="Default actuator gain (actuator_gainprm[:,0])."
    )
    chassis_mass_default_kg: float = Field(
        2.7, gt=0.0, description="Default chassis mass (body_mass + inertia recompute)."
    )
    render_vision: bool = Field(
        False,
        description="Render an RGB camera for vision-on RSSM fine-tuning (off by default).",
    )
    render_width: int = Field(64, gt=0, description="Offscreen RGB render width (px).")
    render_height: int = Field(64, gt=0, description="Offscreen RGB render height (px).")
    camera_name: str = Field(
        "rover_cam", min_length=1, description="Name of the MJCF camera to render from."
    )


class RoverSimConfig(BaseModel):
    """Simulation backend selection and physics timing for rover training."""

    backend: Literal["isaac_lab", "mujoco", "mock"] = Field(
        "mock",
        description=(
            "Sim backend. 'mock' (NumPy, no physics) is the default so CI "
            "and unit tests run without GPU/Isaac dependencies."
        ),
    )
    urdf_path: str = Field(
        "assets/rover/mse6_4wd.urdf",
        description="Path to the rover URDF (relative to repo root)",
    )
    sim_dt_s: float = Field(
        1.0 / 120.0, gt=0, description="Physics step (s); decimation maps to control rate"
    )
    decimation: int = Field(
        4, ge=1, description="Physics steps per control step (30 Hz control at dt=1/120)"
    )
    episode_length_s: float = Field(
        20.0, gt=0, description="Max episode duration before truncation (s)"
    )
    num_envs: int = Field(1, ge=1, description="Parallel envs (use 4096+ for Isaac Lab training)")
    headless: bool = Field(True, description="Run Isaac Lab without a viewer")
    inertial: RoverInertialConfig = Field(
        default_factory=RoverInertialConfig,
        description="Mass-property overrides for the URDF defaults",
    )
    mujoco: MujocoSimConfig = Field(
        default_factory=MujocoSimConfig,
        description="MuJoCo backend parameters (used only when backend == 'mujoco').",
    )


# Action vector dimensionality per supported mode. Centralised so env
# classes don't carry a magic ``2`` — adding e.g. a 4-wheel mecanum mode
# in Phase B is a single-line dict update plus the new mode literal.
_ROVER_ACTION_DIM_BY_MODE: dict[str, int] = {
    "differential": 2,
    "body_velocity": 2,
}


class RoverActionConfig(BaseModel):
    """Action space configuration for the rover policy."""

    mode: Literal["differential", "body_velocity"] = Field(
        "differential",
        description=(
            "'differential' -> [left_wheel_rad_s, right_wheel_rad_s]; "
            "'body_velocity' -> [vx_mps, omega_rads]."
        ),
    )
    max_wheel_rad_s: float = Field(
        25.0, gt=0, description="Hard cap on per-wheel angular velocity (rad/s)"
    )
    slew_rad_s2: float = Field(
        60.0,
        gt=0,
        description=(
            "Max wheel angular acceleration (rad/s^2). Consumed by the "
            "Phase B neurosymbolic action validator; recorded here so the "
            "URDF, env, and safety layer share one source of truth."
        ),
    )

    @property
    def action_dim(self) -> int:
        """Return the action-vector dimensionality implied by ``mode``."""
        return _ROVER_ACTION_DIM_BY_MODE[self.mode]


class RoverObservationConfig(BaseModel):
    """Observation-space toggles for the rover env."""

    include_imu: bool = Field(True, description="6-D linear-accel + ang-vel vector")
    include_wheel_encoders: bool = Field(True, description="4-D wheel angular velocities")
    include_chassis_pose: bool = Field(
        True, description="4-D [x, y, cos(theta), sin(theta)] body pose"
    )
    include_lidar_sectors: bool = Field(True, description="Sector-binned LiDAR clearance features")
    lidar_num_sectors: int = Field(
        16, ge=1, description="Number of angular sectors for LiDAR features"
    )

    def enabled_keys(self) -> tuple[str, ...]:
        """Return the obs-dict keys implied by the enabled modality toggles.

        Single source of truth for the observation contract — the mock and
        Isaac Lab env classes plus the factory log call this so the keys
        and their order can never drift between backends.
        """
        keys: list[str] = []
        if self.include_imu:
            keys.append("imu")
        if self.include_chassis_pose:
            keys.append("chassis_pose")
        if self.include_wheel_encoders:
            keys.append("wheel_vel")
        if self.include_lidar_sectors:
            keys.append("lidar")
        return tuple(keys)


class RoverTaskConfig(BaseModel):
    """Placeholder goal-reach task parameters for the Phase A mock env.

    The mock env's reward is a placeholder ``-||pose - goal_xy_m||`` that
    terminates inside ``goal_reach_radius_m``. The full reward shaper
    (Phase C, ``mousedroid.training.rover_reward``) will replace both
    fields with a structured multi-objective signal; until then these
    knobs let callers steer the placeholder without editing code.
    """

    goal_xy_m: tuple[float, float] = Field(
        (2.0, 0.0),
        description="World-frame goal pose (x, y) in metres",
    )
    goal_reach_radius_m: float = Field(
        0.10,
        gt=0,
        description="Distance to goal at which the episode terminates (m)",
    )


class RoverRewardConfig(BaseModel):
    """Reward weights for the Isaac Lab rover env (Tier C4 — Phase B baseline).

    Implements the design documented in ADR-009 (Isaac Lab Phase B). The
    Isaac Lab ``step()`` body composes the per-step reward as::

        reward = (
            forward_velocity_weight * forward_velocity_mps
            - collision_weight * is_colliding
        )

    Both weights are operator-tunable; no hardcoded reward weights live
    inside :mod:`mousedroid.sim.isaaclab`. Backwards-compatible default
    on :class:`RoverConfig` is ``reward=None``; the Isaac Lab env raises
    a clear :class:`ValueError` when built without an explicit reward
    block so operators set it intentionally per ADR-009.
    """

    forward_velocity_weight: float = Field(
        0.01,
        ge=0,
        description=(
            "Reward per m/s forward (linear body-frame velocity). Must "
            "be ``>= 0``; negative values would invert the safety sign "
            "(rewarding reverse motion) and contradict ADR-009."
        ),
    )
    collision_weight: float = Field(
        0.1,
        ge=0,
        description=(
            "Penalty per collision frame (subtracted from reward). Must "
            "be ``>= 0``; negative values would reward crashes and "
            "violate the constitutional safety invariant."
        ),
    )


class RoverConfig(BaseModel):
    """Top-level rover sim-to-real configuration (None preserves legacy).

    Optional on the root :class:`Settings`. When ``None``, existing YAML
    files load unchanged and the orchestrator behaves as before.
    """

    sim: RoverSimConfig = Field(default_factory=RoverSimConfig)
    action: RoverActionConfig = Field(default_factory=RoverActionConfig)
    observation: RoverObservationConfig = Field(default_factory=RoverObservationConfig)
    task: RoverTaskConfig = Field(default_factory=RoverTaskConfig)
    reward: RoverRewardConfig | None = Field(
        None,
        description=(
            "Isaac Lab reward weights (Tier C4). ``None`` preserves "
            "byte-identical pre-PR behaviour; the Isaac Lab env raises "
            "``ValueError`` when built without an explicit block so "
            "operators set the weights intentionally per ADR-009."
        ),
    )
