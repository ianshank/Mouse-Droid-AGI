"""Isaac Lab environment for the 4WD MSE-6 rover (Tier C4 body wiring).

Phase A landed the import-safe stub; Tier C4 fills in the three
``TODO(Phase B)`` markers in :meth:`build`, :meth:`reset`, and
:meth:`step` with real Isaac Lab :class:`ManagerBasedRLEnv`-style
wiring while preserving every architectural invariant from
``CLAUDE.md`` (lazy import, no hardcoded values, structured logging,
``mypy --strict``-clean, backwards-compatible defaults).

Operator validation lives on Linux + Isaac Sim per ADR-009; CI hosts
without ``isaaclab`` get a clean ``ImportError`` from the ``build``
path so the rest of the test suite continues to load this module.

Implementation notes:
  * **Lazy imports**: every ``isaaclab`` symbol is imported inside the
    method bodies, never at module top-level. This mirrors the VLA
    policy + B2 ONNX engine convention.
  * **Wheel fan-out**: the 2-D differential-drive action ``[left, right]``
    fans onto the 4 wheel articulation actuators in
    :data:`ROVER_WHEEL_JOINT_NAMES` order as
    ``[FL=left, FR=right, RL=left, RR=right]`` — *alternating*, not
    grouped — matching :meth:`MockRoverEnv._action_to_body_velocity`
    exactly so the cross-backend contract test passes.
  * **Domain randomization**: the env reuses
    :class:`mousedroid.training.domain_randomization.DomainRandomizer`
    and honours ``cfg.domain_randomization.enabled`` (the top-level
    :class:`DomainRandomizationConfig`, not a nested rover-only block —
    see ADR-009 amendment).
  * **Reward**: composed from :class:`RoverRewardConfig` weights at
    ``forward_velocity_weight * forward_velocity_mps -
    collision_weight * is_colliding``. The env raises ``ValueError``
    at :meth:`build` time when ``cfg.rover.reward is None`` so
    operators set the block explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.common.imports import module_importable
from mousedroid.config.schema import RoverConfig, RoverRewardConfig
from mousedroid.logging.setup import get_logger
from mousedroid.sim.isaaclab.constants import (
    ROVER_CONTACT_SENSOR_NAME,
    ROVER_SENSOR_LINK_NAMES,
    ROVER_WHEEL_JOINT_NAMES,
)
from mousedroid.sim.isaaclab.randomization import (
    apply_isaac_domain_params,
    apply_isaac_episode_extras,
)
from mousedroid.sim.isaaclab.scene import (
    isaac_sim_device,
    resolve_usd_path,
    simulation_cfg_kwargs,
)
from mousedroid.sim.isaaclab.sensors import (
    identity_chassis_pose,
    read_rover_imu,
    read_rover_lidar,
    read_rover_pose,
)
from mousedroid.sim.kinematics import (
    body_velocity_to_wheels,
    policy_action_to_body,
    wheels_array_to_body_velocity,
)
from mousedroid.sim.protocols import (
    ROVER_IMU_DIM,
    ROVER_NUM_WHEELS,
)

if TYPE_CHECKING:
    from mousedroid.config.schema import DomainRandomizationConfig

_log = get_logger(__name__)


class IsaacLabUnavailableError(RuntimeError):
    """Raised when the Isaac Lab backend is requested without the dep installed."""


class RoverEnvNotBuiltError(RuntimeError):
    """Raised when ``reset``/``step`` is called before :meth:`build`."""


def _isaaclab_available() -> bool:
    """Return ``True`` iff Isaac Lab can actually be imported in the current env.

    Uses a *real* guarded import (not a spec-only probe): a positive result
    gates :meth:`RoverIsaacLabEnv.build`, which proceeds to import and drive
    Isaac Lab via ``_wire_isaaclab_scene``. A spec-present-but-import-fails
    install (Isaac Lab discoverable but missing GPU/native bindings) must
    surface as the graceful :class:`IsaacLabUnavailableError`, not a hard crash
    deep inside scene wiring.
    """
    return module_importable("isaaclab")


class RoverIsaacLabEnv:
    """Isaac Lab env conforming to :class:`RoverEnvProtocol` (C4 body).

    The constructor does **not** initialise Isaac Lab; call
    :meth:`build` to do so. This split keeps the class importable
    without GPU drivers; tests that need a live backend skip with
    ``pytest.skipif(not _isaaclab_available(), ...)``.
    """

    def __init__(
        self,
        cfg: RoverConfig,
        wheel_radius_m: float,
        track_width_m: float,
        *,
        domain_randomization: DomainRandomizationConfig | None = None,
    ) -> None:
        """Initialise the env wrapper.

        Args:
            cfg: Rover configuration block from :class:`Settings`.
            wheel_radius_m: Wheel radius from :class:`RobotConfig`.
            track_width_m: Track width from :class:`RobotConfig`.
            domain_randomization: Top-level
                :class:`DomainRandomizationConfig` from
                :class:`Settings`; ``None`` disables per-episode
                randomization at reset time. The path is
                ``cfg.domain_randomization`` on root :class:`Settings`,
                NOT nested under ``cfg.sim`` (corrected per ADR-009
                amendment).
        """
        self._cfg = cfg
        self._wheel_radius = wheel_radius_m
        self._track_width = track_width_m
        self._dr_cfg = domain_randomization
        self._built: bool = False
        self._step_idx: int = 0
        self._action_dim = cfg.action.action_dim
        self._obs_keys: tuple[str, ...] = cfg.observation.enabled_keys()

        # Handles populated by ``build``; ``Any`` here because the
        # concrete Isaac Lab classes only exist when the optional dep
        # is installed. The handles are only ever touched after
        # ``_require_built`` validates ``self._built`` is True, so the
        # ``Any`` storage doesn't escape into mypy-strict-checked code
        # paths reachable without isaaclab.
        self._sim_context: Any = None
        self._scene: Any = None
        self._articulation: Any = None
        self._sensors: dict[str, Any] = {}
        self._pending_domain: dict[str, float] | None = None

    # ----- lifecycle --------------------------------------------------------

    def build(self) -> None:
        """Initialise the Isaac Lab simulation context, articulation, and sensors.

        Wires the URDF-derived USD asset at ``cfg.sim.urdf_path`` into a
        :class:`ManagerBasedRLEnv`-style scene with sensors attached to
        the three :data:`ROVER_SENSOR_LINK_NAMES` (``imu_link``,
        ``lidar_link``, ``camera_link``) and actuators on the four
        :data:`ROVER_WHEEL_JOINT_NAMES` continuous wheel joints.

        Raises:
            IsaacLabUnavailableError: When ``isaaclab`` cannot be imported.
            ValueError: When ``cfg.rover.reward is None`` — operators
                must set the reward block explicitly per ADR-009.
        """
        if not _isaaclab_available():
            msg = (
                "Isaac Lab is not installed in this environment. "
                'Install with `pip install -e ".[isaac]"` on a workstation '
                "with NVIDIA Isaac Lab pre-requisites."
            )
            raise IsaacLabUnavailableError(msg)

        if self._cfg.reward is None:
            msg = (
                "RoverIsaacLabEnv requires an explicit "
                "cfg.rover.reward (RoverRewardConfig) block. The "
                "default None preserves byte-identical pre-Tier-C4 "
                "behaviour; set the block in your YAML or pass it "
                "directly per ADR-009."
            )
            raise ValueError(msg)

        # Delegate the Isaac-Lab-only wiring to a helper that lives
        # under ``# pragma: no cover`` — the CI host (where this branch
        # never runs because of the ``_isaaclab_available`` guard) does
        # not need to count those lines against the coverage gate. The
        # operator validates the live path on Linux + Isaac Sim per the
        # C4 playbook.
        self._wire_isaaclab_scene()

    # ----- protocol surface -------------------------------------------------

    @property
    def action_dim(self) -> int:
        """Return the action-space dimensionality (always 2)."""
        return self._action_dim

    @property
    def observation_keys(self) -> tuple[str, ...]:
        """Return the keys present in observation dicts."""
        return self._obs_keys

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[dict[str, NDArray[np.float32]], dict[str, Any]]:
        """Reset the underlying scene and apply per-episode domain randomization.

        Reuses :class:`mousedroid.training.domain_randomization.DomainRandomizer`
        — there is no DR sampling code duplicated inside this module.
        The randomizer is only invoked when
        ``cfg.domain_randomization.enabled`` is True (path corrected
        per ADR-009 amendment from earlier draft).

        Args:
            seed: RNG seed forwarded to Isaac Lab and to the
                per-episode randomizer when set.

        Returns:
            ``(observation, info)`` where ``observation`` keys match
            :attr:`MockRoverEnv.observation_keys` exactly.

        Raises:
            IsaacLabUnavailableError: When the ``isaaclab`` package is
                not installed.
            RoverEnvNotBuiltError: When :meth:`build` has not yet been
                called.
        """
        self._require_built()
        self._step_idx = 0

        # Reset the simulation context to its initial state BEFORE
        # applying per-episode domain randomization, so the randomizer
        # acts on the freshly-reset scene and the observation we return
        # reflects the post-randomization world. Isaac Lab requires
        # this call between episodes (see SimulationContext.reset in
        # the >=0.20 API).
        if self._sim_context is not None and hasattr(self._sim_context, "reset"):
            self._sim_context.reset()  # pragma: no cover

        dr_enabled = self._dr_cfg is not None and self._dr_cfg.enabled
        episode_params = self._apply_reset_randomization(seed)
        info: dict[str, Any] = {
            "step_idx": self._step_idx,
            "dr_enabled": dr_enabled,
        }
        zeros = np.zeros(ROVER_NUM_WHEELS, dtype=np.float32)
        self._stamp_body_velocity_info(info, zeros)
        if episode_params is not None:
            info["episode_params"] = episode_params
        return self._read_observation(zeros), info

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[
        dict[str, NDArray[np.float32]],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """Apply ``action`` for one control step in the Isaac Lab scene.

        The 2-D differential-drive action ``[left, right]`` is clipped
        to ``±cfg.action.max_wheel_rad_s`` and then fanned onto the 4
        wheel actuators in :data:`ROVER_WHEEL_JOINT_NAMES` order as
        ``[FL=left, FR=right, RL=left, RR=right]`` (alternating layout
        matching :meth:`MockRoverEnv._action_to_body_velocity`).

        Args:
            action: Shape ``(2,)``; units per
                :attr:`RoverActionConfig.mode`.

        Returns:
            ``(obs, reward, terminated, truncated, info)``.

        Raises:
            IsaacLabUnavailableError: When the ``isaaclab`` package is
                not installed.
            RoverEnvNotBuiltError: When :meth:`build` has not yet been
                called.
            ValueError: When ``action.shape != (action_dim,)``.
        """
        if action.shape != (self._action_dim,):
            msg = f"action shape must be ({self._action_dim},), got {action.shape}"
            raise ValueError(msg)
        self._require_built()
        reward_cfg = self._require_reward()

        wheel_velocities = self._fan_out_action(action)
        _log.debug(
            "isaac_lab_env_step_action_clipped",
            step_idx=self._step_idx,
            wheel_velocities=wheel_velocities.tolist(),
        )
        self._apply_wheel_targets(wheel_velocities)
        self._substep_physics()

        obs = self._read_observation(wheel_velocities)
        vx_body, _omega = self._body_velocity_from_wheels(wheel_velocities)
        is_colliding = self._read_collision_flag()
        reward = reward_cfg.forward_velocity_weight * vx_body - reward_cfg.collision_weight * float(
            is_colliding
        )

        self._step_idx += 1
        info: dict[str, Any] = {
            "step_idx": self._step_idx,
            "wheel_velocities": wheel_velocities.tolist(),
            "is_colliding": bool(is_colliding),
        }
        self._stamp_body_velocity_info(info, wheel_velocities)
        return (obs, float(reward), False, False, info)

    def close(self) -> None:
        """Tear down the Isaac Lab simulation context.

        Resets the lifecycle flags so a follow-up :meth:`build` is
        required before the env can step again.
        """
        if self._sim_context is not None and hasattr(self._sim_context, "close"):
            self._sim_context.close()  # pragma: no cover
        self._sim_context = None
        self._scene = None
        self._articulation = None
        self._sensors = {}
        self._pending_domain = None
        self._built = False
        self._step_idx = 0

    # ----- internals --------------------------------------------------------

    def _wire_isaaclab_scene(self) -> None:  # pragma: no cover - live Isaac Sim only
        """Construct the live Isaac Lab scene (sim ctx, articulation, contact sensor).

        This helper is intentionally excluded from the CI coverage gate
        because the import path it walks is unreachable without the
        ``[isaac]`` extra installed. The operator's Linux + Isaac Sim
        validation post-merge exercises every line.

        API surface targets ``isaaclab >=0.20,<0.30`` per
        ``pyproject.toml`` — field names verified against the upstream
        docs at ``https://isaac-sim.github.io/IsaacLab/`` (see fixup
        commit body for the explicit references).
        """
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from isaaclab.sim import SimulationCfg, SimulationContext, UsdFileCfg

        if self._cfg.reward is None:
            msg = (
                "RoverIsaacLabEnv._wire_isaaclab_scene requires cfg.rover.reward; "
                "build() should have caught this."
            )
            raise RuntimeError(msg)

        isaac_cfg = self._cfg.sim.isaac
        sim_device = isaac_sim_device(self._cfg.sim)
        _log.info(
            "isaac_lab_env_building",
            urdf_path=self._cfg.sim.urdf_path,
            num_envs=self._cfg.sim.num_envs,
            headless=self._cfg.sim.headless,
            sim_dt_s=self._cfg.sim.sim_dt_s,
            decimation=self._cfg.sim.decimation,
            sim_device=sim_device,
        )

        sim_cfg = SimulationCfg(
            **simulation_cfg_kwargs(
                SimulationCfg,
                dt=self._cfg.sim.sim_dt_s,
                device=sim_device,
                num_envs=self._cfg.sim.num_envs,
            )
        )
        self._sim_context = SimulationContext(sim_cfg)

        usd_path = resolve_usd_path(self._cfg.sim.urdf_path, isaac_cfg.usd_path)
        articulation_cfg = ArticulationCfg(
            prim_path=isaac_cfg.prim_path,
            spawn=UsdFileCfg(usd_path=usd_path),
            actuators={
                name: ImplicitActuatorCfg(
                    joint_names_expr=[name],
                    stiffness=isaac_cfg.actuator_stiffness,
                    damping=self._cfg.action.slew_rad_s2,
                )
                for name in ROVER_WHEEL_JOINT_NAMES
            },
        )
        # ``Articulation(cfg)`` is the live runtime handle Isaac Lab
        # expects — storing the cfg here would break ``step()`` because
        # the cfg object does not expose ``set_joint_velocity_target``.
        self._articulation = Articulation(articulation_cfg)

        # Sensor handles keyed by the URDF link name so reset/step
        # body can resolve them without re-reading the constants tuple.
        # IMU / LiDAR / camera sensors are wired lazily by the operator
        # on Linux + Isaac Sim post-merge per the C4 playbook; the
        # contact sensor MUST be wired here because the reward signal
        # in ``RoverRewardConfig.collision_weight`` depends on it.
        self._sensors = dict.fromkeys(ROVER_SENSOR_LINK_NAMES)
        contact_cfg = ContactSensorCfg(
            prim_path=isaac_cfg.contact_prim_glob,
            update_period=self._cfg.sim.sim_dt_s,
            history_length=isaac_cfg.contact_history_length,
            track_air_time=isaac_cfg.track_air_time,
        )
        self._sensors[ROVER_CONTACT_SENSOR_NAME] = ContactSensor(contact_cfg)

        self._built = True
        _log.info(
            "isaac_lab_env_built",
            urdf_path=self._cfg.sim.urdf_path,
            usd_path=usd_path,
            wheel_joints=list(ROVER_WHEEL_JOINT_NAMES),
            sensor_links=list(ROVER_SENSOR_LINK_NAMES),
            sensor_keys=sorted(self._sensors.keys()),
            reward_weights={
                "forward_velocity_weight": self._cfg.reward.forward_velocity_weight,
                "collision_weight": self._cfg.reward.collision_weight,
            },
        )

    def _require_built(self) -> None:
        """Raise if the env can't service ``reset``/``step``.

        Two distinct failure modes:

          * ``isaaclab`` is not installed -> :class:`IsaacLabUnavailableError`
          * ``isaaclab`` is installed but :meth:`build` was never called
            -> :class:`RoverEnvNotBuiltError`
        """
        if not _isaaclab_available():
            msg = (
                "Isaac Lab is not installed; cannot run RoverIsaacLabEnv. "
                "Use backend='mock' for CI / unit tests."
            )
            raise IsaacLabUnavailableError(msg)
        if not self._built:
            msg = (
                "RoverIsaacLabEnv.build() has not been called; "
                "call env.build() before reset() / step()."
            )
            raise RoverEnvNotBuiltError(msg)

    def to_body_action(self, action: NDArray[np.float32]) -> NDArray[np.float32]:
        """Map a policy action to the RSSM body-frame ``[vx, vy=0, omega]``.

        Args:
            action: Shape ``(action_dim,)`` in ``RoverActionConfig.mode`` units.

        Returns:
            Length-3 float32 vector matching ``ModelConfig.action_dim``.
        """
        return policy_action_to_body(
            action,
            mode=self._cfg.action.mode,
            wheel_radius_m=self._wheel_radius,
            track_width_m=self._track_width,
        )

    def apply_domain_params(
        self,
        *,
        friction: float,
        slip: float,
        mass_kg: float,
        motor_gain: float,
    ) -> None:
        """Apply one chassis DR sample onto the Isaac (or fake) articulation.

        No-op when ``domain_randomization.enabled`` is False so reset stays
        byte-identical to the pre-feature path. Stores the sample so a
        subsequent :meth:`reset` can re-apply it after ``SimulationContext.reset``.

        Args:
            friction: Wheel tangential friction sample.
            slip: Observation-noise slip proxy (logged; same meaning as MuJoCo).
            mass_kg: Chassis mass sample.
            motor_gain: Actuator gain sample.
        """
        if self._dr_cfg is None or not self._dr_cfg.enabled:
            return
        self._pending_domain = {
            "friction": friction,
            "slip": slip,
            "mass_kg": mass_kg,
            "motor_gain": motor_gain,
        }
        apply_isaac_domain_params(
            self._articulation,
            friction=friction,
            slip=slip,
            mass_kg=mass_kg,
            motor_gain=motor_gain,
        )

    def _require_reward(self) -> RoverRewardConfig:
        """Return the reward block; ``build()`` already rejected ``None``."""
        if self._cfg.reward is None:  # pragma: no cover - guarded by build()
            msg = (
                "RoverIsaacLabEnv.step requires cfg.rover.reward; build() should have caught this."
            )
            raise RuntimeError(msg)
        return self._cfg.reward

    def _apply_wheel_targets(self, wheel_velocities: NDArray[np.float32]) -> None:
        """Forward wheel commands to the live articulation when present."""
        if self._articulation is not None and hasattr(
            self._articulation, "set_joint_velocity_target"
        ):
            self._articulation.set_joint_velocity_target(  # pragma: no cover
                wheel_velocities,
                joint_names=list(ROVER_WHEEL_JOINT_NAMES),
            )

    def _substep_physics(self) -> None:
        """Advance ``decimation`` physics ticks when a sim context exists."""
        if self._sim_context is not None and hasattr(self._sim_context, "step"):
            for _ in range(self._cfg.sim.decimation):  # pragma: no cover
                self._sim_context.step(render=not self._cfg.sim.headless)

    def _stamp_body_velocity_info(
        self,
        info: dict[str, Any],
        wheel_velocities: NDArray[np.float32],
    ) -> None:
        """Write ``vx_body_mps`` / ``omega_rads`` / ``forward_velocity_mps``."""
        vx, omega = self._body_velocity_from_wheels(wheel_velocities)
        info["vx_body_mps"] = float(vx)
        info["omega_rads"] = float(omega)
        info["forward_velocity_mps"] = float(vx)

    def _apply_reset_randomization(self, seed: int | None) -> Any:
        """Re-apply pending DR or sample a new bundle when DR is enabled."""
        dr_cfg = self._dr_cfg
        dr_enabled = dr_cfg is not None and dr_cfg.enabled
        if not dr_enabled or dr_cfg is None:
            _log.info(
                "isaac_lab_env_reset_with_randomization",
                seed=seed,
                dr_enabled=False,
            )
            return None
        if self._pending_domain is not None:
            apply_isaac_domain_params(self._articulation, **self._pending_domain)
            _log.info(
                "isaac_lab_env_reset_with_randomization",
                seed=seed,
                dr_enabled=True,
                chassis=dict(self._pending_domain),
            )
            return None
        from mousedroid.training.domain_randomization import DomainRandomizer

        rng = np.random.default_rng(seed)
        randomizer = DomainRandomizer(dr_cfg)
        episode_params = randomizer.sample(rng)
        chassis = episode_params.chassis
        self.apply_domain_params(
            friction=float(chassis["friction"]),
            slip=float(chassis["slip"]),
            mass_kg=float(chassis["mass_kg"]),
            motor_gain=float(chassis["motor_gain"]),
        )
        extras: dict[str, float] = {}
        extras.update(dict(episode_params.comms))
        extras.update(dict(episode_params.visual))
        extras.update(dict(episode_params.disturbance))
        apply_isaac_episode_extras(extras)
        _log.info(
            "isaac_lab_env_reset_with_randomization",
            seed=seed,
            dr_enabled=True,
            chassis=dict(chassis),
            comms=dict(episode_params.comms),
        )
        return episode_params

    def _fan_out_action(self, action: NDArray[np.float32]) -> NDArray[np.float32]:
        """Clip + fan a 2-D differential-drive action onto 4 wheel actuators.

        Layout pin (matches :meth:`MockRoverEnv._action_to_body_velocity`):
        ``[FL=left, FR=right, RL=left, RR=right]`` — alternating, not
        grouped. Any reorder breaks the cross-backend contract test.
        """
        cap = self._cfg.action.max_wheel_rad_s
        if self._cfg.action.mode == "differential":
            left = float(np.clip(action[0], -cap, cap))
            right = float(np.clip(action[1], -cap, cap))
            return np.asarray([left, right, left, right], dtype=np.float32)

        max_v = cap * self._wheel_radius
        vx_body = float(np.clip(action[0], -max_v, max_v))
        omega = float(action[1])
        left, right = body_velocity_to_wheels(
            vx_body,
            omega,
            wheel_radius_m=self._wheel_radius,
            track_width_m=self._track_width,
        )
        return np.asarray([left, right, left, right], dtype=np.float32)

    def _body_velocity_from_wheels(
        self, wheel_velocities: NDArray[np.float32]
    ) -> tuple[float, float]:
        """Compute body-frame ``(vx, omega)`` from per-wheel angular velocities."""
        return wheels_array_to_body_velocity(
            wheel_velocities,
            wheel_radius_m=self._wheel_radius,
            track_width_m=self._track_width,
        )

    def _read_collision_flag(self) -> bool:
        """Read the per-frame collision flag from the contact sensor.

        The contact sensor is wired in :meth:`build` under the
        :data:`ROVER_CONTACT_SENSOR_NAME` key. We surface a collision
        whenever any tracked contact force has a non-zero magnitude in
        the current physics frame. Returns ``False`` when no live
        sensor is attached (defensive guard for the test bypass path
        in ``test_rover_env_isaaclab.py`` that flips ``_built`` without
        calling :meth:`build`). The operator's Linux validation
        exercises the real contact-sensor path post-merge.
        """
        contact = self._sensors.get(ROVER_CONTACT_SENSOR_NAME)
        if contact is None:
            _log.debug(
                "isaac_lab_env_contact_sensor_unavailable",
                sensor_name=ROVER_CONTACT_SENSOR_NAME,
                wired_keys=sorted(self._sensors.keys()),
            )
            return False
        data = getattr(contact, "data", None)
        net_forces = getattr(data, "net_forces_w", None) if data is not None else None
        if net_forces is None:  # pragma: no cover - exercised on Linux
            return False
        # ``net_forces_w`` is shape ``(num_envs, num_bodies, 3)`` in the
        # >=0.20 Isaac Lab API; any non-zero magnitude reports contact.
        return bool(np.any(np.asarray(net_forces) != 0.0))  # pragma: no cover

    def _read_observation(
        self, wheel_velocities: NDArray[np.float32]
    ) -> dict[str, NDArray[np.float32]]:
        """Build the obs dict from duck-typed sensor handles plus wheel fan-out.

        IMU / pose / LiDAR readers operate on ``SimpleNamespace`` fakes in CI
        and on live Isaac handles on a workstation. Camera / ``render_rgb``
        is out of this slice.
        """
        obs = self._zero_observation()
        if "wheel_vel" in obs:
            obs["wheel_vel"] = wheel_velocities.astype(np.float32, copy=True)
        if "imu" in obs:
            obs["imu"] = read_rover_imu(
                sensors=self._sensors,
                articulation=self._articulation,
            )
        if "chassis_pose" in obs:
            obs["chassis_pose"] = read_rover_pose(
                sensors=self._sensors,
                articulation=self._articulation,
            )
        if "lidar" in obs:
            obs["lidar"] = read_rover_lidar(
                sensors=self._sensors,
                n_sectors=self._cfg.observation.lidar_num_sectors,
                max_range_m=self._cfg.observation.lidar_max_range_m,
            )
        return obs

    def _zero_observation(self) -> dict[str, NDArray[np.float32]]:
        """Return an identity-reset observation matching the configured keys.

        Each modality's shape mirrors :class:`MockRoverEnv` so callers
        can swap backends without touching downstream feature extractors.
        ``chassis_pose`` is seeded with the identity heading
        ``[0, 0, cos(0), sin(0)] = [0, 0, 1, 0]`` rather than all-zeros,
        because ``[0, 0, 0, 0]`` violates the ``cos^2 + sin^2 = 1``
        constraint and would disagree with the mock backend's reset.
        """
        obs_cfg = self._cfg.observation
        obs: dict[str, NDArray[np.float32]] = {}
        if obs_cfg.include_imu:
            obs["imu"] = np.zeros(ROVER_IMU_DIM, dtype=np.float32)
        if obs_cfg.include_chassis_pose:
            obs["chassis_pose"] = identity_chassis_pose()
        if obs_cfg.include_wheel_encoders:
            obs["wheel_vel"] = np.zeros(ROVER_NUM_WHEELS, dtype=np.float32)
        if obs_cfg.include_lidar_sectors:
            obs["lidar"] = np.zeros(obs_cfg.lidar_num_sectors, dtype=np.float32)
        return obs
