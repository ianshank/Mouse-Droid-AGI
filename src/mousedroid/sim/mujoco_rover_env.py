"""MuJoCo skid-steer rover environment (RoverEnvProtocol backend)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.config.schema import RoverConfig
from mousedroid.logging.setup import get_logger
from mousedroid.sim.protocols import (
    ROVER_CHASSIS_POSE_DIM,
    ROVER_IMU_DIM,
    ROVER_NUM_WHEELS,
)

_log = get_logger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Stable anchors in the base MJCF after which the env splices the full,
# config-driven N-sector lidar ring (sites after the IMU site; rangefinders after
# the gyro). The base MJCF ships NO lidar sites/sensors — the whole ring comes
# from MujocoSimConfig (invariant #3). Replacement is asserted (no silent no-op).
_SITE_ANCHOR = '<site name="imu_site" pos="0 0 0"/>'
_SENSOR_ANCHOR = '<gyro name="imu_gyro" site="imu_site"/>'

_WHEEL_GEOMS = ("g_fl", "g_fr", "g_rl", "g_rr")
_WHEEL_ACTUATORS = ("a_fl", "a_fr", "a_rl", "a_rr")


class RoverMuJoCoEnv:
    """4-wheel skid-steer rover backed by the MuJoCo classic engine.

    Conforms structurally to :class:`~mousedroid.sim.protocols.RoverEnvProtocol`
    and produces the SAME observation-dict contract as
    :class:`~mousedroid.sim.mock_rover_env.MockRoverEnv` (keys, shapes, FL/FR/
    RL/RR wheel order) so backends are interchangeable. ``mujoco`` is imported
    lazily so importing this module never requires the engine.
    """

    def __init__(self, cfg: RoverConfig, wheel_radius_m: float, track_width_m: float) -> None:
        """Initialise the MuJoCo rover env.

        Args:
            cfg: Rover configuration block from :class:`Settings`.
            wheel_radius_m: Wheel radius from :class:`RobotConfig`.
            track_width_m: Track width from :class:`RobotConfig`.
        """
        import mujoco

        self._mj = mujoco
        self._cfg = cfg
        self._mjcfg = cfg.sim.mujoco
        self._wheel_radius = wheel_radius_m
        self._track_width = track_width_m
        self._control_dt_s = cfg.sim.sim_dt_s * cfg.sim.decimation
        self._decimation = cfg.sim.decimation
        self._max_steps = max(1, int(cfg.sim.episode_length_s / self._control_dt_s))
        self._action_dim = cfg.action.action_dim
        self._obs_keys: tuple[str, ...] = cfg.observation.enabled_keys()
        self._lidar_sectors = self._mjcfg.lidar_num_sectors

        # Fail fast on lidar sector-count divergence: the MJCF injects
        # `mujoco.lidar_num_sectors` rangefinders, but the obs vector is read at
        # `observation.lidar_num_sectors`. A mismatch silently pads/truncates the
        # lidar vector and breaks the RSSM lidar head's expected width.
        if cfg.observation.include_lidar_sectors and (
            cfg.observation.lidar_num_sectors != self._lidar_sectors
        ):
            msg = (
                "lidar sector-count mismatch: observation.lidar_num_sectors="
                f"{cfg.observation.lidar_num_sectors} != sim.mujoco.lidar_num_sectors="
                f"{self._lidar_sectors}; they must match when lidar is enabled."
            )
            raise ValueError(msg)

        path = (_REPO_ROOT / self._mjcfg.mjcf_path).resolve()
        self._model = self._build_model(path)
        self._data = mujoco.MjData(self._model)
        self._wheel_vel = np.zeros(ROVER_NUM_WHEELS, dtype=np.float32)
        self._step_idx = 0
        self._noise_rng = np.random.default_rng(self._mjcfg.noise_rng_seed)
        self._slip_noise = self._mjcfg.wheel_slip_default
        self._closed = False
        self._renderer: Any = None  # lazily built on first render_rgb() call

        self._assert_rest_state_stable()
        _log.info(
            "mujoco_rover_env_initialised",
            mjcf=str(path),
            nu=int(self._model.nu),
            lidar_sectors=self._lidar_sectors,
            control_dt_s=self._control_dt_s,
        )

    # ----- model construction ----------------------------------------------

    def _build_model(self, path: Path) -> Any:
        """Load the MJCF and splice in the configured N-sector lidar fan."""
        if not path.exists():
            msg = f"MJCF not found at {path}"
            raise FileNotFoundError(msg)
        xml = path.read_text(encoding="utf-8")
        sites, sensors = self._lidar_fan_xml()
        xml = self._splice_after(xml, _SITE_ANCHOR, sites)
        xml = self._splice_after(xml, _SENSOR_ANCHOR, sensors)
        return self._mj.MjModel.from_xml_string(xml)

    @staticmethod
    def _splice_after(xml: str, anchor: str, injected: str) -> str:
        """Insert ``injected`` directly after ``anchor`` — raise if the anchor is absent.

        ``str.replace`` is a silent no-op when the target is missing; asserting the
        substitution actually fired prevents a drifted MJCF from compiling with a
        truncated lidar ring (which would crash ``_read_lidar`` later).
        """
        spliced = xml.replace(anchor, anchor + injected, 1)
        if spliced == xml:
            msg = f"MJCF injection anchor not found: {anchor!r}"
            raise ValueError(msg)
        return spliced

    def _lidar_fan_xml(self) -> tuple[str, str]:
        """Build the full ``<site>`` + ``<rangefinder>`` XML for sectors 0..N-1.

        Geometry (ring radius + mount height) comes from :class:`MujocoSimConfig`
        so a chassis change propagates through config, not hand-edited literals.
        """
        radius = self._mjcfg.lidar_ring_radius_m
        z = self._mjcfg.lidar_mount_z_m
        site_lines: list[str] = []
        sensor_lines: list[str] = []
        for i in range(self._lidar_sectors):
            ang = 2.0 * math.pi * i / self._lidar_sectors
            cx, sy = math.cos(ang), math.sin(ang)
            site_lines.append(
                f'\n      <site name="lidar_{i}" '
                f'pos="{radius * cx:.5f} {radius * sy:.5f} {z:.5f}" '
                f'zaxis="{cx:.5f} {sy:.5f} 0"/>'
            )
            sensor_lines.append(f'\n    <rangefinder name="lidar_s{i}" site="lidar_{i}"/>')
        return "".join(site_lines), "".join(sensor_lines)

    def _require_open(self) -> None:
        """Raise a clear error if the env was closed (vs an opaque NoneType crash)."""
        if self._closed:
            msg = "operation on a closed RoverMuJoCoEnv; build a new instance"
            raise RuntimeError(msg)

    def _assert_rest_state_stable(self) -> None:
        """Raise if the model free-falls / interpenetrates at rest (silent-NaN guard)."""
        self._mj.mj_forward(self._model, self._data)
        if not np.isfinite(self._data.qacc).all():
            msg = "MuJoCo rover unstable at rest: non-finite qacc (check wheel grounding)"
            raise RuntimeError(msg)

    # ----- protocol surface -------------------------------------------------

    @property
    def action_dim(self) -> int:
        """Return the action-space dimensionality (per ``RoverActionConfig.mode``)."""
        return self._action_dim

    @property
    def observation_keys(self) -> tuple[str, ...]:
        """Return the keys present in observation dicts."""
        return self._obs_keys

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[dict[str, NDArray[np.float32]], dict[str, Any]]:
        """Reset the env to its initial state.

        Args:
            seed: Optional RNG seed for the observation-noise (slip proxy) stream.

        Returns:
            ``(observation, info)``.
        """
        self._require_open()
        self._mj.mj_resetData(self._model, self._data)
        if seed is not None:
            self._noise_rng = np.random.default_rng(seed)
        self._step_idx = 0
        self._wheel_vel = np.zeros(ROVER_NUM_WHEELS, dtype=np.float32)
        self._mj.mj_forward(self._model, self._data)
        return self._observe(), {"step_idx": self._step_idx}

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[dict[str, NDArray[np.float32]], float, bool, bool, dict[str, Any]]:
        """Apply ``action`` for one control step (``decimation`` physics steps).

        Args:
            action: Shape ``(action_dim,)``; meaning per ``RoverActionConfig.mode``.

        Returns:
            ``(obs, reward, terminated, truncated, info)``.
        """
        self._require_open()
        if action.shape != (self._action_dim,):
            msg = f"action shape must be ({self._action_dim},), got {action.shape}"
            raise ValueError(msg)
        left, right = self._action_to_wheel_setpoints(action)
        # Wheel order FL, FR, RL, RR (parity with MockRoverEnv).
        self._data.ctrl[:] = np.asarray([left, right, left, right], dtype=np.float64)
        for _ in range(self._decimation):
            self._mj.mj_step(self._model, self._data)
        self._step_idx += 1

        obs = self._observe()
        goal = np.asarray(self._cfg.task.goal_xy_m, dtype=np.float32)
        px, py = float(self._data.qpos[0]), float(self._data.qpos[1])
        distance = float(np.hypot(goal[0] - px, goal[1] - py))
        reward = -distance
        truncated = self._step_idx >= self._max_steps
        terminated = distance < self._cfg.task.goal_reach_radius_m
        body_vx, omega = self._body_velocity()
        info: dict[str, Any] = {
            "step_idx": self._step_idx,
            "distance_to_goal_m": distance,
            "vx_body_mps": body_vx,
            "omega_rads": omega,
        }
        return obs, reward, terminated, truncated, info

    def render_rgb(self) -> NDArray[np.uint8]:
        """Render a forward-facing RGB frame from the configured camera.

        Lazily builds a ``mujoco.Renderer`` (offscreen) on first use — only the
        vision-fine-tune path pays the GL/render cost. Resolution + camera name
        come from :class:`MujocoSimConfig` (invariant #3).

        Returns:
            RGB frame, shape ``(render_height, render_width, 3)`` ``uint8``.

        Raises:
            RuntimeError: If the env is closed or ``render_vision`` is disabled.
        """
        self._require_open()
        if not self._mjcfg.render_vision:
            msg = "render_rgb() requires rover.sim.mujoco.render_vision=True"
            raise RuntimeError(msg)
        if self._renderer is None:
            self._renderer = self._mj.Renderer(
                self._model,
                height=self._mjcfg.render_height,
                width=self._mjcfg.render_width,
            )
        self._renderer.update_scene(self._data, camera=self._mjcfg.camera_name)
        frame: NDArray[np.uint8] = np.asarray(self._renderer.render(), dtype=np.uint8)
        return frame

    def close(self) -> None:
        """Release MuJoCo data + renderer (idempotent)."""
        self._closed = True
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._data = None

    # ----- domain randomization --------------------------------------------

    def apply_domain_params(
        self, *, friction: float, slip: float, mass_kg: float, motor_gain: float
    ) -> None:
        """Apply per-episode domain-randomization params to the live model.

        ``friction`` -> ``geom_friction[:,0]`` on wheel geoms; ``mass_kg`` ->
        chassis ``body_mass`` + inertia recompute; ``motor_gain`` ->
        ``actuator_gainprm[:,0]``; ``slip`` -> observation-noise magnitude
        (documented proxy; MuJoCo has no first-class slip). Edits the mjModel
        arrays in place (reload-free).
        """
        for name in _WHEEL_GEOMS:
            gid = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_GEOM, name)
            self._model.geom_friction[gid, 0] = friction
        bid = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_BODY, "chassis")
        scale = mass_kg / max(float(self._model.body_mass[bid]), 1e-6)
        self._model.body_mass[bid] = mass_kg
        self._model.body_inertia[bid] *= scale  # keep inertia consistent with new mass
        for name in _WHEEL_ACTUATORS:
            aid = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_ACTUATOR, name)
            self._model.actuator_gainprm[aid, 0] = motor_gain
        self._slip_noise = max(0.0, slip)
        self._assert_rest_state_stable()

    # ----- internals --------------------------------------------------------

    def _action_to_wheel_setpoints(self, action: NDArray[np.float32]) -> tuple[float, float]:
        cap = self._cfg.action.max_wheel_rad_s
        if self._cfg.action.mode == "differential":
            return float(np.clip(action[0], -cap, cap)), float(np.clip(action[1], -cap, cap))
        # body_velocity: [vx, omega] -> wheel setpoints
        vx, omega = float(action[0]), float(action[1])
        left = (vx - 0.5 * omega * self._track_width) / self._wheel_radius
        right = (vx + 0.5 * omega * self._track_width) / self._wheel_radius
        return float(np.clip(left, -cap, cap)), float(np.clip(right, -cap, cap))

    def to_body_action(self, action: NDArray[np.float32]) -> NDArray[np.float32]:
        """Map a policy action to the RSSM's body-frame ``[vx, vy=0, omega]``.

        Keeps the RSSM's action conditioning consistent with deployment
        (``ModelConfig.action_dim = [vx, vy, omega]``, body-frame) regardless of
        the env's action mode: differential WHEEL setpoints are converted through
        the rover kinematics (``vx = r(L+R)/2``, ``omega = r(R-L)/track``), and
        ``body_velocity`` ``[vx, omega]`` actions pass through. Training on raw
        wheel setpoints would otherwise mislabel them as body velocities and make
        the learned dynamics conditioning inconsistent with the deployed policy.
        """
        if self._cfg.action.mode == "differential":
            left, right = float(action[0]), float(action[1])
            vx = self._wheel_radius * (left + right) / 2.0
            omega = self._wheel_radius * (right - left) / self._track_width
        else:  # body_velocity: [vx, omega]
            vx, omega = float(action[0]), float(action[1])
        return np.asarray([vx, 0.0, omega], dtype=np.float32)

    def _body_velocity(self) -> tuple[float, float]:
        """Return ``(forward_speed_mps, yaw_rate_rads)`` from the freejoint qvel."""
        vx_world, vy_world = float(self._data.qvel[0]), float(self._data.qvel[1])
        omega = float(self._data.qvel[5])  # yaw rate (world z)
        theta = self._heading()
        forward = vx_world * math.cos(theta) + vy_world * math.sin(theta)
        return forward, omega

    def _heading(self) -> float:
        """Yaw about world-z from the freejoint quaternion ``qpos[3:7] = (w,x,y,z)``."""
        qw, qx, qy, qz = (float(self._data.qpos[i]) for i in range(3, 7))
        return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    def _observe(self) -> dict[str, NDArray[np.float32]]:
        obs: dict[str, NDArray[np.float32]] = {}
        oc = self._cfg.observation
        if oc.include_imu:
            obs["imu"] = self._read_imu()
        if oc.include_chassis_pose:
            obs["chassis_pose"] = self._read_pose()
        if oc.include_wheel_encoders:
            obs["wheel_vel"] = self._read_wheel_vel()
        if oc.include_lidar_sectors:
            obs["lidar"] = self._read_lidar(oc.lidar_num_sectors)
        return obs

    def _read_imu(self) -> NDArray[np.float32]:
        acc = self._sensor("imu_acc", 3)
        gyro = self._sensor("imu_gyro", 3)
        out: NDArray[np.float32] = np.concatenate([acc, gyro]).astype(np.float32)
        if out.shape != (ROVER_IMU_DIM,):
            msg = f"imu vector must be ({ROVER_IMU_DIM},), got {out.shape}"
            raise RuntimeError(msg)
        return out

    def _read_pose(self) -> NDArray[np.float32]:
        x, y = float(self._data.qpos[0]), float(self._data.qpos[1])
        theta = self._heading()
        pose = np.zeros(ROVER_CHASSIS_POSE_DIM, dtype=np.float32)
        pose[0], pose[1], pose[2], pose[3] = x, y, math.cos(theta), math.sin(theta)
        if self._slip_noise > 0.0:
            pose[:2] += self._noise_rng.normal(0.0, self._slip_noise, size=2).astype(np.float32)
        return pose

    def _read_wheel_vel(self) -> NDArray[np.float32]:
        # 4 hinge joint velocities live after the 6-DoF freejoint in qvel.
        wv = np.asarray(self._data.qvel[6 : 6 + ROVER_NUM_WHEELS], dtype=np.float32)
        if self._slip_noise > 0.0:
            noise = self._noise_rng.normal(0.0, self._slip_noise, size=wv.shape).astype(np.float32)
            wv = wv * (1.0 + noise)
        return wv

    def _read_lidar(self, n: int) -> NDArray[np.float32]:
        out = np.zeros(n, dtype=np.float32)
        rng = self._mjcfg.lidar_max_range_m
        for i in range(min(n, self._lidar_sectors)):
            raw = float(self._sensor(f"lidar_s{i}", 1)[0])
            # -1 sentinel (no hit) -> full range; normalise to [0,1].
            out[i] = 1.0 if raw < 0 else float(np.clip(raw / rng, 0.0, 1.0))
        return out

    def _sensor(self, name: str, dim: int) -> NDArray[np.float32]:
        sid = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            # mj_name2id returns -1 for an unknown name; fail fast rather than
            # indexing sensor_adr[-1] and silently reading the wrong sensor.
            msg = f"sensor {name!r} not found in the compiled MJCF model"
            raise ValueError(msg)
        adr = int(self._model.sensor_adr[sid])
        out: NDArray[np.float32] = np.asarray(
            self._data.sensordata[adr : adr + dim], dtype=np.float32
        )
        return out
