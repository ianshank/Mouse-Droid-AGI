"""Isaac Lab environment stub for the 4WD MSE-6 rover.

This module is a **stub**: it locks in the action / observation contract
and lazy-imports Isaac Lab so static analysers and CI machines without
GPUs can still load it. A real Isaac Lab installation is required to
instantiate the env via :meth:`RoverIsaacLabEnv.build`; without it,
calling ``build`` raises :class:`IsaacLabUnavailableError`.

The env wraps ``isaaclab.envs.ManagerBasedRLEnv``-style scenes once Isaac
Lab is on the path. Until then, the class exists only to document the
shape of the action / observation spaces and to give factory wiring a
concrete return type to typecheck against.

Phase B will fill in:
  - URDF -> USD import,
  - articulation actuator wiring for the 4 continuous wheel joints,
  - IMU / contact / LiDAR sensor handles,
  - per-episode domain randomization from
    :mod:`mousedroid.training.domain_randomization`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.config.schema import RoverConfig
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class IsaacLabUnavailableError(RuntimeError):
    """Raised when the Isaac Lab backend is requested without the dep installed."""


class RoverEnvNotBuiltError(RuntimeError):
    """Raised when ``reset``/``step`` is called before :meth:`build`."""


def _isaaclab_available() -> bool:
    """Return ``True`` iff Isaac Lab can be imported in the current env."""
    try:
        import isaaclab  # noqa: F401
    except ImportError:
        return False
    return True


class RoverIsaacLabEnv:
    """Isaac Lab env stub conforming to :class:`RoverEnvProtocol`.

    The constructor does **not** initialise Isaac Lab; call
    :meth:`build` to do so. This split keeps the class importable
    without GPU drivers; tests that need a live backend skip with
    ``pytest.skipif(not _isaaclab_available(), ...)``.
    """

    def __init__(self, cfg: RoverConfig, wheel_radius_m: float, track_width_m: float) -> None:
        """Initialise the env wrapper.

        Args:
            cfg: Rover configuration block from :class:`Settings`.
            wheel_radius_m: Wheel radius from :class:`RobotConfig`.
            track_width_m: Track width from :class:`RobotConfig`.
        """
        self._cfg = cfg
        self._wheel_radius = wheel_radius_m
        self._track_width = track_width_m
        self._sim: Any = None  # Populated by ``build``.
        self._scene: Any = None
        self._built: bool = False
        self._step_idx: int = 0
        self._action_dim = 2
        self._obs_keys: tuple[str, ...] = cfg.observation.enabled_keys()

    # ----- lifecycle --------------------------------------------------------

    def build(self) -> None:
        """Initialise the underlying Isaac Lab simulation context.

        Raises:
            IsaacLabUnavailableError: When ``isaaclab`` cannot be imported.
        """
        if not _isaaclab_available():
            msg = (
                "Isaac Lab is not installed in this environment. "
                'Install with `pip install -e ".[isaac]"` on a workstation '
                "with NVIDIA Isaac Lab pre-requisites."
            )
            raise IsaacLabUnavailableError(msg)

        # Imports deliberately inside ``build`` — they pull in CUDA-bound
        # heavy deps that should never be touched at module import time.
        import isaaclab  # noqa: F401  # pragma: no cover

        _log.info(
            "rover_isaaclab_env_build_requested",
            urdf_path=self._cfg.sim.urdf_path,
            num_envs=self._cfg.sim.num_envs,
            headless=self._cfg.sim.headless,
        )
        # TODO(Phase B): construct ManagerBasedRLEnv / scene, import the
        # URDF as USD, attach IMU + LiDAR sensors at imu_link / lidar_link,
        # wire actuators to the four continuous wheel joints, and apply
        # ``EpisodeParams.chassis`` domain randomization on reset.
        self._built = True

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
        """Reset the underlying scene.

        Args:
            seed: RNG seed forwarded to Isaac Lab when available.

        Returns:
            ``(observation, info)``.

        Raises:
            IsaacLabUnavailableError: When the ``isaaclab`` package is
                not installed.
            RoverEnvNotBuiltError: When :meth:`build` has not yet been
                called.
        """
        self._require_built()
        # TODO(Phase B): scene.reset, sensor zero, randomization sample.
        _log.debug("rover_isaaclab_env_reset_stub", seed=seed)
        self._step_idx = 0
        return self._zero_observation(), {"step_idx": self._step_idx}

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

        Args:
            action: Shape ``(2,)``, units per
                :attr:`RoverActionConfig.mode`.

        Returns:
            ``(obs, reward, terminated, truncated, info)``.

        Raises:
            IsaacLabUnavailableError: When the ``isaaclab`` package is
                not installed.
            RoverEnvNotBuiltError: When :meth:`build` has not yet been
                called.
        """
        if action.shape != (self._action_dim,):
            msg = f"action shape must be ({self._action_dim},), got {action.shape}"
            raise ValueError(msg)
        self._require_built()
        # TODO(Phase B): forward wheel-velocity commands to articulation
        # actuators, sub-step ``decimation`` physics ticks, read sensors,
        # compute reward, return.
        self._step_idx += 1
        return (
            self._zero_observation(),
            0.0,
            False,
            False,
            {"step_idx": self._step_idx},
        )

    def close(self) -> None:
        """Tear down the Isaac Lab simulation context."""
        if self._sim is not None:
            # TODO(Phase B): self._sim.close() — Isaac Lab teardown.
            self._sim = None
            self._scene = None
        self._built = False
        self._step_idx = 0

    # ----- internals --------------------------------------------------------

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

    def _zero_observation(self) -> dict[str, NDArray[np.float32]]:
        """Return a zero-valued observation matching the configured keys."""
        obs_cfg = self._cfg.observation
        obs: dict[str, NDArray[np.float32]] = {}
        if obs_cfg.include_imu:
            obs["imu"] = np.zeros(6, dtype=np.float32)
        if obs_cfg.include_chassis_pose:
            obs["chassis_pose"] = np.zeros(4, dtype=np.float32)
        if obs_cfg.include_wheel_encoders:
            obs["wheel_vel"] = np.zeros(4, dtype=np.float32)
        if obs_cfg.include_lidar_sectors:
            obs["lidar"] = np.zeros(obs_cfg.lidar_num_sectors, dtype=np.float32)
        return obs
