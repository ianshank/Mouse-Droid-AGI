"""Protocols and shared invariants for rover simulation backends.

The :class:`RoverEnvProtocol` is the minimal Gymnasium-compatible surface
that all rover backends (mock, Isaac Lab, MuJoCo) must satisfy. Concrete
backends are imported only inside factory functions (architectural
invariant #1).

The ``ROVER_*`` module constants pin the observation-modality dimensions
that come from physical / format invariants (IMU layout, chassis-pose
encoding, URDF wheel count). Tunable quantities live on
:class:`RoverObservationConfig` (e.g. ``lidar_num_sectors``).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# Linear-accel (3) + angular-velocity (3) vector — fixed by the IMU
# modality definition recorded on RoverObservationConfig.include_imu.
ROVER_IMU_DIM: int = 6

# Body-frame pose as ``[x, y, cos(theta), sin(theta)]`` — the cos/sin
# encoding keeps the heading continuous for the policy network.
ROVER_CHASSIS_POSE_DIM: int = 4

# URDF (assets/rover/mse6_4wd.urdf) ships four continuous wheel joints.
ROVER_NUM_WHEELS: int = 4


@runtime_checkable
class RoverEnvProtocol(Protocol):
    """Minimal env contract shared by mock + Isaac Lab + MuJoCo backends.

    The surface intentionally mirrors the Gymnasium *lifecycle methods*
    (``reset``, ``step``, ``close``) but deliberately exposes
    :attr:`action_dim` + :attr:`observation_keys` instead of the heavier
    ``action_space`` / ``observation_space`` Gymnasium ``Space`` objects.
    That keeps the protocol import-free of ``gymnasium`` so the mock
    backend can satisfy it with stdlib + NumPy only; concrete backends
    are free to expose proper Gymnasium spaces in addition.
    """

    @property
    def action_dim(self) -> int:
        """Return the action vector dimensionality (per ``RoverActionConfig.mode``)."""
        ...

    @property
    def observation_keys(self) -> tuple[str, ...]:
        """Return the keys present in observation dicts produced by ``step``."""
        ...

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[dict[str, NDArray[np.float32]], dict[str, Any]]:
        """Reset the environment to an initial state.

        Args:
            seed: Optional RNG seed for deterministic episodes.

        Returns:
            ``(observation, info)`` where ``observation`` is a dict keyed
            by :attr:`observation_keys`.
        """
        ...

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
        """Apply ``action`` for one control step.

        Args:
            action: Shape ``(action_dim,)``, units depend on
                :attr:`RoverActionConfig.mode` (rad/s for differential,
                m/s + rad/s for body_velocity).

        Returns:
            ``(obs, reward, terminated, truncated, info)``.
        """
        ...

    def close(self) -> None:
        """Release any backend-held resources."""
        ...
