"""Protocols for rover simulation backends.

The :class:`RoverEnvProtocol` is the minimal Gymnasium-compatible surface
that all rover backends (mock, Isaac Lab, MuJoCo) must satisfy. Concrete
backends are imported only inside factory functions (architectural
invariant #1).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class RoverEnvProtocol(Protocol):
    """Minimal env contract shared by mock + Isaac Lab + MuJoCo backends.

    Mirrors the Gymnasium API (`reset`, `step`, `action_space`,
    `observation_space`, `close`) without importing Gymnasium itself —
    the mock backend has no Gym dependency so this stays import-free at
    the protocol level.
    """

    @property
    def action_dim(self) -> int:
        """Return the action vector dimensionality (2 for differential)."""
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
