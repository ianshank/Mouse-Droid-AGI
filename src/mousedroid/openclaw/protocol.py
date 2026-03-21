"""OpenClaw gateway protocol — interface for high-level reasoning integration.

Defines ``OpenClawProtocol`` (the structural typing contract) and
``OpenClawActionResult`` (the immutable action payload).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class OpenClawActionResult:
    """Immutable action payload returned by the OpenClaw gateway.

    Attributes:
        action: Normalised action vector, shape ``(action_dim,)`` in ``[-1, 1]``.
        goal_id: Unique identifier for the active goal.
        reasoning: Human-readable reasoning text (for telemetry / debug).
        confidence: Confidence score in ``[0, 1]``.
        timestamp: Monotonic timestamp when the action was produced (seconds).
    """

    action: NDArray[np.float32]
    goal_id: str
    reasoning: str
    confidence: float
    timestamp: float

    def is_stale(self, max_age_ms: float) -> bool:
        """Check whether this result has exceeded its freshness window.

        Args:
            max_age_ms: Maximum allowed age in milliseconds.

        Returns:
            ``True`` if the result is older than *max_age_ms*.
        """
        age_ms = (time.monotonic() - self.timestamp) * 1_000.0
        return age_ms > max_age_ms


@runtime_checkable
class OpenClawProtocol(Protocol):
    """Structural interface for OpenClaw gateway implementations.

    All methods are async to support both local mock and remote HTTP
    gateways without blocking the 30 Hz control loop.
    """

    async def start(self) -> None:
        """Establish connection to the OpenClaw service."""
        ...

    async def stop(self) -> None:
        """Disconnect and release resources."""
        ...

    async def get_action(
        self,
        observation_dict: dict[str, Any],
    ) -> OpenClawActionResult | None:
        """Request an action from the OpenClaw reasoning layer.

        Args:
            observation_dict: Current robot state (sensor data, safety, etc.).

        Returns:
            An ``OpenClawActionResult`` or ``None`` if unavailable.
        """
        ...

    async def set_goal(self, goal: str) -> None:
        """Update the high-level goal / mission.

        Args:
            goal: Natural-language goal description.
        """
        ...

    @property
    def is_connected(self) -> bool:
        """Whether the gateway currently has an active connection."""
        ...
