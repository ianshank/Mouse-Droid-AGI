"""Mock OpenClaw gateway for testing without a real OpenClaw service.

Provides controllable behaviour via ``set_*`` methods and tracks call
history for test assertions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.config.schema import OpenClawConfig
from mousedroid.logging.setup import get_logger
from mousedroid.openclaw.protocol import OpenClawActionResult

_log = get_logger(__name__)


class MockOpenClawGateway:
    """Mock implementation of ``OpenClawProtocol`` for tests.

    Args:
        cfg: OpenClaw configuration (used for action_dim inference).
    """

    def __init__(self, cfg: OpenClawConfig) -> None:
        self._cfg = cfg
        self._connected = False
        self._started = False
        self._next_action: OpenClawActionResult | None = None

        # Call tracking for test assertions
        self.start_calls: int = 0
        self.stop_calls: int = 0
        self.action_calls: list[dict[str, Any]] = []
        self.goal_calls: list[str] = []

    async def start(self) -> None:
        """Simulate connection establishment."""
        self._started = True
        self._connected = True
        self.start_calls += 1
        _log.debug("mock_openclaw_started")

    async def stop(self) -> None:
        """Simulate disconnection."""
        self._started = False
        self._connected = False
        self.stop_calls += 1
        _log.debug("mock_openclaw_stopped")

    async def get_action(
        self,
        observation_dict: dict[str, Any],
    ) -> OpenClawActionResult | None:
        """Return the pre-configured action (or ``None``).

        Args:
            observation_dict: Current observation (stored for assertions).

        Returns:
            Whatever was set via ``set_action()``, default ``None``.
        """
        self.action_calls.append(observation_dict)
        return self._next_action

    async def set_goal(self, goal: str) -> None:
        """Record the goal for test assertions.

        Args:
            goal: Natural-language goal.
        """
        self.goal_calls.append(goal)
        _log.debug("mock_openclaw_goal_set", goal=goal)

    @property
    def is_connected(self) -> bool:
        """Whether the mock reports as connected."""
        return self._connected

    # -- Test control methods --------------------------------------------------

    def set_action(self, result: OpenClawActionResult | None) -> None:
        """Configure the next action to return from ``get_action()``.

        Args:
            result: Action result, or ``None`` to simulate unavailability.
        """
        self._next_action = result

    def set_connected(self, connected: bool) -> None:
        """Override connection state.

        Args:
            connected: Desired connection state.
        """
        self._connected = connected

    def make_action(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        omega: float = 0.0,
        goal_id: str = "test-goal",
        confidence: float = 1.0,
    ) -> OpenClawActionResult:
        """Convenience factory for creating test actions.

        Args:
            vx: Forward velocity in ``[-1, 1]``.
            vy: Lateral velocity in ``[-1, 1]``.
            omega: Angular velocity in ``[-1, 1]``.
            goal_id: Goal identifier.
            confidence: Confidence score.

        Returns:
            Frozen ``OpenClawActionResult``.
        """
        action: NDArray[np.float32] = np.array([vx, vy, omega], dtype=np.float32)
        return OpenClawActionResult(
            action=action,
            goal_id=goal_id,
            reasoning="mock_action",
            confidence=confidence,
            timestamp=0.0,
        )

    def reset(self) -> None:
        """Reset all call history and state."""
        self._next_action = None
        self._connected = self._started
        self.start_calls = 0
        self.stop_calls = 0
        self.action_calls.clear()
        self.goal_calls.clear()
