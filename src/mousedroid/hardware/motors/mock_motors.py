"""Mock motor driver for testing and simulation.

Simple mock that stores the last velocity command.
"""

from __future__ import annotations

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class MockMotorDriver:
    """Mock motor driver that records velocity commands.

    Stores last velocity for assertion in tests.
    """

    def __init__(self) -> None:
        """Initialise mock motor driver."""
        self._last_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    async def send_velocity(self, vx: float, vy: float, omega: float) -> None:
        """Store velocity command.

        Args:
            vx: Forward velocity in m/s.
            vy: Lateral velocity in m/s.
            omega: Angular velocity in rad/s.
        """
        self._last_velocity = (vx, vy, omega)
        _log.debug("mock_motor_velocity", vx=vx, vy=vy, omega=omega)

    async def emergency_stop(self) -> None:
        """Zero the stored velocity."""
        self._last_velocity = (0.0, 0.0, 0.0)
        _log.warning("mock_motor_emergency_stop")
