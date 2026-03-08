"""Mock ultrasonic distance sensor for testing and simulation.

Implements ``DistanceSensorProtocol`` with configurable return values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import UltrasonicConfig

_log = get_logger(__name__)


class MockUltrasonic:
    """Mock HC-SR04 implementing ``DistanceSensorProtocol``.

    Returns a configurable distance value for test control.
    """

    def __init__(self, cfg: UltrasonicConfig) -> None:
        """Initialise mock ultrasonic from config.

        Args:
            cfg: Ultrasonic sensor configuration.
        """
        self._cfg = cfg
        self._distance: float = (cfg.max_range_m + cfg.min_range_m) / 2.0
        _log.info("mock_ultrasonic_init", max_range=cfg.max_range_m, min_range=cfg.min_range_m)

    async def read_distance_m(self) -> float:
        """Return the configured mock distance.

        Returns:
            Distance in metres.
        """
        return self._distance

    @property
    def max_range_m(self) -> float:
        """Maximum detection range in metres."""
        return self._cfg.max_range_m

    @property
    def min_range_m(self) -> float:
        """Minimum detection range in metres."""
        return self._cfg.min_range_m

    def set_distance(self, d: float) -> None:
        """Set the distance value returned by ``read_distance_m``.

        Args:
            d: Distance in metres.
        """
        self._distance = d
