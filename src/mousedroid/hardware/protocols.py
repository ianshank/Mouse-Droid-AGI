"""Hardware abstraction protocols for vision and distance sensors.

All hardware interfaces use ``@runtime_checkable`` structural typing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class VisionProtocol(Protocol):
    """Interface for all vision drivers (IMX500, mock, future cameras)."""

    async def capture_features(self) -> NDArray[np.float32]:
        """Capture and extract vision feature vector.

        Returns:
            Feature vector, shape ``(feature_dim,)``.
        """
        ...

    @property
    def feature_dim(self) -> int:
        """Output feature vector dimension."""
        ...

    async def start(self) -> None:
        """Start camera capture pipeline."""
        ...

    async def stop(self) -> None:
        """Stop camera capture pipeline."""
        ...


@runtime_checkable
class DistanceSensorProtocol(Protocol):
    """Interface for all distance sensors (HC-SR04, future LiDAR, etc)."""

    async def read_distance_m(self) -> float:
        """Read distance measurement.

        Returns:
            Distance in metres. Returns max_range_m if no echo detected.
        """
        ...

    @property
    def max_range_m(self) -> float:
        """Maximum detection range in metres."""
        ...

    @property
    def min_range_m(self) -> float:
        """Minimum detection range in metres."""
        ...
