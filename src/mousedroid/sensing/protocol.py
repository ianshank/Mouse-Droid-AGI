"""Observation protocol — common interface for all observation bundles."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class ObservationProtocol(Protocol):
    """Common interface for all observation bundles.

    Platform-specific bundles expose at minimum these fields.
    """

    @property
    def timestamp(self) -> float:
        """Monotonic timestamp in seconds."""
        ...

    @property
    def vision_features(self) -> NDArray[np.float32]:
        """Vision feature vector, shape ``(feature_dim,)``."""
        ...

    @property
    def distance_m(self) -> float:
        """Forward distance measurement in metres."""
        ...

    @property
    def motor_state(self) -> NDArray[np.float32]:
        """Motor state ``[vx, vy, omega, battery_v]``, shape ``(4,)``."""
        ...

    @property
    def valid_mask(self) -> NDArray[np.float32]:
        """Per-sensor validity scores, shape ``(n_modalities,)``."""
        ...

    @property
    def n_modalities(self) -> int:
        """Number of sensor modalities tracked by valid_mask."""
        ...
