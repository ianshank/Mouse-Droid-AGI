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
        """Motor state ``[left_wheel_mps, right_wheel_mps, heading_rad, battery_v]``.

        Shape ``(4,)``. Corrected 2026-09-19 (peer review D-11): this was
        documented as ``[vx, vy, omega, battery_v]`` here and in five other
        places, but :meth:`~mousedroid.sensing.manager.SensorManager._safe_motor_read`
        packs *per-wheel* speeds in slots 0/1 and an **absolute heading angle**
        in slot 2, not a body-frame velocity pair and an angular rate.

        Note the sim pretraining adapter
        (:mod:`mousedroid.training.rover_obs_adapter`) genuinely packs
        ``[vx, 0.0, omega, battery_v]`` -- body-frame linear velocity and an
        angular *rate*. That is a real train/serve difference in slot 2, not a
        documentation error on either side.
        """
        ...

    @property
    def audio_chunk(self) -> NDArray[np.float32]:
        """Audio samples, shape ``(chunk_size * channels,)``."""
        ...

    @property
    def lidar_features(self) -> NDArray[np.float32] | None:
        """LiDAR sector-binned features, shape ``(lidar_dim,)``, or ``None``."""
        ...

    @property
    def valid_mask(self) -> NDArray[np.float32]:
        """Per-sensor validity scores, shape ``(n_modalities,)``."""
        ...

    @property
    def n_modalities(self) -> int:
        """Number of sensor modalities tracked by valid_mask."""
        ...
