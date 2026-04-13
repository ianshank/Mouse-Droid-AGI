"""LiDAR feature extractor — converts raw scans to fixed-size feature vectors.

Divides 360 degrees into configurable angular sectors and computes the
minimum normalised distance per sector.  The result is a fixed-size float32
vector suitable for the world model encoder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.constants import LIDAR_FULL_ROTATION_DEG
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import LidarConfig
    from mousedroid.sensing.lidar_scan import LidarScan

_log = get_logger(__name__)


class LidarFeatureExtractor:
    """Extract sector-binned distance features from LiDAR scans.

    Divides 360 degrees into ``n_sectors`` bins.  Each bin gets the
    minimum distance within that sector, normalised to ``[0, 1]`` by
    ``max_range_m``.  A value of ``1.0`` means no obstacle detected.

    Args:
        cfg: LiDAR configuration with ``n_sectors`` and ``max_range_m``.
    """

    def __init__(self, cfg: LidarConfig) -> None:
        self._n_sectors = cfg.n_sectors
        self._max_range_m = cfg.max_range_m
        self._sector_width_deg = LIDAR_FULL_ROTATION_DEG / cfg.n_sectors
        _log.info(
            "lidar_feature_extractor_init",
            n_sectors=self._n_sectors,
            sector_width_deg=self._sector_width_deg,
        )

    @property
    def feature_dim(self) -> int:
        """Output feature vector dimension (equals ``n_sectors``)."""
        return self._n_sectors

    def extract(self, scan: LidarScan) -> NDArray[np.float32]:
        """Extract a fixed-size feature vector from a LiDAR scan.

        Args:
            scan: A :class:`LidarScan` containing angles and distances.

        Returns:
            Feature vector, shape ``(n_sectors,)``, values in ``[0, 1]``.
            Each element is ``min_distance_in_sector / max_range_m``.
            ``1.0`` indicates no obstacle detected in that sector.
        """
        features = np.ones(self._n_sectors, dtype=np.float32)

        if scan.n_points == 0:
            return features

        # Convert distances from mm to metres.
        distances_m = scan.distances_mm / 1000.0

        # Assign each point to its angular sector.
        sector_indices = np.clip(
            (scan.angles_deg / self._sector_width_deg).astype(np.intp),
            0,
            self._n_sectors - 1,
        )

        # Normalise distances to [0, 1].
        normalised = np.clip(distances_m / self._max_range_m, 0.0, 1.0)

        # Compute minimum normalised distance per sector (vectorized).
        np.minimum.at(features, sector_indices, normalised)

        return features
