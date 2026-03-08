"""Mock camera driver for testing and simulation.

Implements ``VisionProtocol`` with random feature vectors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import CameraConfig

_log = get_logger(__name__)


class MockCamera:
    """Mock camera implementing ``VisionProtocol``.

    Returns random feature vectors of the configured dimension.
    """

    def __init__(self, cfg: CameraConfig) -> None:
        """Initialise mock camera from config.

        Args:
            cfg: Camera configuration.
        """
        self._cfg = cfg
        self._rng = np.random.default_rng()
        _log.info("mock_camera_init", feature_dim=cfg.feature_dim)

    async def capture_features(self) -> NDArray[np.float32]:
        """Return random feature vector.

        Returns:
            Random feature vector of shape ``(feature_dim,)``.
        """
        return self._rng.standard_normal(self._cfg.feature_dim).astype(np.float32)

    @property
    def feature_dim(self) -> int:
        """Output feature vector dimension."""
        return self._cfg.feature_dim

    async def start(self) -> None:
        """Simulate starting the camera pipeline."""
        _log.info("mock_camera_started")

    async def stop(self) -> None:
        """Simulate stopping the camera pipeline."""
        _log.info("mock_camera_stopped")
