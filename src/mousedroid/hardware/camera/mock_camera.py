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
        frame = await self.capture_frame()
        return await self.extract_features(frame)

    async def capture_frame(self) -> NDArray[np.uint8]:
        """Return random BGR frame.

        Returns:
            Random uint8 frame of shape ``(H, W, 3)``.
        """
        return self._rng.integers(
            0, 256,
            size=(self._cfg.resolution_height, self._cfg.resolution_width, 3),
            dtype=np.uint8,
        )

    async def extract_features(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Extract feature vector from an existing frame."""
        flat = frame.astype(np.float32).flatten()
        dim = self._cfg.feature_dim
        if len(flat) >= dim:
            stride = len(flat) // dim
            features = flat[: stride * dim].reshape(dim, stride).mean(axis=1)
        else:
            features = np.zeros(dim, dtype=np.float32)
            features[: len(flat)] = flat
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return np.asarray(features, dtype=np.float32)

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
