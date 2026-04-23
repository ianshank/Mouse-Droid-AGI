"""Depth image processing and point cloud generation.

Converts raw depth frames from RealSense/OAK-D/ZED cameras into
processed depth maps and optional point clouds for downstream detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ArmPerceptionConfig

_log = get_logger(__name__)


class DepthProcessor:
    """Process raw depth frames for object detection and pose estimation.

    Applies noise filtering, hole filling, and optional point cloud
    generation from depth images.

    Args:
        cfg: Arm perception configuration.
        intrinsics: Camera intrinsic matrix (3x3).
    """

    def __init__(self, cfg: ArmPerceptionConfig, intrinsics: NDArray[np.float64]) -> None:
        """Initialise depth processor.

        Args:
            cfg: Arm perception configuration.
            intrinsics: Camera intrinsic matrix, shape ``(3, 3)``.
        """
        self._cfg = cfg
        self._intrinsics = intrinsics
        self._depth_min = cfg.depth_min_m
        self._depth_max = cfg.depth_max_m
        self._hole_threshold = cfg.depth_hole_threshold_m
        self._filter_kernel = cfg.depth_filter_kernel_size
        self._invalid_threshold = cfg.invalid_depth_threshold_m
        _log.info("depth_processor_init", camera_type=cfg.depth_camera_type)

    def filter_depth(self, depth_image: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply noise filtering and hole filling to raw depth image.

        Args:
            depth_image: Raw depth image, shape ``(H, W)``, values in metres.

        Returns:
            Filtered depth image with same shape.
        """
        # Clip invalid depths
        filtered = np.clip(depth_image, self._depth_min, self._depth_max)

        # Replace zeros (holes) with local median
        mask = filtered < self._hole_threshold
        if np.any(mask):
            _log.debug("depth_holes_detected", count=int(np.sum(mask)))
            from scipy.ndimage import median_filter

            filled = median_filter(filtered, size=self._filter_kernel)
            filtered[mask] = filled[mask]

        return filtered

    def depth_to_pointcloud(self, depth_image: NDArray[np.float32]) -> NDArray[np.float64]:
        """Convert depth image to 3D point cloud using camera intrinsics.

        Args:
            depth_image: Filtered depth image, shape ``(H, W)``, values in metres.

        Returns:
            Point cloud, shape ``(N, 3)`` where N = H*W valid points.
        """
        h, w = depth_image.shape
        fx = self._intrinsics[0, 0]
        fy = self._intrinsics[1, 1]
        cx = self._intrinsics[0, 2]
        cy = self._intrinsics[1, 2]

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_image.astype(np.float64)

        # Back-project to 3D
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        # Remove invalid points
        valid = points[:, 2] > self._invalid_threshold
        return cast(NDArray[np.float64], points[valid])
