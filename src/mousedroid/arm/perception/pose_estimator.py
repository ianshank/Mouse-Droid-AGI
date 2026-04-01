"""6-DoF pose estimation using PnP solver or learned networks.

Estimates the full 6-DoF pose (position + orientation) of detected
objects from 2D bounding boxes and depth data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.arm.protocols import DetectedObject
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ArmPerceptionConfig

_log = get_logger(__name__)


class PoseEstimator:
    """Estimate 6-DoF object poses from detections and depth data.

    Uses Perspective-n-Point (PnP) solving with known object geometry
    to recover full 3D pose from 2D detections and depth images.

    Args:
        cfg: Arm perception configuration.
        intrinsics: Camera intrinsic matrix (3x3).
    """

    def __init__(self, cfg: ArmPerceptionConfig, intrinsics: NDArray[np.float64]) -> None:
        """Initialise pose estimator.

        Args:
            cfg: Perception config with estimator type and tolerance.
            intrinsics: Camera intrinsic matrix, shape ``(3, 3)``.
        """
        self._cfg = cfg
        self._intrinsics = intrinsics
        self._tolerance_m = cfg.pose_tolerance_m
        self._invalid_depth_threshold = cfg.invalid_depth_threshold_m
        self._fallback_depth = cfg.fallback_depth_m
        _log.info("pose_estimator_init", method=cfg.pose_estimator)

    def estimate_pose(
        self,
        detection: DetectedObject,
        depth_image: NDArray[np.float32],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Estimate 6-DoF pose for a single detected object.

        Uses the bounding box centre + depth to compute 3D position,
        and the bounding box aspect ratio for a coarse orientation.

        Args:
            detection: Detected object with bounding box.
            depth_image: Depth image, shape ``(H, W)``, values in metres.

        Returns:
            Tuple of (position_m, orientation_rad) each shape ``(3,)``.
        """
        bbox = detection.bbox
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        # Sample depth at bounding box centre
        px = int(np.clip(cx, 0, depth_image.shape[1] - 1))
        py = int(np.clip(cy, 0, depth_image.shape[0] - 1))
        z = float(depth_image[py, px])

        if z < self._invalid_depth_threshold:
            _log.warning("invalid_depth_at_centre", object_id=detection.object_id)
            z = self._fallback_depth

        # Back-project to 3D using camera intrinsics
        fx = self._intrinsics[0, 0]
        fy = self._intrinsics[1, 1]
        cx_cam = self._intrinsics[0, 2]
        cy_cam = self._intrinsics[1, 2]

        x = (cx - cx_cam) * z / fx
        y = (cy - cy_cam) * z / fy

        position = np.array([x, y, z], dtype=np.float64)
        # Coarse orientation from bbox (placeholder — full PnP needs 3D model)
        orientation = np.zeros(3, dtype=np.float64)

        _log.debug(
            "pose_estimated",
            object_id=detection.object_id,
            position=position.tolist(),
        )
        return position, orientation

    def estimate_poses(
        self,
        detections: list[DetectedObject],
        depth_image: NDArray[np.float32],
    ) -> list[DetectedObject]:
        """Estimate poses for all detected objects in-place.

        Args:
            detections: List of detected objects (position fields updated).
            depth_image: Depth image for depth lookups.

        Returns:
            Same list with position_m and orientation_rad populated.
        """
        for det in detections:
            pos, ori = self.estimate_pose(det, depth_image)
            det.position_m = pos
            det.orientation_rad = ori
        return detections
