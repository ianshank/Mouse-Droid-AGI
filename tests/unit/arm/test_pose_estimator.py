"""Tests for 6-DoF pose estimator."""

from __future__ import annotations

import numpy as np

from mousedroid.arm.perception.pose_estimator import PoseEstimator
from mousedroid.arm.protocols import DetectedObject
from mousedroid.config.schema import ArmPerceptionConfig


def _make_estimator() -> PoseEstimator:
    """Create pose estimator with standard camera intrinsics."""
    cfg = ArmPerceptionConfig()
    intrinsics = np.array(
        [
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return PoseEstimator(cfg, intrinsics)


def _make_detection(cx: float = 320.0, cy: float = 240.0) -> DetectedObject:
    """Create a detection at given bounding box centre."""
    half_w = 25.0
    half_h = 25.0
    return DetectedObject(
        object_id="disk_1",
        class_name="disk_1",
        confidence=0.95,
        position_m=np.zeros(3, dtype=np.float64),
        orientation_rad=np.zeros(3, dtype=np.float64),
        bbox=np.array([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dtype=np.float64),
    )


class TestPoseEstimator:
    """Test PoseEstimator pose computation."""

    def test_centre_detection_gives_zero_xy(self) -> None:
        estimator = _make_estimator()
        det = _make_detection(cx=320.0, cy=240.0)
        depth = np.full((480, 640), 0.5, dtype=np.float32)
        pos, _ori = estimator.estimate_pose(det, depth)
        # Centre of image at (320, 240) with cx=320, cy=240 -> x=0, y=0
        assert abs(pos[0]) < 0.01
        assert abs(pos[1]) < 0.01
        assert abs(pos[2] - 0.5) < 0.01

    def test_off_centre_gives_nonzero_xy(self) -> None:
        estimator = _make_estimator()
        det = _make_detection(cx=400.0, cy=300.0)
        depth = np.full((480, 640), 0.5, dtype=np.float32)
        pos, _ = estimator.estimate_pose(det, depth)
        assert pos[0] > 0.0  # Right of centre
        assert pos[1] > 0.0  # Below centre

    def test_depth_used_for_z(self) -> None:
        estimator = _make_estimator()
        det = _make_detection()
        depth = np.full((480, 640), 1.0, dtype=np.float32)
        pos, _ = estimator.estimate_pose(det, depth)
        assert abs(pos[2] - 1.0) < 0.01

    def test_zero_depth_uses_fallback(self) -> None:
        estimator = _make_estimator()
        det = _make_detection()
        depth = np.zeros((480, 640), dtype=np.float32)
        pos, _ = estimator.estimate_pose(det, depth)
        assert pos[2] > 0.0  # Fallback depth

    def test_estimate_poses_batch(self) -> None:
        estimator = _make_estimator()
        dets = [_make_detection(cx=100.0), _make_detection(cx=500.0)]
        depth = np.full((480, 640), 0.5, dtype=np.float32)
        result = estimator.estimate_poses(dets, depth)
        assert len(result) == 2
        assert result[0].position_m[0] != result[1].position_m[0]
