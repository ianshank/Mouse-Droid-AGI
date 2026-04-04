"""Tests for depth image processing."""

from __future__ import annotations

import numpy as np
import pytest

from mousedroid.arm.perception.depth_processor import DepthProcessor
from mousedroid.config.schema import ArmPerceptionConfig


def _make_processor() -> DepthProcessor:
    cfg = ArmPerceptionConfig()
    intrinsics = np.array(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return DepthProcessor(cfg, intrinsics)


class TestFilterDepth:
    """Tests for depth filtering."""

    def test_clips_to_valid_range(self) -> None:
        pytest.importorskip("scipy")
        proc = _make_processor()
        depth = np.array([[0.005, 15.0], [0.5, 1.0]], dtype=np.float32)
        filtered = proc.filter_depth(depth)
        assert filtered.min() >= 0.01
        assert filtered.max() <= 10.0

    def test_fills_holes(self) -> None:
        pytest.importorskip("scipy")
        proc = _make_processor()
        depth = np.array(
            [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        filtered = proc.filter_depth(depth)
        # Centre hole should be filled
        assert filtered[1, 1] > 0.02

    def test_no_holes_returns_clipped(self) -> None:
        proc = _make_processor()
        depth = np.full((4, 4), 2.0, dtype=np.float32)
        filtered = proc.filter_depth(depth)
        np.testing.assert_allclose(filtered, 2.0)

    def test_output_shape_matches_input(self) -> None:
        proc = _make_processor()
        depth = np.random.default_rng(0).uniform(0.5, 3.0, (10, 12)).astype(np.float32)
        filtered = proc.filter_depth(depth)
        assert filtered.shape == (10, 12)


class TestDepthToPointcloud:
    """Tests for point cloud generation."""

    def test_valid_depth_produces_points(self) -> None:
        proc = _make_processor()
        depth = np.full((5, 5), 1.0, dtype=np.float32)
        points = proc.depth_to_pointcloud(depth)
        assert points.shape[1] == 3
        assert points.shape[0] > 0

    def test_zero_depth_excluded(self) -> None:
        proc = _make_processor()
        depth = np.zeros((3, 3), dtype=np.float32)
        points = proc.depth_to_pointcloud(depth)
        assert points.shape[0] == 0

    def test_z_values_match_depth(self) -> None:
        proc = _make_processor()
        depth = np.full((4, 4), 2.5, dtype=np.float32)
        points = proc.depth_to_pointcloud(depth)
        np.testing.assert_allclose(points[:, 2], 2.5)

    def test_uses_config_threshold(self) -> None:
        cfg = ArmPerceptionConfig(invalid_depth_threshold_m=0.5)
        intrinsics = np.eye(3, dtype=np.float64)
        intrinsics[0, 0] = intrinsics[1, 1] = 100.0
        proc = DepthProcessor(cfg, intrinsics)
        depth = np.full((3, 3), 0.3, dtype=np.float32)
        points = proc.depth_to_pointcloud(depth)
        # All below 0.5 threshold, so all excluded
        assert points.shape[0] == 0
