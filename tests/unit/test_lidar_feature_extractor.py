"""Tests for LidarFeatureExtractor."""

from __future__ import annotations

import numpy as np
import pytest

from mousedroid.config.schema import LidarConfig
from mousedroid.hardware.lidar.feature_extractor import LidarFeatureExtractor
from mousedroid.sensing.lidar_scan import LidarScan, empty_scan


@pytest.fixture
def lidar_cfg() -> LidarConfig:
    """Default LidarConfig for testing."""
    return LidarConfig(
        enabled=True,
        serial_port="/dev/ttyUSB1",
        baud_rate=230400,
        max_range_m=12.0,
        min_range_m=0.15,
        scan_frequency_hz=10.0,
        min_confidence=0,
        read_timeout_s=0.2,
        n_sectors=36,
        feature_dim=36,
    )


@pytest.fixture
def extractor(lidar_cfg: LidarConfig) -> LidarFeatureExtractor:
    """LidarFeatureExtractor instance."""
    return LidarFeatureExtractor(lidar_cfg)


def test_feature_dim_matches_n_sectors(extractor: LidarFeatureExtractor) -> None:
    """feature_dim equals n_sectors from config."""
    assert extractor.feature_dim == 36


def test_empty_scan_returns_all_ones(extractor: LidarFeatureExtractor) -> None:
    """Empty scan produces all-ones feature vector (no obstacles)."""
    scan = empty_scan()
    features = extractor.extract(scan)
    assert features.shape == (36,)
    np.testing.assert_array_equal(features, np.ones(36, dtype=np.float32))


def test_uniform_scan_produces_uniform_features(extractor: LidarFeatureExtractor) -> None:
    """Uniform distance at 6m produces uniform normalised features."""
    n_points = 360
    angles = np.linspace(0.0, 359.0, num=n_points, dtype=np.float32)
    distances_mm = np.full(n_points, 6000.0, dtype=np.float32)  # 6m
    confidences = np.full(n_points, 200, dtype=np.uint8)
    scan = LidarScan(
        angles_deg=angles,
        distances_mm=distances_mm,
        confidences=confidences,
        timestamp=0.0,
        n_points=n_points,
    )
    features = extractor.extract(scan)
    assert features.shape == (36,)
    # 6000mm = 6m / 12m max = 0.5
    np.testing.assert_allclose(features, 0.5, atol=0.05)


def test_single_obstacle_in_one_sector(extractor: LidarFeatureExtractor) -> None:
    """A close point in one sector reduces that sector's feature value."""
    n_points = 360
    angles = np.linspace(0.0, 359.0, num=n_points, dtype=np.float32)
    distances_mm = np.full(n_points, 12000.0, dtype=np.float32)  # max range
    confidences = np.full(n_points, 200, dtype=np.uint8)
    # Place obstacle at 1m in the first sector (0-10 degrees)
    distances_mm[0] = 1000.0  # 1m
    scan = LidarScan(
        angles_deg=angles,
        distances_mm=distances_mm,
        confidences=confidences,
        timestamp=0.0,
        n_points=n_points,
    )
    features = extractor.extract(scan)
    # Sector 0 should have min(1000/12000) ≈ 0.083
    assert features[0] < 0.1
    # Other sectors should be ~1.0
    assert features[18] > 0.9


def test_values_bounded_zero_to_one(extractor: LidarFeatureExtractor) -> None:
    """All feature values must be in [0, 1]."""
    rng = np.random.default_rng(42)
    angles = np.linspace(0.0, 359.0, num=100, dtype=np.float32)
    distances_mm = rng.uniform(150.0, 12000.0, size=100).astype(np.float32)
    confidences = np.full(100, 200, dtype=np.uint8)
    scan = LidarScan(
        angles_deg=angles,
        distances_mm=distances_mm,
        confidences=confidences,
        timestamp=0.0,
        n_points=100,
    )
    features = extractor.extract(scan)
    assert np.all(features >= 0.0)
    assert np.all(features <= 1.0)


def test_feature_dim_custom_sectors() -> None:
    """feature_dim follows a non-default n_sectors value."""
    cfg = LidarConfig(
        enabled=True,
        serial_port="/dev/ttyUSB1",
        baud_rate=230400,
        max_range_m=12.0,
        min_range_m=0.15,
        scan_frequency_hz=10.0,
        min_confidence=0,
        read_timeout_s=0.2,
        n_sectors=18,
        feature_dim=18,
    )
    ext = LidarFeatureExtractor(cfg)
    assert ext.feature_dim == 18


def test_output_shape() -> None:
    """Feature vector shape matches (n_sectors,)."""
    cfg = LidarConfig(
        enabled=True,
        serial_port="/dev/ttyUSB1",
        baud_rate=230400,
        max_range_m=12.0,
        min_range_m=0.15,
        scan_frequency_hz=10.0,
        min_confidence=0,
        read_timeout_s=0.2,
        n_sectors=18,
        feature_dim=18,
    )
    ext = LidarFeatureExtractor(cfg)
    n_points = 360
    angles = np.linspace(0.0, 359.0, num=n_points, dtype=np.float32)
    distances_mm = np.full(n_points, 5000.0, dtype=np.float32)
    confidences = np.full(n_points, 200, dtype=np.uint8)
    scan = LidarScan(
        angles_deg=angles,
        distances_mm=distances_mm,
        confidences=confidences,
        timestamp=0.0,
        n_points=n_points,
    )
    features = ext.extract(scan)
    assert features.shape == (18,)


def test_output_dtype(extractor: LidarFeatureExtractor) -> None:
    """Feature vector dtype is float32."""
    n_points = 360
    angles = np.linspace(0.0, 359.0, num=n_points, dtype=np.float32)
    distances_mm = np.full(n_points, 5000.0, dtype=np.float32)
    confidences = np.full(n_points, 200, dtype=np.uint8)
    scan = LidarScan(
        angles_deg=angles,
        distances_mm=distances_mm,
        confidences=confidences,
        timestamp=0.0,
        n_points=n_points,
    )
    features = extractor.extract(scan)
    assert features.dtype == np.float32
