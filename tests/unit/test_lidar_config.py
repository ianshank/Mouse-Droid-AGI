"""Tests for LidarConfig Pydantic model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mousedroid.config.schema import LidarConfig


def test_defaults() -> None:
    """LidarConfig has sane defaults."""
    cfg = LidarConfig()
    assert cfg.enabled is True
    assert cfg.serial_port == "/dev/ttyUSB1"
    assert cfg.baud_rate == 230400
    assert cfg.max_range_m == 12.0
    assert cfg.min_range_m == 0.15
    assert cfg.scan_frequency_hz == 10.0
    assert cfg.min_confidence == 0
    assert cfg.read_timeout_s == 0.2
    assert cfg.n_sectors == 36
    assert cfg.feature_dim == 36


def test_range_ordering_valid() -> None:
    """Valid range ordering passes validation."""
    cfg = LidarConfig(max_range_m=10.0, min_range_m=0.5)
    assert cfg.max_range_m > cfg.min_range_m


def test_range_ordering_invalid_raises() -> None:
    """max_range_m <= min_range_m raises ValidationError."""
    with pytest.raises(ValidationError, match="max_range_m must be > min_range_m"):
        LidarConfig(max_range_m=0.1, min_range_m=0.5)


def test_range_ordering_equal_raises() -> None:
    """max_range_m == min_range_m raises ValidationError."""
    with pytest.raises(ValidationError, match="max_range_m must be > min_range_m"):
        LidarConfig(max_range_m=1.0, min_range_m=1.0)


def test_baud_rate_positive() -> None:
    """baud_rate must be > 0."""
    with pytest.raises(ValidationError):
        LidarConfig(baud_rate=0)


def test_n_sectors_positive() -> None:
    """n_sectors must be > 0."""
    with pytest.raises(ValidationError):
        LidarConfig(n_sectors=0)


def test_baud_rate_negative_raises() -> None:
    """Negative baud_rate raises ValidationError."""
    with pytest.raises(ValidationError):
        LidarConfig(baud_rate=-9600)


def test_min_range_must_be_positive() -> None:
    """min_range_m <= 0 raises ValidationError."""
    with pytest.raises(ValidationError):
        LidarConfig(min_range_m=0.0)


def test_max_range_must_be_positive() -> None:
    """max_range_m <= 0 raises ValidationError."""
    with pytest.raises(ValidationError):
        LidarConfig(max_range_m=0.0, min_range_m=0.0)


def test_scan_frequency_must_be_positive() -> None:
    """scan_frequency_hz <= 0 raises ValidationError."""
    with pytest.raises(ValidationError):
        LidarConfig(scan_frequency_hz=0.0)


def test_min_confidence_lower_bound() -> None:
    """min_confidence < 0 raises ValidationError."""
    with pytest.raises(ValidationError):
        LidarConfig(min_confidence=-1)


def test_min_confidence_upper_bound() -> None:
    """min_confidence > 255 raises ValidationError."""
    with pytest.raises(ValidationError):
        LidarConfig(min_confidence=256)


def test_custom_values() -> None:
    """Custom values are accepted."""
    cfg = LidarConfig(
        enabled=False,
        serial_port="/dev/ttyAMA0",
        baud_rate=115200,
        max_range_m=8.0,
        min_range_m=0.3,
        scan_frequency_hz=5.0,
        min_confidence=50,
        read_timeout_s=0.5,
        n_sectors=72,
        feature_dim=72,
    )
    assert cfg.enabled is False
    assert cfg.serial_port == "/dev/ttyAMA0"
    assert cfg.baud_rate == 115200
    assert cfg.max_range_m == 8.0
    assert cfg.min_range_m == 0.3
    assert cfg.n_sectors == 72
