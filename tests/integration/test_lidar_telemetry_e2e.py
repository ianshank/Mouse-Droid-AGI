"""End-to-end tests for LiDAR data flowing through the telemetry pipeline.

Exercises the full chain: ``MouseDroidObservationBundle.lidar_features`` +
``lidar_n_points`` -> ``build_telemetry_frame`` -> ``TelemetryFrame`` payload
-> MetricsRegistry Prometheus output.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mousedroid.config.schema import MetricsConfig
from mousedroid.sensing.bundle import MouseDroidObservationBundle
from mousedroid.telemetry.frame_builder import build_telemetry_frame
from mousedroid.telemetry.metrics import MetricsRegistry


def _bundle_with_lidar(sectors: np.ndarray, n_points: int) -> MouseDroidObservationBundle:
    return MouseDroidObservationBundle(
        _lidar_features=sectors,
        _lidar_n_points=n_points,
    )


def _safety_ctx(min_dist_m: float = float("inf")) -> SimpleNamespace:
    return SimpleNamespace(
        is_emergency=False,
        law_violations=(),
        forward_clearance_ok=True,
        lidar_clearance_ok=True,
        lidar_min_dist_m=min_dist_m,
    )


def test_frame_builder_populates_lidar_fields() -> None:
    """build_telemetry_frame copies sectors, min distance, and point count."""
    sectors = np.array([0.9, 0.2, 1.0, 1.0, 0.7, 1.0, 1.0, 0.4], dtype=np.float32)
    bundle = _bundle_with_lidar(sectors, n_points=456)

    frame = build_telemetry_frame(
        observation=bundle,
        safety_ctx=_safety_ctx(min_dist_m=2.4),
        loop_time_ms=10.0,
        tick_count=1,
    )

    assert frame.lidar_sectors is not None
    assert len(frame.lidar_sectors) == sectors.shape[0]
    assert frame.lidar_sectors == pytest.approx(sectors.tolist(), abs=1e-6)
    assert frame.lidar_min_dist_m == pytest.approx(2.4)
    assert frame.lidar_n_points == 456


def test_frame_to_dict_round_trip_preserves_lidar() -> None:
    """``TelemetryFrame.to_dict()`` serialises the new LiDAR fields for WS."""
    sectors = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    bundle = _bundle_with_lidar(sectors, n_points=100)
    frame = build_telemetry_frame(
        observation=bundle,
        safety_ctx=_safety_ctx(min_dist_m=0.5),
        loop_time_ms=5.0,
        tick_count=0,
    )
    payload = frame.to_dict()

    assert payload["lidar_sectors"] == pytest.approx([0.1, 0.5, 0.9], abs=1e-6)
    assert payload["lidar_min_dist_m"] == pytest.approx(0.5)
    assert payload["lidar_n_points"] == 100


def test_bundle_without_lidar_yields_none_sectors() -> None:
    """Backwards-compat: bundles without LiDAR features still build frames."""
    bundle = MouseDroidObservationBundle()
    frame = build_telemetry_frame(
        observation=bundle,
        safety_ctx=_safety_ctx(),  # inf => lidar_min_dist_m stays None
        loop_time_ms=5.0,
        tick_count=0,
    )
    assert frame.lidar_sectors is None
    assert frame.lidar_min_dist_m is None
    assert frame.lidar_n_points == 0


def test_metrics_registry_publishes_lidar_gauges() -> None:
    """MetricsRegistry emits the three new LiDAR metric families."""
    registry = MetricsRegistry(MetricsConfig())
    registry.set_lidar_sectors([0.1, 0.5, 1.0], max_range_m=10.0)
    registry.set_lidar_min_distance_m(1.0)
    registry.set_lidar_scan_points(321)

    text = registry.render_prometheus()
    assert 'mousedroid_lidar_sector_distance_m{sector="0"} 1' in text
    assert 'mousedroid_lidar_sector_distance_m{sector="1"} 5' in text
    assert 'mousedroid_lidar_sector_distance_m{sector="2"} 10' in text
    assert "mousedroid_lidar_min_distance_m 1" in text
    assert "mousedroid_lidar_scan_points 321" in text
