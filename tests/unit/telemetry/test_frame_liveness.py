"""Tests for sensor liveness propagation through ``build_telemetry_frame``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mousedroid.safety.context import SafetyContext
from mousedroid.telemetry.frame_builder import build_telemetry_frame
from mousedroid.telemetry.sensor_liveness import SensorLivenessTracker


@dataclass
class _StubObservation:
    """Minimal observation bundle satisfying frame_builder's contract."""

    timestamp: float = 0.0
    distance_m: float = 1.0
    motor_state: np.ndarray = None  # type: ignore[assignment]
    vision_features: np.ndarray = None  # type: ignore[assignment]
    audio_chunk: np.ndarray = None  # type: ignore[assignment]
    valid_mask: np.ndarray = None  # type: ignore[assignment]
    lidar_features: np.ndarray | None = None
    lidar_n_points: int = 0

    def __post_init__(self) -> None:
        if self.motor_state is None:
            self.motor_state = np.zeros(8, dtype=np.float32)
        if self.vision_features is None:
            self.vision_features = np.zeros(0, dtype=np.float32)
        if self.audio_chunk is None:
            self.audio_chunk = np.zeros(0, dtype=np.float32)
        if self.valid_mask is None:
            self.valid_mask = np.ones(4, dtype=np.float32)


def _safe_ctx() -> SafetyContext:
    return SafetyContext(is_emergency=False, lidar_min_dist_m=float("inf"))


def _build(**obs_kwargs: Any) -> Any:
    obs = _StubObservation(**obs_kwargs)
    return obs


class TestSensorLivenessInjection:
    """``sensor_liveness`` reflects what data the observation actually carries."""

    def test_no_tracker_leaves_field_empty(self) -> None:
        obs = _build()
        frame = build_telemetry_frame(obs, _safe_ctx(), 5.0, 1)
        assert frame.sensor_liveness == {}

    def test_tracker_marks_lidar_when_features_present(self) -> None:
        tracker = SensorLivenessTracker(stale_s=10.0)
        tracker.register("lidar", enabled=True)
        obs = _build(lidar_features=np.array([0.5, 0.6, 0.7], dtype=np.float32), lidar_n_points=42)
        frame = build_telemetry_frame(obs, _safe_ctx(), 5.0, 1, liveness_tracker=tracker, now_s=1.0)
        assert frame.sensor_liveness["lidar"]["state"] == "live"

    def test_tracker_reports_disabled(self) -> None:
        tracker = SensorLivenessTracker(stale_s=10.0)
        tracker.register("lidar", enabled=False)
        frame = build_telemetry_frame(
            _build(), _safe_ctx(), 5.0, 1, liveness_tracker=tracker, now_s=1.0
        )
        assert frame.sensor_liveness["lidar"]["state"] == "disabled"

    def test_tracker_reports_awaiting_then_live(self) -> None:
        tracker = SensorLivenessTracker(stale_s=10.0)
        tracker.register("lidar", enabled=True)
        # First frame: no lidar_features → awaiting.
        frame = build_telemetry_frame(
            _build(), _safe_ctx(), 5.0, 1, liveness_tracker=tracker, now_s=1.0
        )
        assert frame.sensor_liveness["lidar"]["state"] == "awaiting"
        # Second frame: lidar_features present → live.
        frame = build_telemetry_frame(
            _build(lidar_features=np.array([0.5, 0.6], dtype=np.float32)),
            _safe_ctx(),
            5.0,
            2,
            liveness_tracker=tracker,
            now_s=2.0,
        )
        assert frame.sensor_liveness["lidar"]["state"] == "live"

    def test_tracker_reports_stale_after_threshold(self) -> None:
        tracker = SensorLivenessTracker(stale_s=1.0)
        tracker.register("lidar", enabled=True)
        # Mark fresh at t=0.
        build_telemetry_frame(
            _build(lidar_features=np.array([0.5], dtype=np.float32)),
            _safe_ctx(),
            5.0,
            1,
            liveness_tracker=tracker,
            now_s=0.0,
        )
        # Re-snapshot far in the future without re-marking lidar.
        frame = build_telemetry_frame(
            _build(),
            _safe_ctx(),
            5.0,
            2,
            liveness_tracker=tracker,
            now_s=5.0,
        )
        assert frame.sensor_liveness["lidar"]["state"] == "stale"

    def test_vision_tracker_only_marks_when_features_present(self) -> None:
        tracker = SensorLivenessTracker(stale_s=10.0)
        tracker.register("vision", enabled=True)
        # Empty features stays awaiting.
        frame = build_telemetry_frame(
            _build(), _safe_ctx(), 5.0, 1, liveness_tracker=tracker, now_s=1.0
        )
        assert frame.sensor_liveness["vision"]["state"] == "awaiting"
        # Populated features → live.
        feats = np.ones(4, dtype=np.float32)
        frame = build_telemetry_frame(
            _build(vision_features=feats),
            _safe_ctx(),
            5.0,
            2,
            liveness_tracker=tracker,
            now_s=2.0,
        )
        assert frame.sensor_liveness["vision"]["state"] == "live"
