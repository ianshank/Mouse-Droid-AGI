"""Tests for mousedroid.telemetry.frame_builder."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from mousedroid.telemetry.frame_builder import build_telemetry_frame
from mousedroid.telemetry.protocol import TelemetryFrame


@dataclass
class FakeSafetyContext:
    is_emergency: bool = False
    law_violations: list[str] = field(default_factory=list)
    forward_clearance_ok: bool = True
    lidar_clearance_ok: bool = True
    lidar_min_dist_m: float = float("inf")


@dataclass
class FakeObservation:
    timestamp: float = 1.0
    vision_features: np.ndarray = field(default_factory=lambda: np.ones(4, dtype=np.float32))
    distance_m: float = 2.5
    motor_state: np.ndarray = field(
        default_factory=lambda: np.array([0.1, 0.2, 0.3, 12.0], dtype=np.float32)
    )
    audio_chunk: np.ndarray = field(default_factory=lambda: np.ones(10, dtype=np.float32) * 0.5)
    lidar_features: np.ndarray | None = None
    valid_mask: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    )


class TestBuildTelemetryFrame:
    def test_basic_frame(self):
        obs = FakeObservation()
        ctx = FakeSafetyContext()
        frame = build_telemetry_frame(obs, ctx, loop_time_ms=10.5, tick_count=42)

        assert isinstance(frame, TelemetryFrame)
        assert frame.timestamp == 1.0
        assert frame.distance_m == 2.5
        assert frame.loop_time_ms == 10.5
        assert frame.tick_count == 42
        assert frame.battery_voltage == pytest.approx(12.0)
        assert frame.vision_norm == pytest.approx(2.0)  # sqrt(4 * 1^2) = 2
        assert frame.audio_rms == pytest.approx(0.5)

    def test_emergency_safety(self):
        obs = FakeObservation()
        ctx = FakeSafetyContext(is_emergency=True, law_violations=["law1_obstacle_too_close"])
        frame = build_telemetry_frame(obs, ctx, loop_time_ms=5.0, tick_count=0)

        assert frame.safety["is_emergency"] is True
        assert "law1_obstacle_too_close" in frame.safety["violations"]

    def test_small_motor_state_no_battery(self):
        obs = FakeObservation()
        obs.motor_state = np.array([0.1, 0.2], dtype=np.float32)
        ctx = FakeSafetyContext()
        frame = build_telemetry_frame(obs, ctx, loop_time_ms=1.0, tick_count=1)

        assert frame.battery_voltage == 0.0

    def test_frame_to_dict(self):
        obs = FakeObservation()
        ctx = FakeSafetyContext()
        frame = build_telemetry_frame(obs, ctx, loop_time_ms=1.0, tick_count=1)
        d = frame.to_dict()
        assert isinstance(d, dict)
        assert "timestamp" in d
        assert "battery_voltage" in d

    def test_vision_feature_downsampling_default(self):
        """Default cap of 256 samples must be applied when vision vector is larger."""
        obs = FakeObservation()
        obs.vision_features = np.ones(1024, dtype=np.float32)
        ctx = FakeSafetyContext()
        frame = build_telemetry_frame(obs, ctx, loop_time_ms=1.0, tick_count=1)
        assert frame.vision_features is not None
        assert len(frame.vision_features) <= 256

    def test_vision_feature_downsampling_custom_cap(self):
        """Caller-provided cap must override the default and bound payload size."""
        obs = FakeObservation()
        obs.vision_features = np.arange(1024, dtype=np.float32)
        ctx = FakeSafetyContext()
        frame = build_telemetry_frame(
            obs, ctx, loop_time_ms=1.0, tick_count=1, vision_feature_max_samples=8
        )
        assert frame.vision_features is not None
        assert len(frame.vision_features) <= 8

    def test_vision_feature_smaller_than_cap_preserved(self):
        """Vectors smaller than the cap must pass through without downsampling."""
        obs = FakeObservation()
        obs.vision_features = np.arange(4, dtype=np.float32)
        ctx = FakeSafetyContext()
        frame = build_telemetry_frame(
            obs, ctx, loop_time_ms=1.0, tick_count=1, vision_feature_max_samples=256
        )
        assert frame.vision_features == [0.0, 1.0, 2.0, 3.0]
