"""Regression: the dashboard fusion field is purely additive + backwards-compatible.

Guards the CLAUDE.md invariant — a new ``TelemetryFrame`` field must not break
existing constructions, serialisation, or the older ``/camera`` + ``/lidar``
pages. ``fused`` mirrors ``sensor_liveness`` (default-factory empty dict).
"""

from __future__ import annotations

import json

from mousedroid.telemetry.protocol import TelemetryFrame


def test_fused_defaults_to_empty_dict() -> None:
    """A bare frame (no observation) has an empty fused summary."""
    frame = TelemetryFrame()
    assert frame.fused == {}
    assert frame.sensor_liveness == {}  # precedent field, unchanged


def test_to_dict_is_superset_of_legacy_keys() -> None:
    """All pre-existing keys remain; ``fused`` is added, not substituted."""
    payload = TelemetryFrame().to_dict()
    legacy_keys = {
        "timestamp",
        "distance_m",
        "motor_state",
        "vision_norm",
        "audio_rms",
        "valid_mask",
        "battery_voltage",
        "safety",
        "health",
        "lidar_min_dist_m",
        "lidar_sectors",
        "lidar_n_points",
        "vision_features",
        "loop_time_ms",
        "tick_count",
        "sensor_liveness",
    }
    assert legacy_keys.issubset(payload.keys())
    assert "fused" in payload


def test_frame_is_json_serialisable_with_fused() -> None:
    frame = TelemetryFrame(fused={"n_valid": 2, "n_modalities": 5, "lidar_present": True})
    round_trip = json.loads(json.dumps(frame.to_dict()))
    assert round_trip["fused"]["n_valid"] == 2


def test_legacy_construction_without_fused_still_works() -> None:
    """Existing call sites that never set ``fused`` keep working."""
    frame = TelemetryFrame(tick_count=5, distance_m=1.0)
    assert frame.tick_count == 5
    assert frame.fused == {}
