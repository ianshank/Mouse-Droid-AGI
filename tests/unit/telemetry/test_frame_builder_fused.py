"""Unit tests for the sensor-fusion summary on telemetry frames.

Exercises ``_build_fused_summary`` directly for both ``valid_mask`` lengths
(4 without lidar, 5 with) and confirms ``build_telemetry_frame`` attaches a
populated ``fused`` dict. The 4-element branch must never IndexError.
"""

from __future__ import annotations

import numpy as np
import pytest

from mousedroid.safety.context import SafetyContext
from mousedroid.sensing.bundle import MouseDroidObservationBundle
from mousedroid.telemetry.frame_builder import _build_fused_summary, build_telemetry_frame


# --------------------------------------------------------------------------- #
# _build_fused_summary — both mask lengths
# --------------------------------------------------------------------------- #
def test_fused_summary_no_lidar_is_length_four() -> None:
    fused = _build_fused_summary([1.0, 1.0, 1.0, 0.0], vision_norm=3.0, audio_rms=4.0)
    assert fused["n_modalities"] == 4
    assert fused["lidar_present"] is False
    assert fused["n_valid"] == 3
    assert fused["modalities"] == {
        "vision": True,
        "ultrasonic": True,
        "motor": True,
        "audio": False,
        "lidar": False,  # surfaced as False even though the slot is absent
    }
    assert fused["fused_norm"] == pytest.approx(5.0)  # sqrt(3^2 + 4^2)


def test_fused_summary_with_lidar_is_length_five() -> None:
    fused = _build_fused_summary([1.0, 0.0, 1.0, 1.0, 1.0], vision_norm=0.0, audio_rms=0.0)
    assert fused["n_modalities"] == 5
    assert fused["lidar_present"] is True
    assert fused["n_valid"] == 4
    assert fused["modalities"]["lidar"] is True
    assert fused["modalities"]["ultrasonic"] is False
    assert fused["fused_norm"] == pytest.approx(0.0)


def test_fused_summary_all_invalid() -> None:
    fused = _build_fused_summary([0.0, 0.0, 0.0, 0.0, 0.0], vision_norm=0.0, audio_rms=0.0)
    assert fused["n_valid"] == 0
    assert all(v is False for v in fused["modalities"].values())


def test_fused_summary_four_mask_never_indexerrors() -> None:
    # Regression guard for the peer-review footgun: a 4-element mask must not
    # raise when the builder maps the fixed 5-name modality tuple.
    fused = _build_fused_summary([1.0, 1.0, 1.0, 1.0], vision_norm=1.0, audio_rms=0.0)
    assert "lidar" in fused["modalities"]


# --------------------------------------------------------------------------- #
# build_telemetry_frame attaches a populated fused summary
# --------------------------------------------------------------------------- #
def _bundle(valid_mask: list[float], *, lidar: bool) -> MouseDroidObservationBundle:
    kwargs: dict[str, object] = {
        "_timestamp": 0.0,
        "_vision_features": np.ones(8, dtype=np.float32),
        "_distance_m": 1.0,
        "_motor_state": np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        "_audio_chunk": np.zeros(16, dtype=np.float32),
        "_valid_mask": np.array(valid_mask, dtype=np.float32),
    }
    if lidar:
        kwargs["_lidar_features"] = np.ones(36, dtype=np.float32)
        kwargs["_lidar_n_points"] = 200
    return MouseDroidObservationBundle(**kwargs)  # type: ignore[arg-type]


def test_build_telemetry_frame_attaches_fused_no_lidar() -> None:
    frame = build_telemetry_frame(
        _bundle([1.0, 1.0, 1.0, 1.0], lidar=False),
        SafetyContext(is_emergency=False),
        loop_time_ms=5.0,
        tick_count=1,
    )
    assert frame.fused["n_modalities"] == 4
    assert frame.fused["lidar_present"] is False
    assert "fused" in frame.to_dict()


def test_build_telemetry_frame_attaches_fused_with_lidar() -> None:
    frame = build_telemetry_frame(
        _bundle([1.0, 1.0, 1.0, 1.0, 1.0], lidar=True),
        SafetyContext(is_emergency=False),
        loop_time_ms=5.0,
        tick_count=1,
    )
    assert frame.fused["n_modalities"] == 5
    assert frame.fused["lidar_present"] is True
    assert frame.fused["modalities"]["lidar"] is True
