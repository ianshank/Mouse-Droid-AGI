"""Tests for :class:`SensorLivenessTracker`."""

from __future__ import annotations

import pytest

from mousedroid.telemetry.sensor_liveness import (
    LIVENESS_STATES,
    SensorLiveness,
    SensorLivenessTracker,
)


def test_state_constants_match_dataclass_literals() -> None:
    """Public ordering covers every literal expressed in the dataclass."""
    assert set(LIVENESS_STATES) == {"disabled", "awaiting", "live", "stale"}


def test_init_rejects_non_positive_stale_threshold() -> None:
    with pytest.raises(ValueError, match="stale_s must be > 0"):
        SensorLivenessTracker(stale_s=0.0)
    with pytest.raises(ValueError, match="stale_s must be > 0"):
        SensorLivenessTracker(stale_s=-1.0)


class TestStateTransitions:
    """States flow correctly through the disabled→awaiting→live→stale path."""

    def test_disabled_state(self) -> None:
        tracker = SensorLivenessTracker(stale_s=2.0)
        tracker.register("lidar", enabled=False)
        snap = tracker.snapshot(now_s=10.0)
        assert snap["lidar"].state == "disabled"
        assert snap["lidar"].age_s is None

    def test_awaiting_state_when_enabled_no_observation(self) -> None:
        tracker = SensorLivenessTracker(stale_s=2.0)
        tracker.register("lidar", enabled=True)
        snap = tracker.snapshot(now_s=10.0)
        assert snap["lidar"].state == "awaiting"
        assert snap["lidar"].age_s is None

    def test_live_state_within_threshold(self) -> None:
        tracker = SensorLivenessTracker(stale_s=2.0)
        tracker.register("lidar", enabled=True)
        tracker.mark_observed("lidar", now_s=9.5)
        snap = tracker.snapshot(now_s=10.0)
        assert snap["lidar"].state == "live"
        assert snap["lidar"].age_s == pytest.approx(0.5, abs=1e-6)

    def test_stale_state_above_threshold(self) -> None:
        tracker = SensorLivenessTracker(stale_s=2.0)
        tracker.register("lidar", enabled=True)
        tracker.mark_observed("lidar", now_s=5.0)
        snap = tracker.snapshot(now_s=10.0)
        assert snap["lidar"].state == "stale"
        assert snap["lidar"].age_s == pytest.approx(5.0, abs=1e-6)

    def test_reregister_preserves_last_timestamp(self) -> None:
        """Reconfiguring enabled flag must not lose the prior observation."""
        tracker = SensorLivenessTracker(stale_s=2.0)
        tracker.register("lidar", enabled=True)
        tracker.mark_observed("lidar", now_s=9.5)
        tracker.register("lidar", enabled=True)
        snap = tracker.snapshot(now_s=10.0)
        assert snap["lidar"].state == "live"

    def test_mark_observed_auto_registers(self) -> None:
        tracker = SensorLivenessTracker(stale_s=2.0)
        tracker.mark_observed("audio", now_s=10.0)
        snap = tracker.snapshot(now_s=10.0)
        assert snap["audio"].state == "live"


def test_to_dict_roundtrip() -> None:
    liveness = SensorLiveness(state="live", age_s=0.42)
    assert liveness.to_dict() == {"state": "live", "age_s": 0.42}


class TestClockSkewSafety:
    """Defensive clamps for clock skew / replay scenarios."""

    def test_negative_age_floored_to_zero(self) -> None:
        """When ``now_s < last_ts`` (clock skew), age must not go negative."""
        tracker = SensorLivenessTracker(stale_s=2.0)
        tracker.register("lidar", enabled=True)
        tracker.mark_observed("lidar", now_s=10.0)
        snap = tracker.snapshot(now_s=5.0)  # clock went backwards
        assert snap["lidar"].state == "live"
        assert snap["lidar"].age_s == 0.0

    def test_enable_toggle_preserves_observation(self) -> None:
        """Re-registering as disabled then enabled keeps the cached timestamp."""
        tracker = SensorLivenessTracker(stale_s=10.0)
        tracker.register("lidar", enabled=True)
        tracker.mark_observed("lidar", now_s=1.0)
        tracker.register("lidar", enabled=False)
        snap = tracker.snapshot(now_s=2.0)
        assert snap["lidar"].state == "disabled"
        # Re-enable: the original observation timestamp should still be live.
        tracker.register("lidar", enabled=True)
        snap = tracker.snapshot(now_s=2.0)
        assert snap["lidar"].state == "live"
