from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.config.schema import SafetyConfig
from mousedroid.safety.context import SafetyContext
from mousedroid.safety.monitor import MouseDroidSafetyMonitor
from mousedroid.sensing.bundle import MouseDroidObservationBundle


def _make_observation(
    distance_m: float = 2.0,
    battery_v: float = 12.0,
    valid_sensors: int = 3,
) -> MouseDroidObservationBundle:
    motor = np.array([0.0, 0.0, 0.0, battery_v], dtype=np.float32)
    mask = np.zeros(4, dtype=np.float32)
    mask[:valid_sensors] = 1.0
    return MouseDroidObservationBundle(
        _distance_m=distance_m,
        _motor_state=motor,
        _valid_mask=mask,
    )


@given(
    distance=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    battery=st.floats(min_value=0.0, max_value=20.0, allow_nan=False),
    loop_time=st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
)
@settings(max_examples=30)
def test_evaluate_always_returns_safety_context(
    distance: float,
    battery: float,
    loop_time: float,
) -> None:
    monitor = MouseDroidSafetyMonitor(SafetyConfig())
    obs = _make_observation(distance_m=distance, battery_v=battery)
    ctx = monitor.evaluate(obs, loop_time)
    assert isinstance(ctx, SafetyContext)


@given(
    distance=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    battery=st.floats(min_value=0.0, max_value=20.0, allow_nan=False),
    loop_time=st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
)
@settings(max_examples=30)
def test_is_emergency_is_bool(
    distance: float,
    battery: float,
    loop_time: float,
) -> None:
    monitor = MouseDroidSafetyMonitor(SafetyConfig())
    obs = _make_observation(distance_m=distance, battery_v=battery)
    ctx = monitor.evaluate(obs, loop_time)
    assert isinstance(ctx.is_emergency, bool)


@given(
    distance=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    battery=st.floats(min_value=0.0, max_value=20.0, allow_nan=False),
    loop_time=st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
)
@settings(max_examples=30)
def test_valid_sensor_count_non_negative(
    distance: float,
    battery: float,
    loop_time: float,
) -> None:
    monitor = MouseDroidSafetyMonitor(SafetyConfig())
    obs = _make_observation(distance_m=distance, battery_v=battery)
    ctx = monitor.evaluate(obs, loop_time)
    assert ctx.valid_sensor_count >= 0


def test_safe_conditions_no_emergency() -> None:
    monitor = MouseDroidSafetyMonitor(SafetyConfig())
    obs = _make_observation(distance_m=2.0, battery_v=12.0, valid_sensors=3)
    ctx = monitor.evaluate(obs, loop_time_ms=10.0)
    assert ctx.is_emergency is False


def test_close_obstacle_triggers_emergency() -> None:
    monitor = MouseDroidSafetyMonitor(SafetyConfig())
    obs = _make_observation(distance_m=0.01, battery_v=12.0, valid_sensors=3)
    ctx = monitor.evaluate(obs, loop_time_ms=10.0)
    assert ctx.is_emergency is True


# ---------------------------------------------------------------------------
# S-2: LiDAR clearance never exceeds what was actually measured.
#
# The pre-fix failure path substituted ``np.ones(feature_dim)`` for a failed
# read; features are normalised range fractions, so that vector asserted
# maximum range in every sector and the monitor converted it into
# ``lidar_max_range_m`` of clearance. These properties hold the invariant that
# an absent reading can never become a favourable one.
# ---------------------------------------------------------------------------

#: Big enough that the generic ``sensor_stale_s`` sweep cannot fire inside a
#: property's tick sequence and supply an ``is_emergency`` these tests would
#: then wrongly credit to the LiDAR policy.
_NO_STALENESS: dict[str, float] = {"sensor_stale_s": 1e6}


def _make_lidar_observation(
    lidar_features: np.ndarray | None,
    *,
    timestamp: float = 0.0,
    lidar_slot_valid: bool = False,
    distance_m: float = 2.0,
) -> MouseDroidObservationBundle:
    """``distance_m`` is exposed so a property that sweeps
    ``min_forward_clearance_m`` can keep the ULTRASONIC check satisfied --
    otherwise a raised threshold trips ``is_emergency`` through the forward
    clearance path and the LiDAR assertion below would pass for the wrong
    reason.
    """
    motor = np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32)
    mask = np.array([1.0, 1.0, 1.0, 1.0, float(lidar_slot_valid)], dtype=np.float32)
    return MouseDroidObservationBundle(
        _timestamp=timestamp,
        _distance_m=distance_m,
        _motor_state=motor,
        _lidar_features=lidar_features,
        _valid_mask=mask,
    )


@given(
    policy=st.sampled_from(["degrade", "emergency"]),
    max_range=st.floats(min_value=0.1, max_value=50.0, allow_nan=False),
    min_clearance=st.floats(min_value=0.01, max_value=5.0, allow_nan=False),
    timestamp=st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
)
@settings(max_examples=50)
def test_absent_lidar_never_reports_clearance_under_fail_closed(
    policy: str,
    max_range: float,
    min_clearance: float,
    timestamp: float,
) -> None:
    """No tuning of the range or clearance thresholds can make absence look safe."""
    monitor = MouseDroidSafetyMonitor(
        SafetyConfig(
            lidar_unavailable_policy=policy,
            lidar_max_range_m=max_range,
            min_forward_clearance_m=min_clearance,
            **_NO_STALENESS,
        )
    )
    ctx = monitor.evaluate(
        _make_lidar_observation(
            None,
            timestamp=timestamp,
            distance_m=min_clearance + 10.0,
        ),
        10.0,
    )
    assert ctx.forward_clearance_ok is True, "ultrasonic must not confound the check"
    assert ctx.lidar_clearance_ok is False
    assert ctx.lidar_min_dist_m < min_clearance
    assert ctx.is_emergency is (policy == "emergency")


@given(
    features=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=1,
        max_size=64,
    ),
    max_range=st.floats(min_value=0.1, max_value=50.0, allow_nan=False),
)
@settings(max_examples=50)
def test_present_lidar_features_alone_determine_reported_distance(
    features: list[float],
    max_range: float,
) -> None:
    """With real features the reported distance is exactly the conversion.

    Pins the arithmetic the fail-closed branch sits beside, so a future change
    to the absent-data path cannot quietly alter the measured path too.
    """
    monitor = MouseDroidSafetyMonitor(
        SafetyConfig(
            lidar_unavailable_policy="emergency", lidar_max_range_m=max_range, **_NO_STALENESS
        )
    )
    feats = np.array(features, dtype=np.float32)
    ctx = monitor.evaluate(
        _make_lidar_observation(feats, lidar_slot_valid=True),
        10.0,
    )
    assert ctx.lidar_min_dist_m == pytest.approx(float(np.min(feats)) * max_range, rel=1e-6)


@given(
    timestamp=st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
    max_range=st.floats(min_value=0.1, max_value=50.0, allow_nan=False),
)
@settings(max_examples=50)
def test_ignore_policy_reproduces_the_pre_s2_read_through(
    timestamp: float,
    max_range: float,
) -> None:
    """The default must stay inert for every input, or YAML deploys change."""
    monitor = MouseDroidSafetyMonitor(SafetyConfig(lidar_max_range_m=max_range, **_NO_STALENESS))
    ctx = monitor.evaluate(_make_lidar_observation(None, timestamp=timestamp), 10.0)
    assert ctx.lidar_min_dist_m == float("inf")
    assert ctx.lidar_clearance_ok is True
    assert ctx.is_emergency is False


@given(
    grace=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    elapsed=st.floats(min_value=0.0, max_value=20.0, allow_nan=False),
)
@settings(max_examples=50)
def test_grace_window_boundary_is_exclusive(grace: float, elapsed: float) -> None:
    """The policy fires exactly when ``elapsed >= grace``, never earlier or later."""
    monitor = MouseDroidSafetyMonitor(
        SafetyConfig(
            lidar_unavailable_policy="emergency",
            lidar_unavailable_grace_s=grace,
            **_NO_STALENESS,
        )
    )
    # First LiDAR-less tick seeds the clock at t=0.
    monitor.evaluate(_make_lidar_observation(None, timestamp=0.0), 10.0)
    ctx = monitor.evaluate(_make_lidar_observation(None, timestamp=elapsed), 10.0)
    assert ctx.is_emergency is (elapsed >= grace)
