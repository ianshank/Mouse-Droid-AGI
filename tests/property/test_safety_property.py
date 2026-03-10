from __future__ import annotations

import numpy as np
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
