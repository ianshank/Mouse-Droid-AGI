from __future__ import annotations

import numpy as np

from mousedroid.config.schema import SafetyConfig
from mousedroid.safety.context import SafetyContext
from mousedroid.safety.monitor import MouseDroidSafetyMonitor
from mousedroid.sensing.bundle import MouseDroidObservationBundle


def _make_monitor(**overrides) -> MouseDroidSafetyMonitor:
    return MouseDroidSafetyMonitor(SafetyConfig(**overrides))


def _make_obs(
    distance_m: float = 2.0,
    battery_v: float = 12.0,
    valid_mask: list[float] | None = None,
) -> MouseDroidObservationBundle:
    motor = np.array([0.0, 0.0, 0.0, battery_v], dtype=np.float32)
    mask = np.array(valid_mask or [1.0, 1.0, 1.0], dtype=np.float32)
    return MouseDroidObservationBundle(
        _distance_m=distance_m,
        _motor_state=motor,
        _valid_mask=mask,
    )


def test_construct():
    m = _make_monitor()
    assert m is not None


def test_evaluate_normal_returns_safe():
    m = _make_monitor()
    obs = _make_obs(distance_m=2.0, battery_v=12.0)
    ctx = m.evaluate(obs, loop_time_ms=10.0)
    assert isinstance(ctx, SafetyContext)
    assert ctx.is_emergency is False
    assert ctx.forward_clearance_ok is True


def test_evaluate_close_obstacle():
    m = _make_monitor(min_forward_clearance_m=0.20)
    obs = _make_obs(distance_m=0.10)
    ctx = m.evaluate(obs, loop_time_ms=10.0)
    assert ctx.forward_clearance_ok is False
    assert ctx.is_emergency is True


def test_evaluate_obstacle_at_threshold():
    m = _make_monitor(min_forward_clearance_m=0.20)
    obs = _make_obs(distance_m=0.20)
    ctx = m.evaluate(obs, loop_time_ms=10.0)
    assert ctx.forward_clearance_ok is True
    assert ctx.is_emergency is False


def test_evaluate_low_battery_warning():
    m = _make_monitor(battery_warn_v=10.5, battery_critical_v=9.5)
    obs = _make_obs(battery_v=10.0)
    ctx = m.evaluate(obs, loop_time_ms=10.0)
    # Low but not critical => not emergency from battery alone
    assert ctx.battery_voltage == 10.0
    # is_emergency depends on whether other checks pass
    assert ctx.forward_clearance_ok is True


def test_evaluate_critical_battery():
    m = _make_monitor(battery_warn_v=10.5, battery_critical_v=9.5)
    obs = _make_obs(battery_v=9.0)
    ctx = m.evaluate(obs, loop_time_ms=10.0)
    assert ctx.is_emergency is True


def test_evaluate_too_few_valid_sensors():
    m = _make_monitor(min_valid_sensors=2)
    obs = _make_obs(valid_mask=[1.0, 0.0, 0.0])
    ctx = m.evaluate(obs, loop_time_ms=10.0)
    assert ctx.valid_sensor_count == 1
    assert ctx.is_emergency is True


def test_evaluate_enough_valid_sensors():
    m = _make_monitor(min_valid_sensors=2)
    obs = _make_obs(valid_mask=[1.0, 1.0, 0.0])
    ctx = m.evaluate(obs, loop_time_ms=10.0)
    assert ctx.valid_sensor_count == 2
    assert ctx.is_emergency is False


def test_evaluate_high_loop_time():
    m = _make_monitor()
    obs = _make_obs()
    ctx = m.evaluate(obs, loop_time_ms=300.0)
    assert ctx.is_emergency is True


def test_evaluate_normal_loop_time():
    m = _make_monitor()
    obs = _make_obs()
    ctx = m.evaluate(obs, loop_time_ms=50.0)
    assert ctx.is_emergency is False


def test_evaluate_all_fields_populated():
    m = _make_monitor()
    obs = _make_obs(distance_m=1.5, battery_v=11.5)
    ctx = m.evaluate(obs, loop_time_ms=25.0)
    assert ctx.ultrasonic_dist_m == 1.5
    assert ctx.battery_voltage == 11.5
    assert ctx.loop_time_ms == 25.0
    assert ctx.valid_sensor_count == 3
    assert ctx.forward_clearance_ok is True


def test_evaluate_returns_frozen_context():
    m = _make_monitor()
    obs = _make_obs()
    ctx = m.evaluate(obs, loop_time_ms=10.0)
    import dataclasses

    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        ctx.is_emergency = True  # type: ignore[misc]


def test_evaluate_boundary_loop_time():
    m = _make_monitor()
    obs = _make_obs()
    ctx = m.evaluate(obs, loop_time_ms=200.0)
    # 200.0 is NOT > 200.0, so no emergency from loop time
    assert ctx.is_emergency is False


def test_evaluate_sensor_staleness_no_crash():
    m = _make_monitor(sensor_stale_s=0.5)
    obs = _make_obs()
    ctx = m.evaluate(obs, loop_time_ms=10.0)
    assert isinstance(ctx, SafetyContext)


def test_evaluate_multiple_emergency_conditions():
    m = _make_monitor(min_forward_clearance_m=0.20, battery_critical_v=9.5, min_valid_sensors=2)
    obs = _make_obs(distance_m=0.05, battery_v=8.0, valid_mask=[1.0, 0.0, 0.0])
    ctx = m.evaluate(obs, loop_time_ms=500.0)
    assert ctx.is_emergency is True
    assert ctx.forward_clearance_ok is False
