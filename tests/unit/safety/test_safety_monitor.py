from __future__ import annotations

import numpy as np

from mousedroid.config.schema import SafetyConfig
from mousedroid.safety.context import SafetyContext
from mousedroid.safety.monitor import MouseDroidSafetyMonitor
from mousedroid.sensing.bundle import MouseDroidObservationBundle

#: Loop-overrun guards that the shipped defaults now switch on: a 30-tick
#: warm-up and a 3-tick debounce (see ``SafetyConfig``). Tests in this module
#: exercise the *threshold comparison* — "does 300 ms beat a 200 ms ceiling" —
#: not the boot guards that sit in front of it, so they arm the interlock
#: immediately and let the dedicated suite in ``test_loop_overrun_debounce.py``
#: own the warm-up and debounce behaviour.
_ARMED_IMMEDIATELY: dict[str, int] = {
    "loop_overrun_warmup_ticks": 0,
    "loop_overrun_consecutive_ticks": 1,
}


def _make_monitor(**overrides) -> MouseDroidSafetyMonitor:
    return MouseDroidSafetyMonitor(SafetyConfig(**{**_ARMED_IMMEDIATELY, **overrides}))


def _make_obs(
    distance_m: float = 2.0,
    battery_v: float = 12.0,
    valid_mask: list[float] | None = None,
) -> MouseDroidObservationBundle:
    motor = np.array([0.0, 0.0, 0.0, battery_v], dtype=np.float32)
    mask = np.array(valid_mask or [1.0, 1.0, 1.0, 1.0], dtype=np.float32)
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
    assert ctx.valid_sensor_count == 4
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


# -- Sensor staleness -------------------------------------------------------


def _make_obs_at(
    timestamp: float,
    distance_m: float = 2.0,
    battery_v: float = 12.0,
    valid_mask: list[float] | None = None,
) -> MouseDroidObservationBundle:
    motor = np.array([0.0, 0.0, 0.0, battery_v], dtype=np.float32)
    mask = np.array(valid_mask or [1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    return MouseDroidObservationBundle(
        _timestamp=timestamp,
        _distance_m=distance_m,
        _motor_state=motor,
        _valid_mask=mask,
    )


def test_stale_sensor_detected():
    """Stale sensor (exceeds threshold) triggers is_emergency."""
    m = _make_monitor(sensor_stale_s=0.5)

    # First tick: sensor 0 is valid at t=100.0
    obs1 = _make_obs_at(100.0, valid_mask=[1.0, 1.0, 1.0, 1.0])
    m.evaluate(obs1, loop_time_ms=10.0)

    # Second tick: sensor 0 goes invalid at t=101.0 (1.0s > 0.5s threshold)
    obs2 = _make_obs_at(101.0, valid_mask=[0.0, 1.0, 1.0, 1.0])
    ctx = m.evaluate(obs2, loop_time_ms=10.0)
    assert ctx.is_emergency is True


def test_fresh_sensor_not_flagged_stale():
    """Sensor that just went invalid (below threshold) does not trigger emergency."""
    m = _make_monitor(sensor_stale_s=1.0)

    obs1 = _make_obs_at(100.0, valid_mask=[1.0, 1.0, 1.0, 1.0])
    m.evaluate(obs1, loop_time_ms=10.0)

    # Sensor 0 invalid but only 0.1s elapsed — not stale yet
    obs2 = _make_obs_at(100.1, valid_mask=[0.0, 1.0, 1.0, 1.0])
    ctx = m.evaluate(obs2, loop_time_ms=10.0)
    assert ctx.is_emergency is False


def test_staleness_threshold_from_config():
    """Different sensor_stale_s configs produce different emergency outcomes."""
    obs1 = _make_obs_at(100.0, valid_mask=[1.0, 1.0, 1.0, 1.0])
    obs2 = _make_obs_at(100.2, valid_mask=[0.0, 1.0, 1.0, 1.0])

    # Short threshold: 0.1s — 0.2s gap IS stale → emergency
    m_short = _make_monitor(sensor_stale_s=0.1)
    m_short.evaluate(obs1, loop_time_ms=10.0)
    ctx = m_short.evaluate(obs2, loop_time_ms=10.0)
    assert ctx.is_emergency is True

    # Long threshold: 10.0s — same gap is NOT stale → no emergency
    m_long = _make_monitor(sensor_stale_s=10.0)
    m_long.evaluate(obs1, loop_time_ms=10.0)
    ctx2 = m_long.evaluate(obs2, loop_time_ms=10.0)
    assert ctx2.is_emergency is False


def test_staleness_tracks_per_sensor():
    """Each sensor has its own staleness timestamp; stale sensor triggers emergency."""
    m = _make_monitor(sensor_stale_s=0.5)

    obs1 = _make_obs_at(100.0, valid_mask=[1.0, 1.0, 1.0, 1.0])
    m.evaluate(obs1, loop_time_ms=10.0)

    # Sensor 0 goes stale (1s > 0.5s) → emergency
    obs2 = _make_obs_at(101.0, valid_mask=[0.0, 1.0, 1.0, 1.0])
    ctx2 = m.evaluate(obs2, loop_time_ms=10.0)
    assert ctx2.is_emergency is True

    # Sensor 0 comes back valid; sensor 1 now stale → still emergency
    obs3 = _make_obs_at(102.0, valid_mask=[1.0, 0.0, 1.0, 1.0])
    ctx3 = m.evaluate(obs3, loop_time_ms=10.0)
    assert ctx3.is_emergency is True


# -- max_loop_time_ms from config ------------------------------------------


def test_max_loop_time_from_config():
    """max_loop_time_ms should come from config, not hardcoded."""
    m = _make_monitor(max_loop_time_ms=100.0)
    obs = _make_obs()
    # 150ms exceeds 100ms config threshold
    ctx = m.evaluate(obs, loop_time_ms=150.0)
    assert ctx.is_emergency is True


def test_backwards_compatible_default_loop_time():
    """Default max_loop_time_ms is 200.0 (matches old hardcoded value)."""
    m = _make_monitor()  # no max_loop_time_ms override
    obs = _make_obs()
    # 199ms should be safe (below default 200ms)
    ctx = m.evaluate(obs, loop_time_ms=199.0)
    assert ctx.is_emergency is False
    # 201ms should trigger emergency
    ctx2 = m.evaluate(obs, loop_time_ms=201.0)
    assert ctx2.is_emergency is True


# -- LiDAR clearance -------------------------------------------------------


def test_lidar_clearance_violation_triggers_emergency():
    """Observation with lidar_features where min < threshold triggers emergency."""
    m = _make_monitor(min_forward_clearance_m=0.20)
    # lidar_features are normalised distances (dist / max_range).
    # With default lidar_max_range_m=12.0, a feature value of 0.01
    # means 0.12 m which is below the 0.20 m threshold.
    lidar_feats = np.ones(36, dtype=np.float32)
    lidar_feats[10] = 0.01  # one sector very close
    obs = MouseDroidObservationBundle(
        _distance_m=2.0,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        _lidar_features=lidar_feats,
    )
    ctx = m.evaluate(obs, loop_time_ms=10.0)
    assert ctx.is_emergency is True
    assert ctx.lidar_clearance_ok is False


def test_lidar_features_none_no_emergency():
    """Observation without lidar_features does not crash or trigger emergency."""
    m = _make_monitor()
    obs = MouseDroidObservationBundle(
        _distance_m=2.0,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )
    ctx = m.evaluate(obs, loop_time_ms=10.0)
    assert ctx.is_emergency is False
    assert ctx.lidar_clearance_ok is True


# ---------------------------------------------------------------------------
# F-025 / D4 — a missing battery reading must not latch an emergency stop.
# ---------------------------------------------------------------------------


def test_missing_battery_reading_is_not_a_critical_battery():
    """0.0 V means "no telemetry", not "flat pack".

    The comms layer reports an unavailable reading as 0.0 V to keep the
    ``ESP32CommProtocol`` signature. Before the plausibility floor that value
    satisfied ``0.0 < battery_critical_v`` on EVERY tick, latching a
    permanent emergency stop from a transient comms fault — while the smoke
    runbook pointed the operator at the battery.
    """
    m = _make_monitor(battery_critical_v=9.5, battery_implausible_below_v=1.0)
    ctx = m.evaluate(_make_obs(battery_v=0.0), loop_time_ms=10.0)
    assert ctx.is_emergency is False


def test_genuinely_flat_pack_still_triggers_emergency():
    """The floor must not mask a real low-voltage condition."""
    m = _make_monitor(battery_critical_v=9.5, battery_implausible_below_v=1.0)
    ctx = m.evaluate(_make_obs(battery_v=9.0), loop_time_ms=10.0)
    assert ctx.is_emergency is True


def test_reading_just_above_the_floor_is_trusted():
    """Only sub-floor values are reclassified; 1.5 V is a real (dire) read."""
    m = _make_monitor(battery_critical_v=9.5, battery_implausible_below_v=1.0)
    ctx = m.evaluate(_make_obs(battery_v=1.5), loop_time_ms=10.0)
    assert ctx.is_emergency is True


def test_floor_of_zero_restores_pre_f025_behaviour():
    """Backwards-compatible escape hatch: 0 disables the plausibility check."""
    m = _make_monitor(battery_critical_v=9.5, battery_implausible_below_v=0.0)
    ctx = m.evaluate(_make_obs(battery_v=0.0), loop_time_ms=10.0)
    assert ctx.is_emergency is True
