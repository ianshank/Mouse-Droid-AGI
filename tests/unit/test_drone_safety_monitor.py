"""Tests for DroneSafetyMonitor — altitude, geofence, GPS, and IMU checks."""

from __future__ import annotations

import numpy as np

from mousedroid.config.schema import FlightEnvelopeConfig, GeofenceConfig, SafetyConfig
from mousedroid.safety.drone_monitor import DroneSafetyMonitor
from mousedroid.sensing.drone_bundle import DroneObservationBundle


def _make_monitor(
    *,
    envelope: FlightEnvelopeConfig | None = None,
    geofence: GeofenceConfig | None = None,
) -> DroneSafetyMonitor:
    """Create a DroneSafetyMonitor with test-friendly defaults."""
    safety_cfg = SafetyConfig(
        min_forward_clearance_m=0.5,
        battery_warn_v=14.4,
        battery_critical_v=13.2,
    )
    return DroneSafetyMonitor(
        safety_cfg=safety_cfg,
        envelope_cfg=envelope or FlightEnvelopeConfig(),
        geofence_cfg=geofence,
    )


def _make_obs(
    *,
    altitude_m: float = 10.0,
    distance_m: float = 5.0,
    battery_v: float = 16.0,
    gps_fix: bool = True,
    imu_healthy: bool = True,
    gps_position: tuple[float, float, float] = (0.0, 0.0, 10.0),
) -> DroneObservationBundle:
    """Create a drone observation bundle for testing."""
    return DroneObservationBundle(
        _timestamp=1.0,
        _distance_m=distance_m,
        _motor_state=np.array([0.0, 0.0, 0.0, 0.0, altitude_m, battery_v, 1.0], dtype=np.float32),
        _valid_mask=np.ones(7, dtype=np.float32),
        _altitude_m=altitude_m,
        _gps_position=gps_position,
        _gps_fix=gps_fix,
        _imu_healthy=imu_healthy,
        _armed=True,
    )


class TestAltitudeChecks:
    def test_normal_altitude_no_emergency(self):
        monitor = _make_monitor()
        obs = _make_obs(altitude_m=10.0)
        ctx = monitor.evaluate(obs, loop_time_ms=10.0)
        assert ctx.altitude_ok is True
        assert ctx.is_emergency is False

    def test_altitude_below_minimum_triggers_emergency(self):
        monitor = _make_monitor(envelope=FlightEnvelopeConfig(min_altitude_m=2.0))
        obs = _make_obs(altitude_m=0.5)
        ctx = monitor.evaluate(obs, loop_time_ms=10.0)
        assert ctx.altitude_ok is False
        assert ctx.is_emergency is True

    def test_altitude_above_maximum_triggers_emergency(self):
        monitor = _make_monitor(envelope=FlightEnvelopeConfig(max_altitude_m=50.0))
        obs = _make_obs(altitude_m=60.0)
        ctx = monitor.evaluate(obs, loop_time_ms=10.0)
        assert ctx.altitude_ok is False
        assert ctx.is_emergency is True


class TestGeofenceChecks:
    def test_no_geofence_config_ok(self):
        monitor = _make_monitor(geofence=None)
        obs = _make_obs()
        ctx = monitor.evaluate(obs, loop_time_ms=10.0)
        assert ctx.geofence_ok is True

    def test_within_geofence_ok(self):
        monitor = _make_monitor(geofence=GeofenceConfig(radius_m=1000.0))
        obs = _make_obs(gps_position=(0.0001, 0.0001, 10.0))
        ctx = monitor.evaluate(obs, loop_time_ms=10.0)
        assert ctx.geofence_ok is True

    def test_altitude_ceiling_breach(self):
        monitor = _make_monitor(geofence=GeofenceConfig(max_altitude_m=50.0))
        obs = _make_obs(altitude_m=60.0)
        ctx = monitor.evaluate(obs, loop_time_ms=10.0)
        assert ctx.geofence_ok is False
        assert ctx.is_emergency is True

    def test_radius_breach(self):
        monitor = _make_monitor(
            geofence=GeofenceConfig(radius_m=10.0, center_lat=0.0, center_lon=0.0)
        )
        # ~111km away — well outside 10m radius
        obs = _make_obs(gps_position=(1.0, 1.0, 10.0), altitude_m=5.0)
        ctx = monitor.evaluate(obs, loop_time_ms=10.0)
        assert ctx.geofence_ok is False
        assert ctx.is_emergency is True


    def test_geofence_disabled_returns_ok(self):
        """Geofence with enabled=False should always return OK."""
        monitor = _make_monitor(geofence=GeofenceConfig(enabled=False, radius_m=1.0))
        obs = _make_obs(gps_position=(90.0, 180.0, 999.0), altitude_m=999.0)
        ctx = monitor.evaluate(obs, loop_time_ms=10.0)
        assert ctx.geofence_ok is True


class TestGPSFixCheck:
    def test_gps_fix_ok(self):
        monitor = _make_monitor()
        obs = _make_obs(gps_fix=True)
        ctx = monitor.evaluate(obs, loop_time_ms=10.0)
        assert ctx.gps_fix is True

    def test_gps_fix_lost_triggers_emergency(self):
        monitor = _make_monitor()
        obs = _make_obs(gps_fix=False)
        ctx = monitor.evaluate(obs, loop_time_ms=10.0)
        assert ctx.gps_fix is False
        assert ctx.is_emergency is True


class TestIMUHealthCheck:
    def test_imu_healthy(self):
        monitor = _make_monitor()
        obs = _make_obs(imu_healthy=True)
        ctx = monitor.evaluate(obs, loop_time_ms=10.0)
        assert ctx.imu_healthy is True

    def test_imu_unhealthy_triggers_emergency(self):
        monitor = _make_monitor()
        obs = _make_obs(imu_healthy=False)
        ctx = monitor.evaluate(obs, loop_time_ms=10.0)
        assert ctx.imu_healthy is False
        assert ctx.is_emergency is True


class TestGroundChecksComposition:
    def test_battery_critical_still_fires(self):
        """Ground safety checks (battery) still trigger through composition."""
        monitor = _make_monitor()
        obs = _make_obs(battery_v=10.0)  # Below critical
        ctx = monitor.evaluate(obs, loop_time_ms=10.0)
        assert ctx.is_emergency is True

    def test_battery_warning_not_critical(self):
        """Battery between warn and critical should log warning but not emergency."""
        monitor = _make_monitor()
        # battery_warn_v=14.4, battery_critical_v=13.2
        obs = _make_obs(battery_v=13.8)  # Between warn and critical
        ctx = monitor.evaluate(obs, loop_time_ms=10.0)
        # Battery warning doesn't trigger emergency by itself
        assert ctx.is_emergency is False

    def test_forward_clearance_still_fires(self):
        """Ground safety checks (forward clearance) still trigger through composition."""
        monitor = _make_monitor()
        obs = _make_obs(distance_m=0.1)  # Below min clearance
        ctx = monitor.evaluate(obs, loop_time_ms=10.0)
        assert ctx.is_emergency is True


class TestNormalFlight:
    def test_normal_flight_no_emergency(self):
        """Normal drone flight with all sensors OK should not trigger emergency."""
        monitor = _make_monitor()
        obs = _make_obs()
        ctx = monitor.evaluate(obs, loop_time_ms=10.0)
        assert ctx.is_emergency is False
        assert ctx.altitude_ok is True
        assert ctx.geofence_ok is True
        assert ctx.gps_fix is True
        assert ctx.imu_healthy is True

    def test_armed_state_propagated(self):
        monitor = _make_monitor()
        obs = _make_obs()
        ctx = monitor.evaluate(obs, loop_time_ms=10.0)
        assert ctx.armed is True
