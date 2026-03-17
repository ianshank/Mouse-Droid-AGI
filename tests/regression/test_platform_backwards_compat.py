"""Regression tests — verify backwards compatibility across platforms.

Ensures existing ground-robot (mouse_droid) configuration, factory
output, and safety behaviour are unaffected by drone support additions.
"""

from __future__ import annotations


import numpy as np

from mousedroid.comms.ground_adapter import GroundMotorAdapter
from mousedroid.comms.motor_protocol import MotorControlProtocol
from mousedroid.config.schema import PlatformType, Settings
from mousedroid.factory import (
    build_motor_controller,
    build_orchestrator,
    build_safety_monitor,
)
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext
from mousedroid.safety.monitor import MouseDroidSafetyMonitor
from mousedroid.sensing.bundle import MouseDroidObservationBundle


class TestSettingsBackwardsCompat:
    def test_default_platform_is_mouse_droid(self):
        """Settings(mock_hardware=True) defaults to mouse_droid platform."""
        cfg = Settings(mock_hardware=True)
        assert cfg.platform == PlatformType.MOUSE_DROID

    def test_drone_platform_valid(self):
        """Settings(mock_hardware=True, platform='drone') is valid."""
        cfg = Settings(mock_hardware=True, platform="drone")
        assert cfg.platform == PlatformType.DRONE

    def test_ground_config_has_no_drone_fields(self):
        """Ground config has None for all drone-specific fields."""
        cfg = Settings(mock_hardware=True)
        assert cfg.drone is None
        assert cfg.flight_controller is None
        assert cfg.flight_envelope is None
        assert cfg.geofence is None

    def test_drone_config_no_ultrasonic_required(self):
        """Drone platform does not require ultrasonic config."""
        cfg = Settings(mock_hardware=False, platform="drone")
        assert cfg.ultrasonic is None  # Not required for drone


class TestSafetyContextBackwardsCompat:
    def test_default_safety_context_safe(self):
        """SafetyContext() with no args has safe defaults for all drone fields."""
        ctx = SafetyContext()
        assert ctx.altitude_ok is True
        assert ctx.geofence_ok is True
        assert ctx.gps_fix is True
        assert ctx.imu_healthy is True
        assert ctx.armed is False
        assert ctx.is_emergency is False

    def test_ground_safety_context_unchanged(self):
        """Ground-specific fields still work as before."""
        ctx = SafetyContext(
            ultrasonic_dist_m=1.0,
            forward_clearance_ok=True,
            battery_voltage=12.0,
            is_emergency=False,
        )
        assert ctx.ultrasonic_dist_m == 1.0
        assert ctx.forward_clearance_ok is True
        assert ctx.battery_voltage == 12.0


class TestFactoryBackwardsCompat:
    def test_build_motor_controller_ground(self):
        """build_motor_controller for ground returns GroundMotorAdapter."""
        cfg = Settings(mock_hardware=True)
        motor = build_motor_controller(cfg)
        assert isinstance(motor, GroundMotorAdapter)
        assert isinstance(motor, MotorControlProtocol)
        assert motor.platform_type == "mouse_droid"

    def test_build_motor_controller_drone(self):
        """build_motor_controller for drone returns DroneMotorAdapter."""
        from mousedroid.comms.drone_adapter import DroneMotorAdapter

        cfg = Settings(mock_hardware=True, platform="drone")
        motor = build_motor_controller(cfg)
        assert isinstance(motor, DroneMotorAdapter)
        assert isinstance(motor, MotorControlProtocol)
        assert motor.platform_type == "drone"

    def test_build_safety_monitor_ground(self):
        """build_safety_monitor for ground returns MouseDroidSafetyMonitor."""
        cfg = Settings(mock_hardware=True)
        monitor = build_safety_monitor(cfg)
        assert isinstance(monitor, MouseDroidSafetyMonitor)

    def test_build_safety_monitor_drone(self):
        """build_safety_monitor for drone returns DroneSafetyMonitor."""
        from mousedroid.safety.drone_monitor import DroneSafetyMonitor

        cfg = Settings(mock_hardware=True, platform="drone")
        monitor = build_safety_monitor(cfg)
        assert isinstance(monitor, DroneSafetyMonitor)

    def test_build_orchestrator_ground_produces_working_orch(self):
        """build_orchestrator with ground config produces a functioning orchestrator."""
        cfg = Settings(mock_hardware=True)
        orch = build_orchestrator(cfg)
        assert isinstance(orch, MouseDroidOrchestrator)

    def test_build_orchestrator_drone_produces_working_orch(self):
        """build_orchestrator with drone config produces a functioning orchestrator."""
        cfg = Settings(mock_hardware=True, platform="drone")
        orch = build_orchestrator(cfg)
        assert isinstance(orch, MouseDroidOrchestrator)


class TestObservationBundleBackwardsCompat:
    def test_ground_bundle_unchanged(self):
        """MouseDroidObservationBundle still works exactly as before."""
        bundle = MouseDroidObservationBundle(
            _timestamp=1.0,
            _distance_m=1.5,
            _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
            _valid_mask=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        )
        assert bundle.distance_m == 1.5
        assert bundle.n_modalities == 4
        assert bundle.motor_state.shape == (4,)
