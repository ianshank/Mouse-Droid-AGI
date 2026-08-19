"""Regression tests pinning backwards compatibility and defaults for Autonomous Architecture."""

from __future__ import annotations

from mousedroid.config.schema.hardware import MotorControllerConfig
from mousedroid.config.schema.root import Settings
from mousedroid.interfaces.protocols import GoalVector


def test_motor_controller_config_defaults() -> None:
    """Regression test: MotorControllerConfig defaults preserve expected baseline values."""
    cfg = MotorControllerConfig()
    assert cfg.enabled is True
    assert cfg.serial_port == "/dev/ttyUSB0"
    assert cfg.baudrate == 115200
    assert cfg.limits.max_linear_velocity == 1.0
    assert cfg.limits.max_angular_velocity == 1.5
    assert cfg.limits.watchdog_timeout_s == 0.5


def test_settings_motor_field_backwards_compat() -> None:
    """Regression test: Settings loads cleanly with motor field populated by default."""
    cfg = Settings()
    assert hasattr(cfg, "motor")
    assert isinstance(cfg.motor, MotorControllerConfig)
    assert cfg.motor.enabled is True
    assert cfg.motor.limits.max_linear_velocity == 1.0


def test_goal_vector_defaults() -> None:
    """Regression test: GoalVector defaults are zero velocity and safe."""
    gv = GoalVector()
    assert gv.linear_velocity == 0.0
    assert gv.angular_velocity == 0.0
    assert gv.arm_action == "idle"
    assert gv.confidence == 1.0
    assert gv.is_safe is True
