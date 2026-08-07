from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from mousedroid.config.schema import (
    CameraConfig,
    CircuitBreakerConfig,
    ESP32Config,
    LoopConfig,
    Settings,
    UltrasonicConfig,
)

positive_float = st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False)
positive_int = st.integers(min_value=1, max_value=10000)


@given(
    serial_baud=positive_int,
    wifi_port=st.integers(min_value=1, max_value=65535),
    command_timeout_s=positive_float,
    keepalive_hz=positive_float,
    max_velocity_mps=positive_float,
    max_omega_rads=positive_float,
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_esp32config_accepts_valid_values(
    serial_baud: int,
    wifi_port: int,
    command_timeout_s: float,
    keepalive_hz: float,
    max_velocity_mps: float,
    max_omega_rads: float,
) -> None:
    cfg = ESP32Config(
        serial_baud=serial_baud,
        wifi_port=wifi_port,
        command_timeout_s=command_timeout_s,
        keepalive_hz=keepalive_hz,
        max_velocity_mps=max_velocity_mps,
        max_omega_rads=max_omega_rads,
    )
    assert cfg.serial_baud == serial_baud
    assert cfg.wifi_port == wifi_port


@given(
    resolution_width=positive_int,
    resolution_height=positive_int,
    fps=st.integers(min_value=1, max_value=120),
    feature_dim=positive_int,
)
@settings(max_examples=20)
def test_camera_config_accepts_valid(
    resolution_width: int,
    resolution_height: int,
    fps: int,
    feature_dim: int,
) -> None:
    cfg = CameraConfig(
        resolution_width=resolution_width,
        resolution_height=resolution_height,
        fps=fps,
        feature_dim=feature_dim,
    )
    assert cfg.resolution_width > 0
    assert cfg.fps > 0


@given(
    mock_hw=st.just(True),
)
@settings(max_examples=5)
def test_settings_accepts_valid_mock(mock_hw: bool) -> None:
    s = Settings(mock_hardware=mock_hw)
    assert s.mock_hardware is True


def test_ultrasonic_rejects_max_le_min() -> None:
    with pytest.raises(ValidationError):
        UltrasonicConfig(trigger_pin=1, echo_pin=2, max_range_m=0.01, min_range_m=0.05)


def test_ultrasonic_rejects_equal_ranges() -> None:
    with pytest.raises(ValidationError):
        UltrasonicConfig(trigger_pin=1, echo_pin=2, max_range_m=1.0, min_range_m=1.0)


@given(
    max_range=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
)
@settings(max_examples=20)
def test_ultrasonic_accepts_valid_ranges(max_range: float) -> None:
    cfg = UltrasonicConfig(
        trigger_pin=1,
        echo_pin=2,
        max_range_m=max_range,
        min_range_m=0.01,
    )
    assert cfg.max_range_m > cfg.min_range_m


def test_esp32_rejects_zero_baud() -> None:
    with pytest.raises(ValidationError):
        ESP32Config(serial_baud=0)


def test_camera_rejects_zero_width() -> None:
    with pytest.raises(ValidationError):
        CameraConfig(resolution_width=0)


def test_loop_config_rejects_zero_hz() -> None:
    with pytest.raises(ValidationError):
        LoopConfig(perception_hz=0)


@given(
    failure_threshold=positive_int,
    recovery_timeout_s=positive_float,
    half_open_max_calls=positive_int,
)
@settings(max_examples=10)
def test_circuit_breaker_config_valid(
    failure_threshold: int,
    recovery_timeout_s: float,
    half_open_max_calls: int,
) -> None:
    cfg = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout_s=recovery_timeout_s,
        half_open_max_calls=half_open_max_calls,
    )
    assert cfg.failure_threshold > 0


# ---------------------------------------------------------------------------
# F-025 — command-set codec invariants over the input space
# ---------------------------------------------------------------------------

_any_velocity = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


@given(vx=_any_velocity, vy=_any_velocity, omega=_any_velocity)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_stock_codec_velocity_clamped_physical_no_quantisation(
    vx: float, vy: float, omega: float
) -> None:
    """Stock X/Z are exact clamped physical values — never PWM-quantised."""
    from mousedroid.comms._utils import clamp
    from mousedroid.comms.command_set import WAVESHARE_STOCK_CODEC

    cfg = ESP32Config.model_validate({"command_set": "waveshare_stock"})
    cmd = WAVESHARE_STOCK_CODEC.build_velocity(vx, vy, omega, cfg)
    assert set(cmd) == {"T", "X", "Z"}
    assert abs(cmd["X"]) <= cfg.max_velocity_mps
    assert abs(cmd["Z"]) <= cfg.max_omega_rads
    assert cmd["X"] == clamp(vx, -cfg.max_velocity_mps, cfg.max_velocity_mps)
    assert cmd["Z"] == clamp(omega, -cfg.max_omega_rads, cfg.max_omega_rads)


@given(vx=_any_velocity, vy=_any_velocity, omega=_any_velocity)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_legacy_codec_pwm_bounded_and_sign_preserving(vx: float, vy: float, omega: float) -> None:
    """Legacy PWM stays within ±MAX_PWM and never flips sign."""
    from mousedroid.comms._utils import MAX_PWM
    from mousedroid.comms.command_set import LEGACY_CODEC

    cfg = ESP32Config()
    cmd = LEGACY_CODEC.build_velocity(vx, vy, omega, cfg)
    for axis, physical in (("vx", vx), ("vy", vy), ("omega", omega)):
        assert abs(cmd[axis]) <= MAX_PWM
        if cmd[axis] != 0:
            assert (cmd[axis] > 0) == (physical > 0)


@given(
    keepalive_hz=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
    multiple=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_heartbeat_window_positive_int_and_monotone(keepalive_hz: float, multiple: float) -> None:
    """The derived heartbeat window is a positive int, monotone in the multiple."""
    from mousedroid.comms.command_set import heartbeat_window_ms

    cfg = ESP32Config.model_validate(
        {
            "command_set": "waveshare_stock",
            "keepalive_hz": keepalive_hz,
            "heartbeat_window_multiple": multiple,
        }
    )
    window = heartbeat_window_ms(cfg)
    assert isinstance(window, int)
    assert window >= 0
    bigger = ESP32Config.model_validate(
        {
            "command_set": "waveshare_stock",
            "keepalive_hz": keepalive_hz,
            "heartbeat_window_multiple": multiple * 2.0,
        }
    )
    assert heartbeat_window_ms(bigger) >= window
