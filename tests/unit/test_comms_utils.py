"""Tests for ``comms/_utils`` helper functions.

Covers ``build_velocity_cmd`` and ``parse_encoder_reading``, which are the
single-source-of-truth implementations used by both serial and WiFi drivers
via ``BaseESP32Driver``.
"""

from __future__ import annotations

import pytest

from mousedroid.comms._utils import (
    ESP32_CMD_TYPE_BATTERY,
    ESP32_CMD_TYPE_STOP,
    ESP32_CMD_TYPE_VELOCITY,
    MAX_PWM,
    build_velocity_cmd,
    clamp,
    parse_encoder_reading,
)
from mousedroid.config.schema import ESP32Config


# ---------------------------------------------------------------------------
# clamp (existing, keep for regression)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "lo", "hi", "expected"),
    [
        (0.5, 0.0, 1.0, 0.5),  # within range
        (-0.5, 0.0, 1.0, 0.0),  # below lower bound
        (1.5, 0.0, 1.0, 1.0),  # above upper bound
        (-1.5, -1.0, 1.0, -1.0),  # negative lower bound
        (3.0, -1.0, 1.0, 1.0),  # positive upper bound
        (0.0, 0.0, 0.0, 0.0),  # degenerate range
    ],
)
def test_clamp(value: float, lo: float, hi: float, expected: float) -> None:
    assert clamp(value, lo, hi) == expected


# ---------------------------------------------------------------------------
# build_velocity_cmd
# ---------------------------------------------------------------------------


def _cfg(max_vel: float = 1.0, max_omega: float = 1.0) -> ESP32Config:
    return ESP32Config(max_velocity_mps=max_vel, max_omega_rads=max_omega)


def test_build_velocity_cmd_type_is_velocity():
    cmd = build_velocity_cmd(0.0, 0.0, 0.0, _cfg())
    assert cmd["T"] == ESP32_CMD_TYPE_VELOCITY


def test_build_velocity_cmd_zero_velocity():
    cmd = build_velocity_cmd(0.0, 0.0, 0.0, _cfg())
    assert cmd["vx"] == 0
    assert cmd["vy"] == 0
    assert cmd["omega"] == 0


def test_build_velocity_cmd_full_forward():
    cmd = build_velocity_cmd(1.0, 0.0, 0.0, _cfg(max_vel=1.0))
    assert cmd["vx"] == MAX_PWM
    assert cmd["vy"] == 0
    assert cmd["omega"] == 0


def test_build_velocity_cmd_full_reverse():
    cmd = build_velocity_cmd(-1.0, 0.0, 0.0, _cfg(max_vel=1.0))
    assert cmd["vx"] == -MAX_PWM


def test_build_velocity_cmd_half_speed():
    cmd = build_velocity_cmd(0.5, 0.0, 0.0, _cfg(max_vel=1.0))
    assert cmd["vx"] == 127  # int(0.5 * 255) = 127


def test_build_velocity_cmd_clamps_above_max():
    cmd = build_velocity_cmd(100.0, 0.0, 0.0, _cfg(max_vel=1.0))
    assert cmd["vx"] == MAX_PWM


def test_build_velocity_cmd_clamps_below_neg_max():
    cmd = build_velocity_cmd(-100.0, 0.0, 0.0, _cfg(max_vel=1.0))
    assert cmd["vx"] == -MAX_PWM


def test_build_velocity_cmd_omega_scaling():
    cmd = build_velocity_cmd(
        0.0,
        0.0,
        0.5,
        _cfg(max_vel=1.0, max_omega=1.0),
    )
    assert cmd["omega"] == 127


def test_build_velocity_cmd_custom_max_vel():
    # At 1.0 m/s with max_vel = 2.0 → ratio = 0.5 → 127 PWM
    cmd = build_velocity_cmd(1.0, 0.0, 0.0, _cfg(max_vel=2.0))
    assert cmd["vx"] == 127


def test_build_velocity_cmd_returns_int_values():
    cmd = build_velocity_cmd(0.3, -0.3, 0.6, _cfg())
    assert isinstance(cmd["vx"], int)
    assert isinstance(cmd["vy"], int)
    assert isinstance(cmd["omega"], int)


def test_build_velocity_cmd_all_axes():
    cmd = build_velocity_cmd(0.5, -0.5, 1.0, _cfg(max_vel=1.0, max_omega=1.0))
    assert cmd["T"] == ESP32_CMD_TYPE_VELOCITY
    assert cmd["vx"] == 127
    assert cmd["vy"] == -127
    assert cmd["omega"] == MAX_PWM


# ---------------------------------------------------------------------------
# parse_encoder_reading
# ---------------------------------------------------------------------------


def test_parse_encoder_reading_full_data():
    from mousedroid.comms.protocol import EncoderReading

    data = {"lv": 0.5, "rv": 0.3, "ox": 1.0, "oy": 2.0, "h": 0.1, "ts": 100.0}
    reading = parse_encoder_reading(data)
    assert isinstance(reading, EncoderReading)
    assert reading.left_velocity_mps == pytest.approx(0.5)
    assert reading.right_velocity_mps == pytest.approx(0.3)
    assert reading.odometry_x_m == pytest.approx(1.0)
    assert reading.odometry_y_m == pytest.approx(2.0)
    assert reading.heading_rad == pytest.approx(0.1)
    assert reading.timestamp == pytest.approx(100.0)


def test_parse_encoder_reading_empty_dict():
    reading = parse_encoder_reading({})
    assert reading.left_velocity_mps == 0.0
    assert reading.right_velocity_mps == 0.0
    assert reading.odometry_x_m == 0.0
    assert reading.odometry_y_m == 0.0
    assert reading.heading_rad == 0.0
    assert reading.timestamp == 0.0


def test_parse_encoder_reading_partial_data():
    data = {"lv": 1.2, "ts": 42.0}
    reading = parse_encoder_reading(data)
    assert reading.left_velocity_mps == pytest.approx(1.2)
    assert reading.right_velocity_mps == 0.0
    assert reading.timestamp == pytest.approx(42.0)


def test_parse_encoder_reading_returns_floats():
    data = {"lv": 1, "rv": 2, "ox": 3, "oy": 4, "h": 5, "ts": 6}
    reading = parse_encoder_reading(data)
    assert isinstance(reading.left_velocity_mps, float)
    assert isinstance(reading.right_velocity_mps, float)


# ---------------------------------------------------------------------------
# Protocol constants sanity checks
# ---------------------------------------------------------------------------


def test_constants_are_distinct():
    constants = {
        ESP32_CMD_TYPE_STOP,
        ESP32_CMD_TYPE_VELOCITY,
        ESP32_CMD_TYPE_BATTERY,
    }
    assert len(constants) == 3


def test_max_pwm_is_positive():
    assert MAX_PWM > 0
