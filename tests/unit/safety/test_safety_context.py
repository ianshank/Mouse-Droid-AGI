from __future__ import annotations

import dataclasses
import math

import pytest

from mousedroid.safety.context import SafetyContext


def test_safety_context_defaults():
    ctx = SafetyContext()
    assert ctx.surprise == 0.0
    assert ctx.valid_sensor_count == 0
    assert ctx.loop_time_ms == 0.0


def test_safety_context_is_frozen():
    ctx = SafetyContext()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.surprise = 1.0  # type: ignore[misc]


def test_safety_context_custom_values():
    ctx = SafetyContext(
        surprise=1.5,
        valid_sensor_count=3,
        loop_time_ms=33.0,
        ultrasonic_dist_m=0.5,
        forward_clearance_ok=False,
        battery_voltage=11.0,
        gpu_temp_c=65.0,
        esp32_connected=False,
        is_emergency=True,
    )
    assert ctx.surprise == 1.5
    assert ctx.valid_sensor_count == 3
    assert ctx.forward_clearance_ok is False
    assert ctx.esp32_connected is False


def test_safety_context_default_ultrasonic_dist_is_inf():
    ctx = SafetyContext()
    assert ctx.ultrasonic_dist_m == math.inf


def test_safety_context_default_is_emergency_false():
    ctx = SafetyContext()
    assert ctx.is_emergency is False


def test_safety_context_with_emergency_true():
    ctx = SafetyContext(is_emergency=True)
    assert ctx.is_emergency is True


def test_safety_context_default_battery_voltage():
    ctx = SafetyContext()
    assert ctx.battery_voltage == 12.0


def test_safety_context_default_forward_clearance():
    ctx = SafetyContext()
    assert ctx.forward_clearance_ok is True
