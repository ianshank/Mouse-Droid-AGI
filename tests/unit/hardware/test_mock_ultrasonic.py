from __future__ import annotations

import pytest

from mousedroid.config.schema import UltrasonicConfig
from mousedroid.hardware.sensors.mock_ultrasonic import MockUltrasonic


@pytest.fixture
def sensor() -> MockUltrasonic:
    cfg = UltrasonicConfig(trigger_pin=23, echo_pin=24, max_range_m=4.0, min_range_m=0.02)
    return MockUltrasonic(cfg)


def test_construct(sensor: MockUltrasonic):
    assert sensor is not None


async def test_read_distance_m_default(sensor: MockUltrasonic):
    d = await sensor.read_distance_m()
    assert d == pytest.approx((4.0 + 0.02) / 2.0)


async def test_set_distance_changes_value(sensor: MockUltrasonic):
    sensor.set_distance(1.5)
    d = await sensor.read_distance_m()
    assert d == 1.5


def test_max_range_m(sensor: MockUltrasonic):
    assert sensor.max_range_m == 4.0


def test_min_range_m(sensor: MockUltrasonic):
    assert sensor.min_range_m == 0.02


async def test_read_distance_after_set(sensor: MockUltrasonic):
    sensor.set_distance(0.1)
    sensor.set_distance(2.0)
    d = await sensor.read_distance_m()
    assert d == 2.0
