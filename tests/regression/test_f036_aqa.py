"""AQA pins for F-036 stock IMU parse (no RSSM slot widen).

Pins the dataclass/protocol surface rather than a new Pydantic field:
legacy ``EncoderReading`` construction still works, new attitude fields
default to the pre-feature zeros, and ``heading_for_motor`` is a real
method. SENSOR_SLOT_MAP must stay width-5 so this cannot silently grow
into F-040.
"""

from __future__ import annotations

import inspect

from mousedroid.comms.protocol import EncoderReading
from mousedroid.constants import N_SENSOR_MODALITIES_WITH_LIDAR, SENSOR_SLOT_MAP


def test_encoder_reading_attitude_fields_default_inert() -> None:
    """A pre-F-036 ``EncoderReading()`` still means 'no IMU'."""
    reading = EncoderReading()
    assert reading.roll_rad == 0.0
    assert reading.pitch_rad == 0.0
    assert reading.yaw_rad == 0.0
    assert reading.imu_valid is False
    assert reading.heading_rad == 0.0
    assert reading.heading_for_motor() == 0.0


def test_heading_for_motor_is_a_bound_method() -> None:
    """``isinstance`` on a Protocol would not catch a non-callable attribute."""
    member = EncoderReading.heading_for_motor
    assert callable(member)
    sig = inspect.signature(member)
    assert list(sig.parameters) == ["self"]


def test_legacy_positional_construction_still_works() -> None:
    """New fields were appended with defaults — existing call sites stay valid."""
    reading = EncoderReading(0.1, 0.2, 1.0, 2.0, 0.3, 9.0)
    assert reading.left_velocity_mps == 0.1
    assert reading.heading_rad == 0.3
    assert reading.timestamp == 9.0
    assert reading.imu_valid is False
    assert reading.heading_for_motor() == 0.3


def test_sensor_slot_map_does_not_gain_an_imu_key() -> None:
    """F-036 must not widen the ONNX-stable mask; that is F-040."""
    assert "imu" not in SENSOR_SLOT_MAP
    assert N_SENSOR_MODALITIES_WITH_LIDAR == 5
    assert set(SENSOR_SLOT_MAP) == {"vision", "ultrasonic", "motor", "audio", "lidar"}
