"""Tests for TelemetryFrame and telemetry protocols."""

from __future__ import annotations

from mousedroid.telemetry.protocol import (
    TelemetryFrame,
    TelemetryPublisherProtocol,
    TelemetryServerProtocol,
)


def test_telemetry_frame_defaults():
    frame = TelemetryFrame()
    assert frame.timestamp == 0.0
    assert frame.distance_m == 0.0
    assert frame.motor_state == []
    assert frame.vision_norm == 0.0
    assert frame.audio_rms == 0.0
    assert frame.valid_mask == []
    assert frame.encoder == {}
    assert frame.battery_voltage == 0.0
    assert frame.safety == {}
    assert frame.health == {}
    assert frame.loop_time_ms == 0.0
    assert frame.tick_count == 0


def test_telemetry_frame_with_values():
    frame = TelemetryFrame(
        timestamp=123.456,
        distance_m=1.5,
        motor_state=[0.1, 0.2, 0.3, 11.8],
        vision_norm=42.0,
        audio_rms=0.05,
        valid_mask=[1.0, 1.0, 1.0, 0.0],
        battery_voltage=11.8,
        safety={"is_emergency": False},
        loop_time_ms=15.2,
        tick_count=100,
    )
    assert frame.timestamp == 123.456
    assert frame.distance_m == 1.5
    assert frame.vision_norm == 42.0
    assert frame.tick_count == 100


def test_telemetry_frame_to_dict():
    frame = TelemetryFrame(timestamp=1.0, distance_m=2.5, tick_count=42)
    d = frame.to_dict()
    assert isinstance(d, dict)
    assert d["timestamp"] == 1.0
    assert d["distance_m"] == 2.5
    assert d["tick_count"] == 42
    assert "motor_state" in d
    assert "vision_norm" in d


def test_telemetry_frame_is_frozen():
    import pytest

    frame = TelemetryFrame()
    with pytest.raises(AttributeError):
        frame.timestamp = 999.0  # type: ignore[misc]


def test_telemetry_frame_to_dict_is_json_serializable():
    import json

    frame = TelemetryFrame(
        timestamp=1.0,
        motor_state=[0.1, 0.2, 0.3, 11.8],
        valid_mask=[1.0, 1.0, 1.0, 0.0],
        safety={"is_emergency": False, "violations": []},
    )
    serialized = json.dumps(frame.to_dict())
    assert isinstance(serialized, str)
    deserialized = json.loads(serialized)
    assert deserialized["timestamp"] == 1.0


def test_publisher_protocol_is_runtime_checkable():
    from mousedroid.config.schema import TelemetryConfig
    from mousedroid.telemetry.publisher import TelemetryPublisher

    pub = TelemetryPublisher(TelemetryConfig())
    assert isinstance(pub, TelemetryPublisherProtocol)


def test_server_protocol_is_runtime_checkable():
    from mousedroid.telemetry.mock_server import MockTelemetryServer

    server = MockTelemetryServer()
    assert isinstance(server, TelemetryServerProtocol)


def test_telemetry_frame_encoder_field():
    frame = TelemetryFrame(
        encoder={"left_velocity_mps": 0.5, "right_velocity_mps": 0.3}
    )
    assert frame.encoder["left_velocity_mps"] == 0.5
