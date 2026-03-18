"""Tests for MockTelemetryServer — lifecycle and protocol conformance."""

from __future__ import annotations

from mousedroid.telemetry.mock_server import MockTelemetryServer
from mousedroid.telemetry.protocol import TelemetryServerProtocol


def test_mock_server_initial_state():
    server = MockTelemetryServer()
    assert server.is_running is False
    assert server.client_count == 0
    assert server.received_frames == []


async def test_mock_server_start():
    server = MockTelemetryServer()
    await server.start()
    assert server.is_running is True


async def test_mock_server_stop():
    server = MockTelemetryServer()
    await server.start()
    await server.stop()
    assert server.is_running is False


def test_mock_server_implements_protocol():
    server = MockTelemetryServer()
    assert isinstance(server, TelemetryServerProtocol)


async def test_mock_server_lifecycle():
    server = MockTelemetryServer()
    assert server.is_running is False
    await server.start()
    assert server.is_running is True
    await server.stop()
    assert server.is_running is False
