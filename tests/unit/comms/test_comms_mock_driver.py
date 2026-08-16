"""Tests for mousedroid.comms.mock_driver — mock ESP32 communication driver."""

from __future__ import annotations

import pytest

from mousedroid.comms.mock_driver import MockESP32Driver
from mousedroid.comms.protocol import EncoderReading, ESP32CommProtocol
from mousedroid.config.schema import ESP32Config


@pytest.fixture
def mock_driver() -> MockESP32Driver:
    """Create a MockESP32Driver with default config."""
    cfg = ESP32Config()
    return MockESP32Driver(cfg)


class TestMockESP32Driver:
    """Tests for MockESP32Driver."""

    def test_implements_protocol(self, mock_driver: MockESP32Driver) -> None:
        assert isinstance(mock_driver, ESP32CommProtocol)

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, mock_driver: MockESP32Driver) -> None:
        assert not mock_driver._connected
        await mock_driver.connect()
        assert mock_driver._connected
        await mock_driver.disconnect()
        assert not mock_driver._connected

    @pytest.mark.asyncio
    async def test_send_velocity(self, mock_driver: MockESP32Driver) -> None:
        await mock_driver.send_velocity(1.0, 0.5, -0.3)
        assert mock_driver._last_velocity == (1.0, 0.5, -0.3)

    @pytest.mark.asyncio
    async def test_read_encoders_returns_default(self, mock_driver: MockESP32Driver) -> None:
        reading = await mock_driver.read_encoders()
        assert isinstance(reading, EncoderReading)
        assert reading.left_velocity_mps == 0.0
        assert reading.right_velocity_mps == 0.0

    @pytest.mark.asyncio
    async def test_get_battery_voltage(self, mock_driver: MockESP32Driver) -> None:
        voltage = await mock_driver.get_battery_voltage()
        assert voltage == 12.0

    @pytest.mark.asyncio
    async def test_emergency_stop_zeros_velocity(self, mock_driver: MockESP32Driver) -> None:
        await mock_driver.send_velocity(1.0, 0.5, 0.2)
        await mock_driver.emergency_stop()
        assert mock_driver._last_velocity == (0.0, 0.0, 0.0)
