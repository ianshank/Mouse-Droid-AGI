"""Power-chain assertion unit tests using MockESP32Driver."""

from __future__ import annotations

import pytest

from mousedroid.comms.mock_driver import MockESP32Driver
from mousedroid.config.schema import Settings
from mousedroid.diagnostics.power_chain import (
    PowerChainResult,
    assert_power_chain,
)


@pytest.fixture
def settings() -> Settings:
    # tests/conftest.py sets MOUSEDROID_MOCK_HARDWARE=true so bare Settings()
    # bypasses the hardware_requires_pins validator under pytest.
    return Settings()


async def test_zero_velocity_round_trip_succeeds(settings: Settings) -> None:
    driver = MockESP32Driver(cfg=settings.esp32)
    await driver.connect()
    try:
        result: PowerChainResult = await assert_power_chain(
            driver=driver,
            esp32_cfg=settings.esp32,
            allow_motion=False,
        )
        assert result.battery_voltage_v >= 0.0
        assert result.estop_latency_ms <= settings.esp32.emergency_stop_budget_ms
        assert result.notes  # human-readable summary present
    finally:
        await driver.disconnect()


async def test_motion_gate_uses_zero_velocity_when_disallowed(settings: Settings) -> None:
    driver = MockESP32Driver(cfg=settings.esp32)
    await driver.connect()
    try:
        result = await assert_power_chain(
            driver=driver,
            esp32_cfg=settings.esp32,
            allow_motion=False,
        )
        assert result.commanded_velocity_mps == 0.0
    finally:
        await driver.disconnect()


async def test_motion_gate_uses_smoke_velocity_when_allowed(settings: Settings) -> None:
    """allow_motion=True drives the configured smoke setpoint, not zero."""
    driver = MockESP32Driver(cfg=settings.esp32)
    await driver.connect()
    try:
        result = await assert_power_chain(
            driver=driver,
            esp32_cfg=settings.esp32,
            allow_motion=True,
        )
        assert result.commanded_velocity_mps == settings.esp32.smoke_test_velocity_mps
    finally:
        await driver.disconnect()
