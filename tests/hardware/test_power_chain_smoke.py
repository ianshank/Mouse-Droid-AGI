"""Hardware smoke — battery + motor command + e-stop within budget."""

from __future__ import annotations

import pytest

from mousedroid.config.schema import Settings
from mousedroid.diagnostics.power_chain import assert_power_chain

pytestmark = pytest.mark.hardware


async def test_power_chain_within_budget(
    jetson_settings: Settings,
    allow_motion: bool,
) -> None:
    from mousedroid.factory import build_esp32_driver

    if jetson_settings.mock_hardware:
        pytest.skip("mock_hardware=true; power chain smoke requires real ESP32")

    driver = build_esp32_driver(jetson_settings)
    await driver.connect()
    try:
        result = await assert_power_chain(
            driver=driver,
            esp32_cfg=jetson_settings.esp32,
            allow_motion=allow_motion,
        )
        assert result.estop_latency_ms <= jetson_settings.esp32.emergency_stop_budget_ms, (
            f"e-stop latency {result.estop_latency_ms:.1f}ms exceeded budget "
            f"{jetson_settings.esp32.emergency_stop_budget_ms:.0f}ms"
        )
        if jetson_settings.safety.battery_critical_v > 0.0:
            assert result.battery_voltage_v >= jetson_settings.safety.battery_critical_v, (
                f"battery {result.battery_voltage_v:.2f}V below critical "
                f"{jetson_settings.safety.battery_critical_v:.2f}V"
            )
    finally:
        await driver.disconnect()
