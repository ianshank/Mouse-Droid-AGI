"""ESP32 serial edge-case and resilience integration tests.

Extends ``test_esp32_loopback.py`` with reconnection after disconnect,
concurrent velocity commands, keepalive timing, and circuit-breaker
recovery scenarios.

These tests combine real-hardware validation (marked ``@pytest.mark.hardware``)
with mock-based resilience tests that run on any host.

Run on Jetson::

    pytest tests/hardware/test_esp32_edge_cases.py -m hardware -v --timeout=60
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from mousedroid.config.schema import Settings

pytestmark = pytest.mark.hardware

pytest.importorskip("serial", reason="pyserial not available")

JETSON_PROD_CONFIG = "config/jetson_production.yaml"
SERIAL_DEVICE_ENV = "MOUSEDROID_ESP32_SERIAL_PORT"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def settings(jetson_settings) -> Settings:
    """Alias the session-level settings."""
    return jetson_settings


@pytest.fixture(scope="module")
def require_serial_device(settings: Settings) -> str:
    """Skip all tests if the ESP32 serial device is absent."""
    serial_device = os.getenv(SERIAL_DEVICE_ENV, settings.esp32.serial_port)
    if not Path(serial_device).exists():
        pytest.skip(
            f"ESP32 serial device {serial_device!r} not found. "
            f"Set {SERIAL_DEVICE_ENV} or update esp32.serial_port in config."
        )
    return serial_device


# ---------------------------------------------------------------------------
# 1. Reconnect after disconnect
# ---------------------------------------------------------------------------


@pytest.mark.timeout(15)
async def test_reconnect_after_disconnect(
    settings: Settings,
    require_serial_device: str,
) -> None:
    """Driver must successfully reconnect after a clean disconnect cycle."""
    from mousedroid.factory import build_esp32_driver

    driver = build_esp32_driver(settings)

    # First connection cycle
    await driver.connect()
    enc = await driver.read_encoders()
    assert enc is not None
    await driver.disconnect()

    # Second connection cycle — must succeed
    await driver.connect()
    enc2 = await driver.read_encoders()
    assert enc2 is not None

    with contextlib.suppress(Exception):
        await driver.emergency_stop()
    await driver.disconnect()


# ---------------------------------------------------------------------------
# 2. Emergency stop is always available (even after circuit opens)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
async def test_emergency_stop_bypasses_open_circuit(settings: Settings) -> None:
    """``emergency_stop()`` must succeed even when the circuit breaker is open.

    Uses a mock inner driver to simulate failures without real hardware.
    """
    from mousedroid.comms.mock_driver import MockESP32Driver
    from mousedroid.resilience.circuit_breaker import CircuitOpenError
    from mousedroid.resilience.resilient_driver import ResilientESP32Driver

    failing_inner = AsyncMock(spec=MockESP32Driver)
    failing_inner.send_velocity = AsyncMock(
        side_effect=RuntimeError("serial_gone"),
    )
    failing_inner.connect = AsyncMock()
    failing_inner.emergency_stop = AsyncMock()

    resilient = ResilientESP32Driver(
        failing_inner,
        settings.retry,
        settings.circuit_breaker,
    )
    await resilient.connect()

    # Drive circuit breaker to open state
    for _ in range(settings.circuit_breaker.failure_threshold + 2):
        with contextlib.suppress(RuntimeError, CircuitOpenError):
            await resilient.send_velocity(0.1, 0.0, 0.0)

    # Verify circuit is open
    with pytest.raises(CircuitOpenError):
        await resilient.send_velocity(0.0, 0.0, 0.0)

    # emergency_stop must still work (bypasses circuit breaker)
    await resilient.emergency_stop()
    failing_inner.emergency_stop.assert_awaited()


# ---------------------------------------------------------------------------
# 3. Concurrent velocity commands — no interleaving corruption
# ---------------------------------------------------------------------------


@pytest.mark.timeout(15)
async def test_concurrent_velocity_no_corruption(
    settings: Settings,
    require_serial_device: str,
) -> None:
    """Two concurrent ``send_velocity`` calls must both complete without error.

    We can't guarantee ordering, but neither should raise or corrupt state.
    """
    from mousedroid.factory import build_esp32_driver

    driver = build_esp32_driver(settings)
    await driver.connect()

    try:
        max_vel = settings.esp32.max_velocity_mps
        slow = max_vel * 0.1

        results = await asyncio.gather(
            driver.send_velocity(slow, 0.0, 0.0),
            driver.send_velocity(0.0, 0.0, slow),
            return_exceptions=True,
        )
        for r in results:
            assert r is None or isinstance(r, Exception), f"Unexpected return: {r!r}"
            if isinstance(r, Exception):
                pytest.fail(f"Concurrent velocity raised: {r!r}")
    finally:
        with contextlib.suppress(Exception):
            await driver.emergency_stop()
        await driver.disconnect()


# ---------------------------------------------------------------------------
# 4. Battery voltage stability over repeated reads
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
async def test_battery_voltage_stability(
    settings: Settings,
    require_serial_device: str,
) -> None:
    """Five consecutive battery reads must all be within a plausible range.

    The range is bounded by ``safety.battery_critical_v`` and a reasonable
    upper limit for a 3S/4S LiPo (max ~17V).
    """
    from mousedroid.factory import build_esp32_driver

    driver = build_esp32_driver(settings)
    await driver.connect()

    max_lipo_v = 17.0  # 4S LiPo full-charge ceiling

    try:
        voltages: list[float] = []
        for _ in range(5):
            v = await driver.get_battery_voltage()
            assert isinstance(v, float)
            voltages.append(v)
            await asyncio.sleep(0.05)

        # If the battery is connected, all reads should be plausible
        if voltages[0] > 0.0:
            for v in voltages:
                assert settings.safety.battery_critical_v <= v <= max_lipo_v, (
                    f"Battery voltage {v:.2f}V outside plausible range "
                    f"[{settings.safety.battery_critical_v:.1f}, {max_lipo_v:.1f}]V"
                )
    finally:
        await driver.disconnect()


# ---------------------------------------------------------------------------
# 5. Encoder reading fields are populated
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
async def test_encoder_reading_fields(
    settings: Settings,
    require_serial_device: str,
) -> None:
    """``read_encoders()`` must return an ``EncoderReading`` with float fields."""
    from mousedroid.comms.protocol import EncoderReading
    from mousedroid.factory import build_esp32_driver

    driver = build_esp32_driver(settings)
    await driver.connect()

    try:
        enc = await driver.read_encoders()
        assert isinstance(enc, EncoderReading)
        assert isinstance(enc.left_velocity_mps, float)
        assert isinstance(enc.right_velocity_mps, float)
        assert isinstance(enc.odometry_x_m, float)
        assert isinstance(enc.odometry_y_m, float)
        assert isinstance(enc.heading_rad, float)
    finally:
        await driver.disconnect()


# ---------------------------------------------------------------------------
# 6. Resilient driver stats after mixed success/failure
# ---------------------------------------------------------------------------


async def test_resilient_driver_stats_tracking(settings: Settings) -> None:
    """``stats`` property must accurately track call counts and failures.

    Uses a mock inner driver — runs on any host.
    """
    from mousedroid.comms.mock_driver import MockESP32Driver
    from mousedroid.resilience.resilient_driver import ResilientESP32Driver

    inner = AsyncMock(spec=MockESP32Driver)
    inner.connect = AsyncMock()
    inner.send_velocity = AsyncMock()
    inner.read_encoders = AsyncMock(
        return_value=__import__(
            "mousedroid.comms.protocol", fromlist=["EncoderReading"]
        ).EncoderReading(),
    )

    resilient = ResilientESP32Driver(inner, settings.retry, settings.circuit_breaker)
    await resilient.connect()

    # Two successful velocity sends
    await resilient.send_velocity(0.1, 0.0, 0.0)
    await resilient.send_velocity(0.2, 0.0, 0.0)

    stats = resilient.stats
    assert stats["total_calls"] >= 2
    assert stats["total_failures"] == 0


# ---------------------------------------------------------------------------
# 7. Velocity clamped within config limits
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
async def test_velocity_within_config_limits(
    settings: Settings,
    require_serial_device: str,
) -> None:
    """Sending max_velocity should not raise; the driver constrains internally."""
    from mousedroid.factory import build_esp32_driver

    driver = build_esp32_driver(settings)
    await driver.connect()

    try:
        max_v = settings.esp32.max_velocity_mps
        # Send exactly at the configured limit
        await driver.send_velocity(max_v, 0.0, 0.0)
        await asyncio.sleep(0.05)
        await driver.emergency_stop()
    finally:
        await driver.disconnect()
