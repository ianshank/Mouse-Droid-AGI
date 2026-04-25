"""ESP32 serial loopback integration tests for Jetson hardware.

These tests require:
    - ESP32 (Wave Rover) connected via USB serial at ``/dev/ttyUSB0``
    - ``pyserial`` installed: ``pip install pyserial``
    - Robot powered on and motors connected
    - No other process holding the serial port

Run on Jetson::

    pytest tests/hardware/test_esp32_loopback.py -m hardware -v --timeout=60

All velocity, timeout, and threshold values are sourced from config — no
hardcoded numbers appear in assertions.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from mousedroid.config.schema import Settings
from mousedroid.validation.runtime import load_runtime_settings
from tests._jetson_hardware import is_jetson_host, load_jetson_runtime_settings

pytestmark = pytest.mark.hardware

# Skip entire module if pyserial is unavailable
pytest.importorskip("serial", reason="pyserial not available")

JETSON_PROD_CONFIG = "config/jetson_production.yaml"
SERIAL_DEVICE_ENV = "MOUSEDROID_ESP32_SERIAL_PORT"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load_settings() -> Settings:
    """Load Settings through the shared runtime loader.

    Jetson hosts run with real hardware enabled. Non-Jetson hosts keep the
    historical mock-friendly behavior for ad-hoc local execution.
    """
    config_path = Path(JETSON_PROD_CONFIG)
    if not config_path.exists():
        config_path = Path("config/default.yaml")

    if is_jetson_host():
        return load_jetson_runtime_settings()

    settings = load_runtime_settings((config_path,))
    return settings.model_copy(update={"mock_hardware": True})


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Load full Settings from jetson_production.yaml."""
    return _load_settings()


@pytest.fixture(scope="module")
def require_serial_device(settings: Settings) -> str:
    """Skip all tests if the ESP32 serial device is absent."""
    from pathlib import Path

    serial_device = os.getenv(SERIAL_DEVICE_ENV, settings.esp32.serial_port)
    if not Path(serial_device).exists():
        pytest.skip(
            f"ESP32 serial device {serial_device!r} not found. "
            f"Set {SERIAL_DEVICE_ENV} or update esp32.serial_port in config."
        )
    return serial_device


@pytest.fixture(scope="module")
async def driver(settings: Settings, require_serial_device):
    """Build and connect a ResilientESP32Driver, disconnect after module."""
    from mousedroid.factory import build_esp32_driver

    # Build with real hardware (mock_hardware=false from jetson_production.yaml)
    d = build_esp32_driver(settings)
    await d.connect()
    yield d
    # Safe stop before disconnect
    import contextlib

    with contextlib.suppress(Exception):
        await d.emergency_stop()
    await d.disconnect()


# ---------------------------------------------------------------------------
# 1. Connection
# ---------------------------------------------------------------------------


async def test_driver_connects(driver) -> None:
    """Factory builds and connects a ResilientESP32Driver without exceptions."""
    # If we reach this point, connect() succeeded
    assert driver is not None


# ---------------------------------------------------------------------------
# 2. Velocity → encoder loopback
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
async def test_send_velocity_moves_encoders(driver, settings: Settings) -> None:
    """Sending a small forward velocity must result in encoder left_velocity_mps > 0."""
    # Use 20% of max velocity — small, safe nudge
    max_vel = settings.esp32.max_velocity_mps
    test_vel = max_vel * 0.2

    await driver.send_velocity(test_vel, 0.0, 0.0)
    await asyncio.sleep(0.1)  # allow the rover to respond

    enc = await driver.read_encoders()
    # Stop immediately after reading
    await driver.emergency_stop()

    assert enc.left_velocity_mps > 0.0, (
        f"Expected positive left encoder after {test_vel:.2f} m/s, got {enc.left_velocity_mps:.4f}"
    )


@pytest.mark.timeout(10)
async def test_send_zero_velocity_stops_encoders(driver, settings: Settings) -> None:
    """Sending vx=0 must result in zero (or near-zero) encoder velocity."""
    await driver.send_velocity(0.0, 0.0, 0.0)
    await asyncio.sleep(0.15)

    enc = await driver.read_encoders()
    tolerance = settings.esp32.max_velocity_mps * 0.05  # within 5% of max

    assert abs(enc.left_velocity_mps) <= tolerance, (
        f"Expected ~0 velocity after stop, got {enc.left_velocity_mps:.4f} m/s"
    )
    assert abs(enc.right_velocity_mps) <= tolerance, (
        f"Expected ~0 velocity after stop, got {enc.right_velocity_mps:.4f} m/s"
    )


# ---------------------------------------------------------------------------
# 3. Emergency stop latency
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
async def test_emergency_stop_latency(driver, settings: Settings) -> None:
    """emergency_stop() must complete within cfg.esp32.command_timeout_s."""
    max_vel = settings.esp32.max_velocity_mps
    await driver.send_velocity(max_vel * 0.3, 0.0, 0.0)
    await asyncio.sleep(0.05)

    t0 = time.monotonic()
    await driver.emergency_stop()
    elapsed_s = time.monotonic() - t0

    assert elapsed_s <= settings.esp32.command_timeout_s, (
        f"emergency_stop() took {elapsed_s * 1000:.1f} ms; "
        f"timeout={settings.esp32.command_timeout_s * 1000:.0f} ms"
    )


# ---------------------------------------------------------------------------
# 4. Battery voltage
# ---------------------------------------------------------------------------


async def test_battery_voltage_non_negative(driver) -> None:
    """get_battery_voltage() must return a non-negative float."""
    v = await driver.get_battery_voltage()
    assert isinstance(v, float), f"Expected float, got {type(v)}"
    assert v >= 0.0, f"Negative battery voltage: {v}"


async def test_battery_voltage_in_plausible_range(driver, settings: Settings) -> None:
    """When connected, battery voltage should be above the critical threshold."""
    v = await driver.get_battery_voltage()
    # When actually powered, voltage must be above critical threshold defined in config
    if v > 0.0:
        assert v >= settings.safety.battery_critical_v, (
            f"Battery below critical: {v:.2f}V < {settings.safety.battery_critical_v:.2f}V"
        )


# ---------------------------------------------------------------------------
# 5. Circuit breaker integration
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
async def test_circuit_breaker_opens_after_failures(settings: Settings) -> None:
    """After failure_threshold errors, the circuit breaker must open."""
    from unittest.mock import AsyncMock

    from mousedroid.comms.mock_driver import MockESP32Driver
    from mousedroid.resilience.circuit_breaker import CircuitOpenError
    from mousedroid.resilience.resilient_driver import ResilientESP32Driver

    failing_inner: AsyncMock = AsyncMock(spec=MockESP32Driver)  # type: ignore[type-abstract]
    failing_inner.send_velocity = AsyncMock(side_effect=RuntimeError("simulated_serial_error"))
    failing_inner.connect = AsyncMock()

    resilient = ResilientESP32Driver(failing_inner, settings.retry, settings.circuit_breaker)
    await resilient.connect()

    import contextlib

    failure_threshold = settings.circuit_breaker.failure_threshold
    # Drive the circuit past its failure threshold
    for _ in range(failure_threshold + 2):
        with contextlib.suppress(RuntimeError, CircuitOpenError):
            await resilient.send_velocity(0.1, 0.0, 0.0)

    # Circuit must now be open — next call must raise CircuitOpenError
    with pytest.raises(CircuitOpenError):
        await resilient.send_velocity(0.1, 0.0, 0.0)


async def test_emergency_stop_bypasses_circuit_breaker(settings: Settings) -> None:
    """emergency_stop() must succeed even when circuit breaker is open."""
    from unittest.mock import AsyncMock

    from mousedroid.comms.mock_driver import MockESP32Driver
    from mousedroid.resilience.circuit_breaker import CircuitOpenError
    from mousedroid.resilience.resilient_driver import ResilientESP32Driver

    failing_inner: AsyncMock = AsyncMock(spec=MockESP32Driver)  # type: ignore[type-abstract]
    failing_inner.send_velocity = AsyncMock(side_effect=RuntimeError("fail"))
    failing_inner.emergency_stop = AsyncMock()  # stop succeeds
    failing_inner.connect = AsyncMock()

    resilient = ResilientESP32Driver(failing_inner, settings.retry, settings.circuit_breaker)
    await resilient.connect()

    import contextlib

    # Open the circuit breaker
    for _ in range(settings.circuit_breaker.failure_threshold + 2):
        with contextlib.suppress(RuntimeError, CircuitOpenError):
            await resilient.send_velocity(0.1, 0.0, 0.0)

    # emergency_stop must NOT raise CircuitOpenError — it bypasses the breaker
    await resilient.emergency_stop()
    failing_inner.emergency_stop.assert_called()
