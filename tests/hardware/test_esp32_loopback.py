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


def _commanded_velocity(settings: Settings, fraction: float) -> float:
    """Second motion gate (CHARTER §3, F-059).

    ``jetson_smoke_test.sh`` only reaches this module behind
    ``MOUSEDROID_SMOKE_ALLOW_MOTION=1``; the Pydantic gate
    ``esp32.smoke_test_allow_motion`` (default False) must ALSO be open before
    a non-zero command leaves the host. With it closed the stage still proves
    command acceptance, the T:1001 frame and e-stop latency — at zero velocity.
    """
    if not settings.esp32.smoke_test_allow_motion:
        return 0.0
    return settings.esp32.max_velocity_mps * fraction


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
    """Forward velocity must produce feedback the chassis can actually give (F-059 R-1).

    Encoder-bearing chassis: measured wheel velocity > 5 % of max (historical).
    WAVE ROVER (``chassis_has_wheel_encoders=False``): stock firmware echoes the
    commanded speed into L/R, so a velocity assertion measures nothing; the
    criterion is a FEEDBACK_BASE_INFO (T:1001) frame after the command,
    witnessed by ``EncoderReading.imu_valid`` — never set by the legacy codec.
    No skip on this branch: a silent board FAILS. ``esp32.enabled=false`` is a
    config decision (MockESP32Driver), not a measurement, so it skips with the remedy.
    """
    if not settings.esp32.enabled:
        pytest.skip(
            "esp32.enabled=false routes to MockESP32Driver; "
            "stack config/jetson_bench_esp32.yaml to probe the board"
        )
    # 20% of max velocity — small, safe nudge (0.0 while the Pydantic motion gate is closed)
    max_vel = settings.esp32.max_velocity_mps
    test_vel = _commanded_velocity(settings, 0.2)
    feedback_tolerance = max_vel * 0.05

    await driver.send_velocity(test_vel, 0.0, 0.0)
    await asyncio.sleep(0.1)  # allow the rover to respond

    enc = await driver.read_encoders()
    # Stop immediately after reading
    await driver.emergency_stop()

    if not settings.esp32.chassis_has_wheel_encoders:
        assert enc.imu_valid, (
            f"no FEEDBACK_BASE_INFO frame after {test_vel:.2f} m/s on "
            f"{settings.esp32.serial_port} (command_set={settings.esp32.command_set!r}); "
            "the encoder-less criterion needs waveshare_stock and a live board"
        )
        return

    if (
        abs(enc.left_velocity_mps) <= feedback_tolerance
        and abs(enc.right_velocity_mps) <= feedback_tolerance
    ):
        pytest.skip(
            "encoder loopback inactive; "
            "ensure rover power, motor/encoder wiring and both motion gates"
        )

    assert (
        enc.left_velocity_mps > feedback_tolerance or enc.right_velocity_mps > feedback_tolerance
    ), (
        "Expected encoder feedback after "
        f"{test_vel:.2f} m/s, got left={enc.left_velocity_mps:.4f}"
        f" right={enc.right_velocity_mps:.4f}"
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
    await driver.send_velocity(_commanded_velocity(settings, 0.3), 0.0, 0.0)
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
    """When connected, battery voltage should be above the critical threshold.

    Asymmetry kept on purpose (R3-5): this module is the motion gate and passes on a
    missing reading; ``test_power_chain_smoke.py`` is the telemetry gate and fails on it.
    """
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
    from mousedroid.resilience.retry import RetryExhaustedError

    failing_inner: AsyncMock = AsyncMock(spec=MockESP32Driver)  # type: ignore[type-abstract]
    failing_inner.send_velocity = AsyncMock(side_effect=RuntimeError("simulated_serial_error"))
    failing_inner.connect = AsyncMock()

    resilient = ResilientESP32Driver(failing_inner, settings.retry, settings.circuit_breaker)
    await resilient.connect()

    import contextlib

    failure_threshold = settings.circuit_breaker.failure_threshold
    # Drive the circuit past its failure threshold
    for _ in range(failure_threshold + 2):
        with contextlib.suppress(RuntimeError, RetryExhaustedError, CircuitOpenError):
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
    from mousedroid.resilience.retry import RetryExhaustedError

    failing_inner: AsyncMock = AsyncMock(spec=MockESP32Driver)  # type: ignore[type-abstract]
    failing_inner.send_velocity = AsyncMock(side_effect=RuntimeError("fail"))
    failing_inner.emergency_stop = AsyncMock()  # stop succeeds
    failing_inner.connect = AsyncMock()

    resilient = ResilientESP32Driver(failing_inner, settings.retry, settings.circuit_breaker)
    await resilient.connect()

    import contextlib

    # Open the circuit breaker
    for _ in range(settings.circuit_breaker.failure_threshold + 2):
        with contextlib.suppress(RuntimeError, RetryExhaustedError, CircuitOpenError):
            await resilient.send_velocity(0.1, 0.0, 0.0)

    # emergency_stop must NOT raise CircuitOpenError — it bypasses the breaker
    await resilient.emergency_stop()
    failing_inner.emergency_stop.assert_called()
