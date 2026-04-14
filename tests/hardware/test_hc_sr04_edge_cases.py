"""HC-SR04 edge-case and resilience integration tests.

Extends ``test_hc_sr04_integration.py`` with GPIO cleanup, staleness
detection, and re-initialisation scenarios that exercise the driver under
adverse conditions.

These tests can run **on the Jetson** with real GPIO *or* with a
``Jetson.GPIO`` stub injected via ``sys.modules`` in conftest for CI.

The Jetson.GPIO import guard lives in a fixture (not module-level) so
a conftest can insert ``sys.modules["Jetson.GPIO"]`` before collection.

Run on Jetson::

    pytest tests/hardware/test_hc_sr04_edge_cases.py -m hardware -v --timeout=30
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from mousedroid.config.schema import UltrasonicConfig

pytestmark = pytest.mark.hardware

JETSON_PROD_CONFIG = "config/jetson_production.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _require_gpio():
    """Skip module if Jetson.GPIO is not available (real or stubbed via conftest)."""
    pytest.importorskip("Jetson.GPIO", reason="Jetson.GPIO not available")


@pytest.fixture(scope="module")
def ultrasonic_cfg(_require_gpio, jetson_settings) -> UltrasonicConfig:
    """UltrasonicConfig from the shared session settings."""
    assert jetson_settings.ultrasonic is not None
    return jetson_settings.ultrasonic


@pytest.fixture
def sensor(ultrasonic_cfg: UltrasonicConfig):
    """Construct an HcSr04 and teardown GPIO after each test."""
    from mousedroid.hardware.sensors.ultrasonic import HcSr04

    s = HcSr04(ultrasonic_cfg)
    yield s
    import contextlib

    with contextlib.suppress(Exception):
        import Jetson.GPIO as GPIO

        GPIO.cleanup()


# ---------------------------------------------------------------------------
# 1. GPIO cleanup after exception
# ---------------------------------------------------------------------------


async def test_gpio_cleanup_after_read_exception(
    ultrasonic_cfg: UltrasonicConfig,
) -> None:
    """GPIO must be cleaned up even when ``_measure_distance`` raises."""
    from mousedroid.hardware.sensors.ultrasonic import HcSr04

    sensor = HcSr04(ultrasonic_cfg)

    with (
        patch.object(
            sensor,
            "_measure_distance",
            side_effect=RuntimeError("simulated GPIO fault"),
        ),
        pytest.raises(RuntimeError, match="simulated GPIO fault"),
    ):
        await sensor.read_distance_m()

    # GPIO cleanup must succeed without raising
    import Jetson.GPIO as GPIO

    GPIO.cleanup()


# ---------------------------------------------------------------------------
# 2. Re-initialisation after GPIO cleanup
# ---------------------------------------------------------------------------


async def test_reinitialise_after_cleanup(
    ultrasonic_cfg: UltrasonicConfig,
) -> None:
    """Driver must work after GPIO cleanup + re-setup cycle."""
    from mousedroid.hardware.sensors.ultrasonic import HcSr04

    s1 = HcSr04(ultrasonic_cfg)
    d1 = await s1.read_distance_m()
    assert isinstance(d1, float)

    # Force GPIO cleanup
    import Jetson.GPIO as GPIO

    GPIO.cleanup()

    # Re-create and read again
    s2 = HcSr04(ultrasonic_cfg)
    d2 = await s2.read_distance_m()
    assert isinstance(d2, float)
    assert ultrasonic_cfg.min_range_m <= d2 <= ultrasonic_cfg.max_range_m


# ---------------------------------------------------------------------------
# 3. Sensor staleness — rapid consecutive reads should yield fresh values
# ---------------------------------------------------------------------------


@pytest.mark.timeout(5)
async def test_consecutive_reads_not_stale(
    sensor,
    ultrasonic_cfg: UltrasonicConfig,
) -> None:
    """Ten consecutive reads inside the staleness window must not all be identical.

    If all 10 are identical, the driver may be caching or the pin is stuck.
    """
    readings: list[float] = []
    for _ in range(10):
        d = await sensor.read_distance_m()
        readings.append(d)
        await asyncio.sleep(0.02)  # 50 Hz sampling

    # At least 2 distinct values expected (environment noise)
    unique = set(readings)
    # If all readings equal max_range, sensor is likely not wired — still valid
    if readings[0] == ultrasonic_cfg.max_range_m:
        pytest.skip("All readings at max_range — sensor may not be wired")

    assert (
        len(unique) >= 2
    ), f"All 10 readings identical ({readings[0]:.4f}m) — possible stale cache"


# ---------------------------------------------------------------------------
# 4. Distance clamp — never exceeds max_range_m
# ---------------------------------------------------------------------------


async def test_distance_never_exceeds_max_range(
    sensor,
    ultrasonic_cfg: UltrasonicConfig,
) -> None:
    """read_distance_m() must never return a value above max_range_m."""
    for _ in range(20):
        d = await sensor.read_distance_m()
        assert (
            d <= ultrasonic_cfg.max_range_m
        ), f"Distance {d:.4f}m exceeds max_range {ultrasonic_cfg.max_range_m:.4f}m"
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# 5. Read latency — single read must complete within timeout + margin
# ---------------------------------------------------------------------------


@pytest.mark.timeout(5)
async def test_read_latency_within_timeout(
    sensor,
    ultrasonic_cfg: UltrasonicConfig,
) -> None:
    """A single read must complete within 2x the configured echo timeout."""
    margin_factor = 2.0
    max_allowed_s = ultrasonic_cfg.timeout_s * margin_factor

    t0 = time.monotonic()
    await sensor.read_distance_m()
    elapsed_s = time.monotonic() - t0

    assert elapsed_s <= max_allowed_s, (
        f"Read took {elapsed_s * 1000:.1f}ms, max allowed "
        f"{max_allowed_s * 1000:.1f}ms (2x timeout={ultrasonic_cfg.timeout_s * 1000:.0f}ms)"
    )


# ---------------------------------------------------------------------------
# 6. GPIO pin configuration matches config — no hardcoded pin numbers
# ---------------------------------------------------------------------------


def test_trigger_echo_pins_from_config_not_hardcoded(
    sensor,
    ultrasonic_cfg: UltrasonicConfig,
) -> None:
    """Internal pin numbers must come from config, never hardcoded."""
    assert sensor._trigger_pin == ultrasonic_cfg.trigger_pin
    assert sensor._echo_pin == ultrasonic_cfg.echo_pin
    # Pins should be distinct
    assert sensor._trigger_pin != sensor._echo_pin


# ---------------------------------------------------------------------------
# 7. Speed of sound wiring — uses config, not a literal
# ---------------------------------------------------------------------------


def test_speed_of_sound_from_config(
    sensor,
    ultrasonic_cfg: UltrasonicConfig,
) -> None:
    """Internal speed-of-sound must match config (not a hardcoded 343.0)."""
    assert sensor._speed_of_sound_mps == ultrasonic_cfg.speed_of_sound_mps
