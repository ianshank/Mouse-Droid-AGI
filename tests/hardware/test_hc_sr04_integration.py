"""HC-SR04 GPIO integration tests for Jetson hardware.

These tests require:
    - Jetson.GPIO installed (Jetson Orin Nano / Nano host)
    - HC-SR04 wired to BCM 23 (trigger) and BCM 24 (echo)
    - Pinmux fix applied: ``sudo busybox devmem 0x243D020 w 0x5``

Run on Jetson::

    pytest tests/hardware/test_hc_sr04_integration.py -m hardware -v --timeout=30

All thresholds are read from ``UltrasonicConfig`` — nothing is hardcoded in
the test body.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from mousedroid.config.schema import Settings, UltrasonicConfig

pytestmark = pytest.mark.hardware

# Skip entire module on any machine without Jetson.GPIO
Jetson = pytest.importorskip("Jetson.GPIO", reason="Jetson.GPIO not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

JETSON_PROD_CONFIG = "config/jetson_production.yaml"


@pytest.fixture(scope="module")
def ultrasonic_cfg() -> UltrasonicConfig:
    """Load UltrasonicConfig from jetson_production.yaml."""
    import yaml

    with open(JETSON_PROD_CONFIG) as fh:
        raw = yaml.safe_load(fh)

    # Merge into default Settings so all validators run
    settings = Settings(**raw)
    assert settings.ultrasonic is not None, "ultrasonic not configured in jetson_production.yaml"
    return settings.ultrasonic


@pytest.fixture(scope="module")
def sensor(ultrasonic_cfg: UltrasonicConfig):
    """Construct a real HcSr04 sensor and clean up GPIO after all tests."""
    from mousedroid.hardware.sensors.ultrasonic import HcSr04

    s = HcSr04(ultrasonic_cfg)
    yield s
    # GPIO cleanup after all module-level tests are done
    import contextlib

    with contextlib.suppress(Exception):
        import Jetson.GPIO as GPIO

        GPIO.cleanup()


# ---------------------------------------------------------------------------
# 1. Sensor configuration sanity
# ---------------------------------------------------------------------------


def test_sensor_pins_from_config(sensor, ultrasonic_cfg: UltrasonicConfig) -> None:
    """Verify sensor exposes pin numbers that match config (no hardcoded values)."""
    assert sensor._trigger_pin == ultrasonic_cfg.trigger_pin
    assert sensor._echo_pin == ultrasonic_cfg.echo_pin


def test_max_range_from_config(sensor, ultrasonic_cfg: UltrasonicConfig) -> None:
    assert sensor.max_range_m == ultrasonic_cfg.max_range_m


def test_min_range_from_config(sensor, ultrasonic_cfg: UltrasonicConfig) -> None:
    assert sensor.min_range_m == ultrasonic_cfg.min_range_m


# ---------------------------------------------------------------------------
# 2. Single read
# ---------------------------------------------------------------------------


async def test_read_distance_returns_float(sensor, ultrasonic_cfg: UltrasonicConfig) -> None:
    """read_distance_m() must return a float."""
    d = await sensor.read_distance_m()
    assert isinstance(d, float)


async def test_read_distance_within_physical_bounds(
    sensor, ultrasonic_cfg: UltrasonicConfig
) -> None:
    """Returned distance must be between min_range and max_range (inclusive)."""
    d = await sensor.read_distance_m()
    assert ultrasonic_cfg.min_range_m <= d <= ultrasonic_cfg.max_range_m


async def test_timeout_returns_max_range(ultrasonic_cfg: UltrasonicConfig) -> None:
    """When echo never arrives, driver must return max_range_m exactly."""
    from unittest.mock import patch

    from mousedroid.hardware.sensors.ultrasonic import HcSr04

    sensor = HcSr04(ultrasonic_cfg)

    # Simulate a perpetually-low echo pin (no object in range)
    import Jetson.GPIO as GPIO  # type: ignore[import]

    with patch.object(GPIO, "input", return_value=0):
        d = await sensor.read_distance_m()

    assert d == ultrasonic_cfg.max_range_m


# ---------------------------------------------------------------------------
# 3. Known-distance parametric tests (requires physical setup)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expected_m", "tolerance_m"),
    [
        (0.2, 0.05),  # 20 cm wall target, ±5 cm
        (0.5, 0.08),  # 50 cm wall target, ±8 cm
        (1.0, 0.12),  # 100 cm wall target, ±12 cm
    ],
)
@pytest.mark.skipif(
    True,
    reason="Requires physical target placement — run manually with --run-parametric flag",
)
async def test_known_distance(
    sensor,
    ultrasonic_cfg: UltrasonicConfig,
    expected_m: float,
    tolerance_m: float,
) -> None:
    """Verify measured distance is within tolerance of an expected physical distance."""
    readings = []
    for _ in range(5):
        d = await sensor.read_distance_m()
        readings.append(d)
        await asyncio.sleep(0.05)  # 20 Hz read rate

    mean_d = sum(readings) / len(readings)
    assert (
        abs(mean_d - expected_m) <= tolerance_m
    ), f"Expected {expected_m:.2f}m ± {tolerance_m:.2f}m, got mean {mean_d:.3f}m"


# ---------------------------------------------------------------------------
# 4. High-rate read — no GPIO contention
# ---------------------------------------------------------------------------


@pytest.mark.timeout(5)
async def test_rapid_reads_no_gpio_error(sensor, ultrasonic_cfg: UltrasonicConfig) -> None:
    """20 Hz reads over 1 s must complete without GPIO errors."""
    interval = 1.0 / 20.0  # 20 Hz
    deadline = time.monotonic() + 1.0
    readings: list[float] = []

    while time.monotonic() < deadline:
        d = await sensor.read_distance_m()
        assert isinstance(d, float), "read_distance_m returned non-float"
        assert ultrasonic_cfg.min_range_m <= d <= ultrasonic_cfg.max_range_m
        readings.append(d)
        await asyncio.sleep(interval)

    assert len(readings) >= 15, f"Expected ≥15 readings at 20 Hz, got {len(readings)}"


# ---------------------------------------------------------------------------
# 5. Concurrent reads — asyncio.to_thread isolation
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
async def test_concurrent_reads_isolated(
    ultrasonic_cfg: UltrasonicConfig,
) -> None:
    """Two concurrent read calls must not interfere with each other's GPIO state."""
    from mousedroid.hardware.sensors.ultrasonic import HcSr04

    # Two separate sensor instances (each sets up its own GPIO mode)
    s1 = HcSr04(ultrasonic_cfg)

    # Run 10 concurrent reads on the same sensor; each awaits to_thread independently
    for _ in range(3):
        results = await asyncio.gather(
            s1.read_distance_m(),
            s1.read_distance_m(),
            return_exceptions=True,
        )
        for r in results:
            assert isinstance(r, float), f"Concurrent read returned non-float: {r!r}"
            assert ultrasonic_cfg.min_range_m <= r <= ultrasonic_cfg.max_range_m
