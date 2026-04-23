"""Tests for ResilientLidarDriver."""

from __future__ import annotations

import pytest

from mousedroid.config.schema import CircuitBreakerConfig, LidarConfig, RetryConfig
from mousedroid.hardware.lidar.ld19_driver import LD19ReadStats
from mousedroid.hardware.lidar.mock_lidar import MockLidar
from mousedroid.resilience.circuit_breaker import CircuitOpenError
from mousedroid.resilience.resilient_lidar import ResilientLidarDriver
from mousedroid.resilience.retry import RetryExhaustedError
from mousedroid.sensing.lidar_scan import LidarScan


@pytest.fixture
def lidar_cfg() -> LidarConfig:
    """LidarConfig for testing."""
    return LidarConfig(
        enabled=True,
        serial_port="/dev/ttyUSB1",
        baud_rate=230400,
        max_range_m=12.0,
        min_range_m=0.15,
        scan_frequency_hz=10.0,
        min_confidence=0,
        read_timeout_s=0.2,
        n_sectors=36,
        feature_dim=36,
    )


@pytest.fixture
def retry_cfg() -> RetryConfig:
    """Fast retry config for tests."""
    return RetryConfig(
        max_attempts=1,
        base_delay_s=0.01,
        max_delay_s=0.1,
        exponential_base=2.0,
    )


@pytest.fixture
def cb_cfg() -> CircuitBreakerConfig:
    """Low-threshold circuit breaker for tests."""
    return CircuitBreakerConfig(
        failure_threshold=2,
        recovery_timeout_s=60.0,
        half_open_max_calls=1,
    )


@pytest.fixture
def mock_inner(lidar_cfg: LidarConfig) -> MockLidar:
    """MockLidar inner driver."""
    return MockLidar(lidar_cfg)


@pytest.fixture
def resilient(
    mock_inner: MockLidar,
    retry_cfg: RetryConfig,
    cb_cfg: CircuitBreakerConfig,
) -> ResilientLidarDriver:
    """ResilientLidarDriver wrapping a MockLidar."""
    return ResilientLidarDriver(mock_inner, retry_cfg, cb_cfg)


def test_construct(resilient: ResilientLidarDriver) -> None:
    """ResilientLidarDriver can be constructed."""
    assert resilient is not None


def test_max_range_m_delegates(resilient: ResilientLidarDriver) -> None:
    """max_range_m delegates to inner driver."""
    assert resilient.max_range_m == 12.0


def test_min_range_m_delegates(resilient: ResilientLidarDriver) -> None:
    """min_range_m delegates to inner driver."""
    assert resilient.min_range_m == 0.15


def test_scan_frequency_hz_delegates(resilient: ResilientLidarDriver) -> None:
    """scan_frequency_hz delegates to inner driver."""
    assert resilient.scan_frequency_hz == 10.0


def test_circuit_state_initial(resilient: ResilientLidarDriver) -> None:
    """Initial circuit state is CLOSED."""
    assert resilient.circuit_state == "CLOSED"


async def test_read_scan_delegates(resilient: ResilientLidarDriver) -> None:
    """read_scan returns data from the inner driver."""
    scan = await resilient.read_scan()
    assert scan.n_points == 360


async def test_read_scan_with_diagnostics_delegates(
    retry_cfg: RetryConfig,
    cb_cfg: CircuitBreakerConfig,
    lidar_cfg: LidarConfig,
) -> None:
    """Diagnostic reads should pass through the resilience wrapper."""

    class DiagnosticLidar(MockLidar):
        async def read_scan_with_diagnostics(self) -> tuple[LidarScan, LD19ReadStats]:
            return await self.read_scan(), LD19ReadStats(
                bytes_read=188,
                chunks_read=1,
                frames_parsed=3,
                covered_angle_deg=270.0,
                elapsed_s=0.05,
            )

    inner = DiagnosticLidar(lidar_cfg)
    driver = ResilientLidarDriver(inner, retry_cfg, cb_cfg)

    scan, stats = await driver.read_scan_with_diagnostics()

    assert scan.n_points == 360
    assert stats.bytes_read == 188
    assert stats.frames_parsed == 3
    assert stats.covered_angle_deg == pytest.approx(270.0)


async def test_start_delegates(
    resilient: ResilientLidarDriver,
    mock_inner: MockLidar,
) -> None:
    """start() delegates to inner driver."""
    await resilient.start()
    assert mock_inner.started


async def test_stop_best_effort(
    resilient: ResilientLidarDriver,
    mock_inner: MockLidar,
) -> None:
    """stop() delegates to inner driver (best-effort, no exceptions)."""
    await mock_inner.start()
    await resilient.stop()
    assert not mock_inner.started


async def test_circuit_opens_after_failures(
    retry_cfg: RetryConfig,
    cb_cfg: CircuitBreakerConfig,
    lidar_cfg: LidarConfig,
) -> None:
    """Circuit breaker opens after exceeding failure threshold."""

    class FailingLidar(MockLidar):
        """MockLidar that always fails on read_scan."""

        async def read_scan(self) -> LidarScan:
            raise RuntimeError("sensor failure")

    inner = FailingLidar(lidar_cfg)
    driver = ResilientLidarDriver(inner, retry_cfg, cb_cfg)

    # Fail enough times to open the circuit (threshold=2).
    # The retry wrapper converts RuntimeError to RetryExhaustedError,
    # so we catch the broad Exception base.
    for _ in range(cb_cfg.failure_threshold):
        with pytest.raises(RetryExhaustedError):
            await driver.read_scan()

    # Now the circuit should be open.
    with pytest.raises(CircuitOpenError):
        await driver.read_scan()


async def test_reset_closes_open_circuit(
    retry_cfg: RetryConfig,
    cb_cfg: CircuitBreakerConfig,
    lidar_cfg: LidarConfig,
) -> None:
    """reset() returns an opened circuit to CLOSED state."""

    class FailingLidar(MockLidar):
        """MockLidar that always fails on read_scan."""

        async def read_scan(self) -> LidarScan:
            raise RuntimeError("sensor failure")

    inner = FailingLidar(lidar_cfg)
    driver = ResilientLidarDriver(inner, retry_cfg, cb_cfg)

    # Open the circuit.
    for _ in range(cb_cfg.failure_threshold):
        with pytest.raises(RetryExhaustedError):
            await driver.read_scan()
    assert driver.circuit_state == "OPEN"

    driver.reset()
    assert driver.circuit_state == "CLOSED"


async def test_stop_best_effort_on_failure(
    retry_cfg: RetryConfig,
    cb_cfg: CircuitBreakerConfig,
    lidar_cfg: LidarConfig,
) -> None:
    """stop() does not raise even when inner driver's stop fails."""

    class FailingStopLidar(MockLidar):
        """MockLidar whose stop() raises."""

        async def stop(self) -> None:
            raise RuntimeError("stop failed")

    inner = FailingStopLidar(lidar_cfg)
    driver = ResilientLidarDriver(inner, retry_cfg, cb_cfg)
    # stop() should not propagate the exception.
    await driver.stop()
