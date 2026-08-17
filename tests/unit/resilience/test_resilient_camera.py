"""Tests for ResilientCamera."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from structlog.testing import capture_logs

from mousedroid.config.schema import CameraConfig, CircuitBreakerConfig, RetryConfig
from mousedroid.hardware.camera.mock_camera import MockCamera
from mousedroid.hardware.protocols import RawFrameSourceProtocol
from mousedroid.resilience.circuit_breaker import CircuitOpenError
from mousedroid.resilience.resilient_camera import ResilientCamera
from mousedroid.resilience.retry import RetryExhaustedError


@pytest.fixture
def camera_cfg() -> CameraConfig:
    """CameraConfig for testing."""
    return CameraConfig(feature_dim=16)


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
def mock_inner(camera_cfg: CameraConfig) -> MockCamera:
    """MockCamera inner driver (implements RawFrameSourceProtocol too)."""
    return MockCamera(camera_cfg)


@pytest.fixture
def resilient(
    mock_inner: MockCamera,
    retry_cfg: RetryConfig,
    cb_cfg: CircuitBreakerConfig,
) -> ResilientCamera:
    """ResilientCamera wrapping a MockCamera."""
    return ResilientCamera(mock_inner, retry_cfg, cb_cfg)


def test_construct(resilient: ResilientCamera) -> None:
    """ResilientCamera can be constructed."""
    assert resilient is not None


def test_inner_delegates(resilient: ResilientCamera, mock_inner: MockCamera) -> None:
    """inner exposes the wrapped driver, mirroring ResilientESP32Driver.inner."""
    assert resilient.inner is mock_inner


def test_feature_dim_delegates(resilient: ResilientCamera) -> None:
    """feature_dim delegates to inner driver."""
    assert resilient.feature_dim == 16


def test_circuit_state_initial(resilient: ResilientCamera) -> None:
    """Initial circuit state is CLOSED."""
    assert resilient.circuit_state == "CLOSED"


async def test_start_delegates(
    retry_cfg: RetryConfig,
    cb_cfg: CircuitBreakerConfig,
    camera_cfg: CameraConfig,
) -> None:
    """start() actually invokes the inner driver's start(), not a no-op."""

    class StateTrackingCamera(MockCamera):
        """MockCamera that records whether start()/stop() were really called."""

        started = False

        async def start(self) -> None:
            self.started = True
            await super().start()

    inner = StateTrackingCamera(camera_cfg)
    driver = ResilientCamera(inner, retry_cfg, cb_cfg)

    await driver.start()

    assert inner.started


async def test_stop_best_effort(
    retry_cfg: RetryConfig,
    cb_cfg: CircuitBreakerConfig,
    camera_cfg: CameraConfig,
) -> None:
    """stop() actually invokes the inner driver's stop(), not a no-op."""

    class StateTrackingCamera(MockCamera):
        """MockCamera that records whether start()/stop() were really called."""

        stopped = False

        async def stop(self) -> None:
            self.stopped = True
            await super().stop()

    inner = StateTrackingCamera(camera_cfg)
    driver = ResilientCamera(inner, retry_cfg, cb_cfg)

    await driver.stop()

    assert inner.stopped


async def test_capture_features_delegates(resilient: ResilientCamera) -> None:
    """capture_features returns data shaped per the inner driver's feature_dim."""
    features: NDArray[np.float32] = await resilient.capture_features()
    assert features.shape == (16,)


async def test_raw_frame_source_protocol_when_inner_supports_it(
    resilient: ResilientCamera,
) -> None:
    """isinstance(..., RawFrameSourceProtocol) holds when the inner driver has it."""
    assert isinstance(resilient, RawFrameSourceProtocol)


async def test_capture_raw_jpeg_delegates(resilient: ResilientCamera) -> None:
    """capture_raw_jpeg delegates to the inner driver's real JPEG encoder."""
    pytest.importorskip("PIL")
    result = await resilient.capture_raw_jpeg()  # type: ignore[attr-defined]
    assert result is not None
    # JPEG SOI (start-of-image) marker — proves this is real encoded output
    # from the inner driver, not just "some bytes or None".
    assert result.startswith(b"\xff\xd8\xff")


async def test_raw_frame_source_protocol_absent_when_inner_lacks_it(
    retry_cfg: RetryConfig,
    cb_cfg: CircuitBreakerConfig,
    camera_cfg: CameraConfig,
) -> None:
    """A VisionProtocol-only inner driver never gains capture_raw_jpeg."""

    class BareCamera:
        """Minimal VisionProtocol implementation with no raw-frame capability."""

        def __init__(self, cfg: CameraConfig) -> None:
            self._cfg = cfg

        @property
        def feature_dim(self) -> int:
            return self._cfg.feature_dim

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        async def capture_features(self) -> NDArray[np.float32]:
            return np.zeros(self._cfg.feature_dim, dtype=np.float32)

    inner = BareCamera(camera_cfg)
    driver = ResilientCamera(inner, retry_cfg, cb_cfg)

    assert not hasattr(driver, "capture_raw_jpeg")
    assert not hasattr(driver, "capture_raw_frame")
    assert not isinstance(driver, RawFrameSourceProtocol)


async def test_capture_raw_frame_delegates_when_inner_supports_it(
    retry_cfg: RetryConfig,
    cb_cfg: CircuitBreakerConfig,
    camera_cfg: CameraConfig,
) -> None:
    """capture_raw_frame delegates to the inner driver.

    ``JetsonCSICamera``/``IMX500Camera`` both implement this non-Protocol
    convention independently of ``capture_raw_jpeg`` (only JetsonCSICamera
    has that one) — exercise it in isolation.
    """

    class RawFrameOnlyCamera:
        """VisionProtocol driver exposing capture_raw_frame but not capture_raw_jpeg."""

        def __init__(self, cfg: CameraConfig) -> None:
            self._cfg = cfg

        @property
        def feature_dim(self) -> int:
            return self._cfg.feature_dim

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        async def capture_features(self) -> NDArray[np.float32]:
            return np.zeros(self._cfg.feature_dim, dtype=np.float32)

        async def capture_raw_frame(self) -> NDArray[np.uint8]:
            return np.full((4, 4, 3), 9, dtype=np.uint8)

    inner = RawFrameOnlyCamera(camera_cfg)
    driver = ResilientCamera(inner, retry_cfg, cb_cfg)

    assert hasattr(driver, "capture_raw_frame")
    assert not hasattr(driver, "capture_raw_jpeg")

    frame = await driver.capture_raw_frame()  # type: ignore[attr-defined]
    assert frame.shape == (4, 4, 3)
    assert int(frame[0, 0, 0]) == 9


async def test_capture_raw_frame_propagates_inner_failure(
    retry_cfg: RetryConfig,
    cb_cfg: CircuitBreakerConfig,
    camera_cfg: CameraConfig,
) -> None:
    """Unlike capture_raw_jpeg, capture_raw_frame does not swallow errors.

    It feeds a diagnostic tool that needs the real failure, not a silent
    ``None`` degrade.
    """

    class FailingRawFrameCamera(MockCamera):
        """MockCamera whose capture_raw_frame always raises."""

        async def capture_raw_frame(self) -> NDArray[np.uint8]:
            raise RuntimeError("raw frame capture failure")

    inner = FailingRawFrameCamera(camera_cfg)
    driver = ResilientCamera(inner, retry_cfg, cb_cfg)

    with pytest.raises(RuntimeError, match="raw frame capture failure"):
        await driver.capture_raw_frame()  # type: ignore[attr-defined]


async def test_capture_raw_jpeg_degrades_to_none_on_inner_failure(
    retry_cfg: RetryConfig,
    cb_cfg: CircuitBreakerConfig,
    camera_cfg: CameraConfig,
) -> None:
    """capture_raw_jpeg swallows inner errors and returns None (dashboard-only path)."""

    class FailingRawCamera(MockCamera):
        """MockCamera whose raw-jpeg capture always raises."""

        async def capture_raw_jpeg(self) -> bytes | None:
            raise RuntimeError("jpeg encode failure")

    inner = FailingRawCamera(camera_cfg)
    driver = ResilientCamera(inner, retry_cfg, cb_cfg)

    result = await driver.capture_raw_jpeg()  # type: ignore[attr-defined]

    assert result is None
    # The circuit breaker guards capture_features only — a raw-jpeg failure
    # must never trip it.
    assert driver.circuit_state == "CLOSED"


async def test_circuit_opens_after_failures(
    retry_cfg: RetryConfig,
    cb_cfg: CircuitBreakerConfig,
    camera_cfg: CameraConfig,
) -> None:
    """Circuit breaker opens after exceeding failure threshold."""

    class FailingCamera(MockCamera):
        """MockCamera that always fails on capture_features."""

        async def capture_features(self) -> NDArray[np.float32]:
            raise RuntimeError("sensor failure")

    inner = FailingCamera(camera_cfg)
    driver = ResilientCamera(inner, retry_cfg, cb_cfg)

    for _ in range(cb_cfg.failure_threshold):
        with pytest.raises(RetryExhaustedError):
            await driver.capture_features()

    with pytest.raises(CircuitOpenError):
        await driver.capture_features()


async def test_reset_closes_open_circuit(
    retry_cfg: RetryConfig,
    cb_cfg: CircuitBreakerConfig,
    camera_cfg: CameraConfig,
) -> None:
    """reset() returns an opened circuit to CLOSED state."""

    class FailingCamera(MockCamera):
        """MockCamera that always fails on capture_features."""

        async def capture_features(self) -> NDArray[np.float32]:
            raise RuntimeError("sensor failure")

    inner = FailingCamera(camera_cfg)
    driver = ResilientCamera(inner, retry_cfg, cb_cfg)

    for _ in range(cb_cfg.failure_threshold):
        with pytest.raises(RetryExhaustedError):
            await driver.capture_features()
    assert driver.circuit_state == "OPEN"

    driver.reset()
    assert driver.circuit_state == "CLOSED"


async def test_start_retries_and_raises_retry_exhausted(
    retry_cfg: RetryConfig,
    cb_cfg: CircuitBreakerConfig,
    camera_cfg: CameraConfig,
) -> None:
    """start() propagates RetryExhaustedError when the inner driver keeps failing."""

    class FailingStartCamera(MockCamera):
        """MockCamera whose start() always raises."""

        async def start(self) -> None:
            raise RuntimeError("pipeline init failure")

    inner = FailingStartCamera(camera_cfg)
    driver = ResilientCamera(inner, retry_cfg, cb_cfg)

    with pytest.raises(RetryExhaustedError):
        await driver.start()


async def test_stop_best_effort_on_failure(
    retry_cfg: RetryConfig,
    cb_cfg: CircuitBreakerConfig,
    camera_cfg: CameraConfig,
) -> None:
    """stop() does not raise even when inner driver's stop fails, and logs it."""

    class FailingStopCamera(MockCamera):
        """MockCamera whose stop() raises."""

        async def stop(self) -> None:
            raise RuntimeError("stop failed")

    inner = FailingStopCamera(camera_cfg)
    driver = ResilientCamera(inner, retry_cfg, cb_cfg)

    with capture_logs() as logs:
        # stop() should not propagate the exception.
        await driver.stop()

    assert any(entry["event"] == "resilient_camera_stop_error" for entry in logs)
