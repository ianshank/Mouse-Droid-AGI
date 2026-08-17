"""Resilient camera driver — wraps any VisionProtocol with circuit breaker + retry.

Transparent to consumers: implements :class:`~mousedroid.hardware.protocols.VisionProtocol`.
Follows the :class:`~mousedroid.resilience.resilient_lidar.ResilientLidarDriver` pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from mousedroid.logging.setup import get_logger
from mousedroid.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError
from mousedroid.resilience.retry import retry_async

if TYPE_CHECKING:
    from typing import Protocol

    import numpy as np
    from numpy.typing import NDArray

    from mousedroid.config.schema import CircuitBreakerConfig, RetryConfig
    from mousedroid.hardware.protocols import RawFrameSourceProtocol, VisionProtocol

    class _RawFrameCaptureProtocol(Protocol):
        """File-local protocol for the ``capture_raw_frame`` duck-type.

        Not promoted to ``hardware/protocols.py`` — only this wrapper and
        ``validation/runtime/_camera.py``'s diagnostic capture path care
        about this convention; ``VisionProtocol``/``RawFrameSourceProtocol``
        remain the only public camera-driver contracts.
        """

        async def capture_raw_frame(self) -> NDArray[np.uint8]: ...


_log = get_logger(__name__)


class ResilientCamera:
    """Camera driver wrapper with circuit breaker and retry.

    Implements :class:`~mousedroid.hardware.protocols.VisionProtocol`
    transparently — the orchestrator and telemetry server are unaware
    of the resilience layer. When the inner driver additionally
    implements :class:`~mousedroid.hardware.protocols.RawFrameSourceProtocol`
    (e.g. ``MockCamera``, on-NPU drivers that can tee the pre-inference
    frame), that optional capability is delegated through unwrapped so
    ``isinstance(camera, RawFrameSourceProtocol)`` duck-typing at the
    telemetry-server factory seam keeps working. The same holds for
    ``capture_raw_frame`` — a non-Protocol convention that
    ``JetsonCSICamera``/``IMX500Camera`` both implement and that
    ``validation/runtime/_camera.py``'s diagnostic capture path resolves
    via duck-typing — so wrapping never breaks the on-device camera probe.

    Args:
        inner: Underlying camera driver.
        retry_cfg: Retry policy configuration.
        cb_cfg: Circuit breaker configuration.
    """

    def __init__(
        self,
        inner: VisionProtocol,
        retry_cfg: RetryConfig,
        cb_cfg: CircuitBreakerConfig,
    ) -> None:
        self._inner = inner
        self._retry_cfg = retry_cfg
        self._cb = CircuitBreaker("camera", cb_cfg)
        # Preserve optional, non-Protocol capabilities: only bind a proxy on
        # self when the wrapped driver actually has it, so a plain
        # VisionProtocol-only driver doesn't spuriously start passing
        # isinstance(camera, RawFrameSourceProtocol) checks or gain a
        # capture_raw_frame attribute it never had.
        raw_capture = getattr(inner, "capture_raw_jpeg", None)
        if callable(raw_capture):
            self.capture_raw_jpeg = self._capture_raw_jpeg
        raw_frame = getattr(inner, "capture_raw_frame", None)
        if callable(raw_frame):
            self.capture_raw_frame = self._capture_raw_frame

    # -- VisionProtocol properties -----------------------------------------

    @property
    def feature_dim(self) -> int:
        """Output feature vector dimension."""
        return self._inner.feature_dim

    # -- Lifecycle --------------------------------------------------------

    async def start(self) -> None:
        """Start the underlying camera pipeline with retry."""
        await retry_async(
            self._inner.start,
            cfg=self._retry_cfg,
            retryable_exceptions=(Exception,),
        )

    async def stop(self) -> None:
        """Stop the underlying camera pipeline (best-effort, no retry)."""
        try:
            await self._inner.stop()
        except Exception:
            _log.warning("resilient_camera_stop_error", exc_info=True)

    # -- Data acquisition -------------------------------------------------

    async def capture_features(self) -> NDArray[np.float32]:
        """Capture a feature vector through the circuit breaker with retry.

        Raises:
            CircuitOpenError: If the circuit breaker is open.
        """
        try:
            return await self._cb.call(
                retry_async,
                self._inner.capture_features,
                cfg=self._retry_cfg,
                retryable_exceptions=(Exception,),
            )
        except CircuitOpenError:
            _log.warning(
                "camera_capture_rejected",
                circuit_state=self._cb.state.name,
            )
            raise

    async def _capture_raw_jpeg(self) -> bytes | None:
        """Delegate the optional raw-JPEG capture (no circuit breaker).

        Feeds the dashboard MJPEG/snapshot path only — never the hot 30 Hz
        sense loop — so it degrades to ``None`` on failure rather than
        tripping the same breaker that guards ``capture_features``.
        """
        raw = cast("RawFrameSourceProtocol", self._inner)
        try:
            result: bytes | None = await raw.capture_raw_jpeg()
        except Exception:
            _log.warning("resilient_camera_raw_jpeg_error", exc_info=True)
            return None
        return result

    async def _capture_raw_frame(self) -> NDArray[np.uint8]:
        """Delegate the optional raw-frame capture (no circuit breaker).

        Feeds ``validation/runtime/_camera.py``'s diagnostic capture path
        only — never the hot 30 Hz sense loop — so failures propagate
        directly rather than being swallowed, matching how the diagnostic
        tool called the unwrapped driver before this wrapper existed.
        """
        raw = cast("_RawFrameCaptureProtocol", self._inner)
        result: NDArray[np.uint8] = await raw.capture_raw_frame()
        return result

    # -- Introspection ----------------------------------------------------

    @property
    def inner(self) -> VisionProtocol:
        """The wrapped inner camera driver."""
        return self._inner

    @property
    def circuit_state(self) -> str:
        """Current circuit breaker state name."""
        return self._cb.state.name

    def reset(self) -> None:
        """Reset the circuit breaker to CLOSED state."""
        self._cb.reset()
