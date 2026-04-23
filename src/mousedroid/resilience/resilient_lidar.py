"""Resilient LiDAR driver — wraps any LidarProtocol with circuit breaker + retry.

Transparent to consumers: implements :class:`~mousedroid.hardware.protocols.LidarProtocol`.
Follows the :class:`~mousedroid.resilience.resilient_driver.ResilientESP32Driver` pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger
from mousedroid.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError
from mousedroid.resilience.retry import retry_async

if TYPE_CHECKING:
    from mousedroid.config.schema import CircuitBreakerConfig, RetryConfig
    from mousedroid.hardware.lidar.ld19_driver import LD19ReadStats
    from mousedroid.hardware.protocols import LidarProtocol
    from mousedroid.sensing.lidar_scan import LidarScan

_log = get_logger(__name__)


class ResilientLidarDriver:
    """LiDAR driver wrapper with circuit breaker and retry.

    Implements :class:`~mousedroid.hardware.protocols.LidarProtocol`
    transparently — the orchestrator and sensor manager are unaware
    of the resilience layer.

    Args:
        inner: Underlying LiDAR driver.
        retry_cfg: Retry policy configuration.
        cb_cfg: Circuit breaker configuration.
    """

    def __init__(
        self,
        inner: LidarProtocol,
        retry_cfg: RetryConfig,
        cb_cfg: CircuitBreakerConfig,
    ) -> None:
        self._inner = inner
        self._retry_cfg = retry_cfg
        self._cb = CircuitBreaker("lidar", cb_cfg)

    # -- LidarProtocol properties -----------------------------------------

    @property
    def max_range_m(self) -> float:
        """Maximum detection range in metres."""
        return self._inner.max_range_m

    @property
    def min_range_m(self) -> float:
        """Minimum detection range in metres."""
        return self._inner.min_range_m

    @property
    def scan_frequency_hz(self) -> float:
        """Nominal scan rotation frequency in Hz."""
        return self._inner.scan_frequency_hz

    # -- Lifecycle --------------------------------------------------------

    async def start(self) -> None:
        """Start the underlying LiDAR driver with retry."""
        await retry_async(
            self._inner.start,
            cfg=self._retry_cfg,
            retryable_exceptions=(Exception,),
        )

    async def stop(self) -> None:
        """Stop the underlying LiDAR driver (best-effort, no retry)."""
        try:
            await self._inner.stop()
        except Exception:
            _log.warning("resilient_lidar_stop_error", exc_info=True)

    # -- Data acquisition -------------------------------------------------

    async def read_scan(self) -> LidarScan:
        """Read a scan through the circuit breaker with retry.

        Raises:
            CircuitOpenError: If the circuit breaker is open.
        """
        try:
            return await self._cb.call(
                retry_async,
                self._inner.read_scan,
                cfg=self._retry_cfg,
                retryable_exceptions=(Exception,),
            )
        except CircuitOpenError:
            _log.warning(
                "lidar_scan_rejected",
                circuit_state=self._cb.state.name,
            )
            raise

    async def read_scan_with_diagnostics(self) -> tuple[LidarScan, LD19ReadStats]:
        """Read a scan with low-level diagnostics when the inner driver supports it."""
        from mousedroid.hardware.lidar.ld19_driver import LD19ReadStats

        read_with_diagnostics = getattr(self._inner, "read_scan_with_diagnostics", None)
        if not callable(read_with_diagnostics):
            return await self.read_scan(), LD19ReadStats()

        try:
            return await self._cb.call(
                retry_async,
                read_with_diagnostics,
                cfg=self._retry_cfg,
                retryable_exceptions=(Exception,),
            )
        except CircuitOpenError:
            _log.warning(
                "lidar_scan_rejected",
                circuit_state=self._cb.state.name,
            )
            raise

    # -- Introspection ----------------------------------------------------

    @property
    def circuit_state(self) -> str:
        """Current circuit breaker state name."""
        return self._cb.state.name

    def reset(self) -> None:
        """Reset the circuit breaker to CLOSED state."""
        self._cb.reset()
