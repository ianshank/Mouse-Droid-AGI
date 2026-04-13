"""Resilient ESP32 driver — wraps any ESP32CommProtocol with retry + circuit breaker.

Drop-in replacement: implements ``ESP32CommProtocol`` so the orchestrator
and factory don't need to know about the wrapper.  Safety-critical calls
(``emergency_stop``) bypass the circuit breaker entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mousedroid.logging.setup import get_logger
from mousedroid.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
from mousedroid.resilience.retry import retry_async

if TYPE_CHECKING:
    from mousedroid.comms.protocol import EncoderReading, ESP32CommProtocol
    from mousedroid.config.schema import CircuitBreakerConfig, RetryConfig

_log = get_logger(__name__)


class ResilientESP32Driver:
    """ESP32 driver wrapper with circuit breaker and retry.

    Implements ``ESP32CommProtocol`` — transparent to the orchestrator.

    Args:
        inner: The underlying ESP32 driver to wrap.
        retry_cfg: Retry timing configuration.
        cb_cfg: Circuit breaker thresholds.
    """

    def __init__(
        self,
        inner: ESP32CommProtocol,
        retry_cfg: RetryConfig,
        cb_cfg: CircuitBreakerConfig,
    ) -> None:
        self._inner = inner
        self._retry_cfg = retry_cfg
        self._cb_command = CircuitBreaker("esp32_command", cb_cfg)
        self._cb_query = CircuitBreaker("esp32_query", cb_cfg)
        self._total_calls: int = 0
        self._total_failures: int = 0

    # -- ESP32CommProtocol -------------------------------------------------

    async def connect(self) -> None:
        """Connect with retry (no circuit breaker — must succeed to start)."""
        await retry_async(
            self._inner.connect,
            cfg=self._retry_cfg,
            retryable_exceptions=(Exception,),
        )
        _log.info("resilient_driver_connected")

    async def disconnect(self) -> None:
        """Disconnect — best-effort, no retry needed."""
        try:
            await self._inner.disconnect()
        except Exception:
            _log.warning("resilient_driver_disconnect_error", exc_info=True)

    async def send_velocity(self, vx: float, vy: float, omega: float) -> None:
        """Send velocity command with circuit breaker + retry."""
        self._total_calls += 1
        try:
            await self._cb_command.call(
                retry_async,
                self._inner.send_velocity,
                vx,
                vy,
                omega,
                cfg=self._retry_cfg,
                retryable_exceptions=(Exception,),
            )
        except CircuitOpenError:
            self._total_failures += 1
            _log.warning(
                "resilient_driver_velocity_rejected",
                circuit_state=self._cb_command.state.value,
            )
            raise
        except Exception:
            self._total_failures += 1
            _log.warning("resilient_driver_velocity_error", exc_info=True)
            raise

    async def read_encoders(self) -> EncoderReading:
        """Read encoders with circuit breaker + retry.

        Returns:
            Encoder reading from the inner driver.
        """
        self._total_calls += 1
        try:
            return await self._cb_query.call(
                retry_async,
                self._inner.read_encoders,
                cfg=self._retry_cfg,
                retryable_exceptions=(Exception,),
            )
        except CircuitOpenError:
            self._total_failures += 1
            _log.warning(
                "resilient_driver_encoders_rejected",
                circuit_state=self._cb_query.state.value,
            )
            raise
        except Exception:
            self._total_failures += 1
            _log.warning("resilient_driver_encoder_error", exc_info=True)
            raise

    async def get_battery_voltage(self) -> float:
        """Read battery voltage with circuit breaker + retry.

        Returns:
            Battery voltage in volts.
        """
        self._total_calls += 1
        try:
            return await self._cb_query.call(
                retry_async,
                self._inner.get_battery_voltage,
                cfg=self._retry_cfg,
                retryable_exceptions=(Exception,),
            )
        except CircuitOpenError:
            self._total_failures += 1
            _log.warning(
                "resilient_driver_battery_rejected",
                circuit_state=self._cb_query.state.value,
            )
            raise
        except Exception:
            self._total_failures += 1
            _log.warning("resilient_driver_battery_error", exc_info=True)
            raise

    async def emergency_stop(self) -> None:
        """Emergency stop — bypasses circuit breaker entirely.

        Safety-critical: always attempts to reach the ESP32 directly.
        """
        try:
            await self._inner.emergency_stop()
        except Exception:
            _log.error("resilient_driver_emergency_stop_failed", exc_info=True)
            raise

    # -- Introspection -----------------------------------------------------

    @property
    def command_circuit_state(self) -> CircuitState:
        """Current state of the command circuit breaker."""
        return self._cb_command.state

    @property
    def query_circuit_state(self) -> CircuitState:
        """Current state of the query circuit breaker."""
        return self._cb_query.state

    @property
    def inner(self) -> ESP32CommProtocol:
        """The wrapped inner driver."""
        return self._inner

    @property
    def stats(self) -> dict[str, Any]:
        """Resilience statistics for health reporting.

        Returns:
            Dict with call counts, failure counts, and circuit states.
        """
        return {
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "command_circuit": self._cb_command.state.value,
            "query_circuit": self._cb_query.state.value,
            "command_failures": self._cb_command.failure_count,
            "query_failures": self._cb_query.failure_count,
        }

    def reset(self) -> None:
        """Reset both circuit breakers to closed state."""
        self._cb_command.reset()
        self._cb_query.reset()
        _log.info("resilient_driver_reset")
