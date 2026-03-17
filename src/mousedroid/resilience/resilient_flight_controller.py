"""Resilient flight controller — wraps FlightControllerProtocol with retry + circuit breaker.

Drop-in replacement: implements ``FlightControllerProtocol`` so the factory
and adapters don't need to know about the wrapper.  Safety-critical calls
(``emergency_stop``) bypass the circuit breaker entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger
from mousedroid.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
from mousedroid.resilience.retry import retry_async

if TYPE_CHECKING:
    from mousedroid.comms.flight_protocol import FlightControllerProtocol
    from mousedroid.config.schema import CircuitBreakerConfig, RetryConfig

_log = get_logger(__name__)


class ResilientFlightController:
    """Flight controller wrapper with circuit breaker and retry.

    Implements ``FlightControllerProtocol`` — transparent to adapters.

    Args:
        inner: The underlying flight controller driver to wrap.
        retry_cfg: Retry timing configuration.
        cb_cfg: Circuit breaker thresholds.
    """

    def __init__(
        self,
        inner: FlightControllerProtocol,
        retry_cfg: RetryConfig,
        cb_cfg: CircuitBreakerConfig,
    ) -> None:
        self._inner = inner
        self._retry_cfg = retry_cfg
        self._cb_command = CircuitBreaker("fc_command", cb_cfg)
        self._cb_query = CircuitBreaker("fc_query", cb_cfg)
        self._total_calls: int = 0
        self._total_failures: int = 0

    # -- Lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        """Connect with retry (no circuit breaker — must succeed to start)."""
        await retry_async(
            self._inner.connect,
            cfg=self._retry_cfg,
            retryable_exceptions=(Exception,),
        )
        _log.info("resilient_fc_connected")

    async def disconnect(self) -> None:
        """Disconnect — best-effort, no retry needed."""
        try:
            await self._inner.disconnect()
        except Exception:
            _log.warning("resilient_fc_disconnect_error", exc_info=True)

    # -- Arming ------------------------------------------------------------

    async def arm(self) -> None:
        """Arm motors with circuit breaker + retry."""
        self._total_calls += 1
        try:
            await self._cb_command.call(
                retry_async,
                self._inner.arm,
                cfg=self._retry_cfg,
                retryable_exceptions=(Exception,),
            )
        except CircuitOpenError:
            self._total_failures += 1
            _log.warning("resilient_fc_arm_rejected", circuit_state=self._cb_command.state.value)
            raise
        except Exception:
            self._total_failures += 1
            raise

    async def disarm(self) -> None:
        """Disarm motors with circuit breaker + retry."""
        self._total_calls += 1
        try:
            await self._cb_command.call(
                retry_async,
                self._inner.disarm,
                cfg=self._retry_cfg,
                retryable_exceptions=(Exception,),
            )
        except CircuitOpenError:
            self._total_failures += 1
            _log.warning("resilient_fc_disarm_rejected", circuit_state=self._cb_command.state.value)
            raise
        except Exception:
            self._total_failures += 1
            raise

    # -- Flight commands ---------------------------------------------------

    async def takeoff(self, altitude_m: float) -> None:
        """Takeoff with circuit breaker + retry.

        Args:
            altitude_m: Target altitude AGL in metres.
        """
        self._total_calls += 1
        try:
            await self._cb_command.call(
                retry_async,
                self._inner.takeoff,
                altitude_m,
                cfg=self._retry_cfg,
                retryable_exceptions=(Exception,),
            )
        except CircuitOpenError:
            self._total_failures += 1
            _log.warning(
                "resilient_fc_takeoff_rejected", circuit_state=self._cb_command.state.value
            )
            raise
        except Exception:
            self._total_failures += 1
            raise

    async def land(self) -> None:
        """Land with circuit breaker + retry."""
        self._total_calls += 1
        try:
            await self._cb_command.call(
                retry_async,
                self._inner.land,
                cfg=self._retry_cfg,
                retryable_exceptions=(Exception,),
            )
        except CircuitOpenError:
            self._total_failures += 1
            _log.warning("resilient_fc_land_rejected", circuit_state=self._cb_command.state.value)
            raise
        except Exception:
            self._total_failures += 1
            raise

    async def send_velocity_ned(
        self, vn: float, ve: float, vd: float, yaw_rate: float
    ) -> None:
        """Send velocity with circuit breaker + retry."""
        self._total_calls += 1
        try:
            await self._cb_command.call(
                retry_async,
                self._inner.send_velocity_ned,
                vn,
                ve,
                vd,
                yaw_rate,
                cfg=self._retry_cfg,
                retryable_exceptions=(Exception,),
            )
        except CircuitOpenError:
            self._total_failures += 1
            _log.warning(
                "resilient_fc_velocity_rejected", circuit_state=self._cb_command.state.value
            )
            raise
        except Exception:
            self._total_failures += 1
            raise

    # -- Telemetry ---------------------------------------------------------

    async def get_altitude_m(self) -> float:
        """Read altitude with circuit breaker + retry."""
        self._total_calls += 1
        try:
            return await self._cb_query.call(
                retry_async,
                self._inner.get_altitude_m,
                cfg=self._retry_cfg,
                retryable_exceptions=(Exception,),
            )
        except CircuitOpenError:
            self._total_failures += 1
            _log.warning(
                "resilient_fc_altitude_rejected", circuit_state=self._cb_query.state.value
            )
            raise
        except Exception:
            self._total_failures += 1
            raise

    async def get_gps_position(self) -> tuple[float, float, float]:
        """Read GPS with circuit breaker + retry."""
        self._total_calls += 1
        try:
            return await self._cb_query.call(
                retry_async,
                self._inner.get_gps_position,
                cfg=self._retry_cfg,
                retryable_exceptions=(Exception,),
            )
        except CircuitOpenError:
            self._total_failures += 1
            _log.warning("resilient_fc_gps_rejected", circuit_state=self._cb_query.state.value)
            raise
        except Exception:
            self._total_failures += 1
            raise

    async def get_imu_data(self) -> NDArray[np.float32]:
        """Read IMU with circuit breaker + retry."""
        self._total_calls += 1
        try:
            return await self._cb_query.call(
                retry_async,
                self._inner.get_imu_data,
                cfg=self._retry_cfg,
                retryable_exceptions=(Exception,),
            )
        except CircuitOpenError:
            self._total_failures += 1
            _log.warning("resilient_fc_imu_rejected", circuit_state=self._cb_query.state.value)
            raise
        except Exception:
            self._total_failures += 1
            raise

    async def get_battery_voltage(self) -> float:
        """Read battery voltage with circuit breaker + retry."""
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
                "resilient_fc_battery_rejected", circuit_state=self._cb_query.state.value
            )
            raise
        except Exception:
            self._total_failures += 1
            raise

    async def set_flight_mode(self, mode: str) -> None:
        """Set flight mode with circuit breaker + retry."""
        self._total_calls += 1
        try:
            await self._cb_command.call(
                retry_async,
                self._inner.set_flight_mode,
                mode,
                cfg=self._retry_cfg,
                retryable_exceptions=(Exception,),
            )
        except CircuitOpenError:
            self._total_failures += 1
            _log.warning("resilient_fc_mode_rejected", circuit_state=self._cb_command.state.value)
            raise
        except Exception:
            self._total_failures += 1
            raise

    async def return_to_launch(self) -> None:
        """RTL with circuit breaker + retry."""
        self._total_calls += 1
        try:
            await self._cb_command.call(
                retry_async,
                self._inner.return_to_launch,
                cfg=self._retry_cfg,
                retryable_exceptions=(Exception,),
            )
        except CircuitOpenError:
            self._total_failures += 1
            _log.warning("resilient_fc_rtl_rejected", circuit_state=self._cb_command.state.value)
            raise
        except Exception:
            self._total_failures += 1
            raise

    async def emergency_stop(self) -> None:
        """Emergency stop — bypasses circuit breaker entirely.

        Safety-critical: always attempts to reach the FC directly.
        """
        try:
            await self._inner.emergency_stop()
        except Exception:
            _log.error("resilient_fc_emergency_stop_failed", exc_info=True)
            raise

    # -- Properties --------------------------------------------------------

    @property
    def armed(self) -> bool:
        """Whether the motors are currently armed."""
        return self._inner.armed

    @property
    def flight_mode(self) -> str:
        """Current flight mode."""
        return self._inner.flight_mode

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
    def inner(self) -> FlightControllerProtocol:
        """The wrapped inner driver."""
        return self._inner

    @property
    def stats(self) -> dict[str, Any]:
        """Resilience statistics for health reporting."""
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
        _log.info("resilient_fc_reset")
