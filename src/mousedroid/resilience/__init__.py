"""Resilience primitives — circuit breaker, retry, and resilient wrappers.

Provides fault-tolerance building blocks that wrap existing protocol
implementations without modifying them.  All thresholds come from
``mousedroid.config.schema`` — nothing is hardcoded.
"""

from mousedroid.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)
from mousedroid.resilience.resilient_driver import ResilientESP32Driver
from mousedroid.resilience.retry import (
    RetryExhaustedError,
    retry_async,
    with_retry,
)

__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "ResilientESP32Driver",
    "RetryExhaustedError",
    "retry_async",
    "with_retry",
]
