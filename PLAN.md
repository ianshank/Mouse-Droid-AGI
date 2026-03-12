# Self-Healing Core Resilience — Implementation Plan

## Scope: Core Resilience First
Circuit Breaker + Retry + Resilient ESP32 Driver Wrapper + Sensor Staleness Detection

---

## Gap Analysis

| What Exists | Gap |
|---|---|
| `CircuitBreakerConfig` in schema.py (failure_threshold, recovery_timeout_s, half_open_max_calls) | No circuit breaker implementation — config is defined but unused |
| `RetryConfig` in schema.py (max_attempts, base_delay_s, max_delay_s, exponential_base) | No retry implementation — config is defined but unused |
| `SafetyConfig.sensor_stale_s = 0.5` | Never checked — safety monitor ignores sensor timestamps entirely |
| `_MAX_LOOP_TIME_MS = 200.0` hardcoded in safety/monitor.py | Should come from `SafetyConfig` |
| `BaseESP32Driver` has no retry/CB | Errors propagate raw to orchestrator's `_sense()` try/except |
| `ObservationProtocol.timestamp` exists | But per-sensor timestamps don't exist — only bundle-level timestamp |
| Factory builds raw drivers | No resilient wrapper option |
| NEXT_STEPS.md §6.4 explicitly plans this work | Not yet started |

---

## Phase 1: Circuit Breaker (`src/mousedroid/resilience/circuit_breaker.py`)

### Design
- Generic async circuit breaker, usable with any async callable
- Three states: `CLOSED` (normal) → `OPEN` (fast-fail) → `HALF_OPEN` (probe) → `CLOSED`
- Uses `CircuitBreakerConfig` from schema — no hardcoded values
- Thread-safe via asyncio.Lock
- Emits structured log events on every state transition

### API
```python
class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, name: str, cfg: CircuitBreakerConfig) -> None: ...
    @property
    def state(self) -> CircuitState: ...
    @property
    def failure_count(self) -> int: ...
    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T: ...
    def reset(self) -> None: ...

class CircuitOpenError(Exception):
    """Raised when circuit is open and call is rejected."""
```

### Reuses
- `CircuitBreakerConfig` from `mousedroid.config.schema`
- `get_logger(__name__)` from `mousedroid.logging.setup`

### Test File: `tests/unit/test_circuit_breaker.py`
- `test_initial_state_closed`
- `test_successful_call_stays_closed`
- `test_failures_below_threshold_stay_closed`
- `test_failures_at_threshold_opens_circuit`
- `test_open_circuit_rejects_calls`
- `test_open_transitions_to_half_open_after_timeout`
- `test_half_open_success_closes_circuit`
- `test_half_open_failure_reopens_circuit`
- `test_half_open_limits_concurrent_calls`
- `test_reset_clears_state`
- `test_concurrent_calls_thread_safe`
- `test_config_values_not_hardcoded` (verifies behavior changes with different configs)

---

## Phase 2: Retry with Exponential Backoff (`src/mousedroid/resilience/retry.py`)

### Design
- Generic async retry decorator and context manager
- Exponential backoff with jitter (prevents thundering herd)
- Configurable retryable exception types (default: `Exception`)
- Uses `RetryConfig` from schema — no hardcoded values
- Composable: can wrap inside circuit breaker

### API
```python
class RetryExhaustedError(Exception):
    """All retry attempts failed."""
    attempts: int
    last_exception: BaseException

async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    cfg: RetryConfig,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
    **kwargs: Any,
) -> T: ...

def with_retry(
    cfg: RetryConfig,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]: ...
```

### Backoff Formula
```python
delay = min(cfg.base_delay_s * (cfg.exponential_base ** attempt), cfg.max_delay_s)
jitter = random.uniform(0, delay * 0.1)  # 10% jitter
actual_delay = delay + jitter
```

### Reuses
- `RetryConfig` from `mousedroid.config.schema`
- `get_logger(__name__)` from `mousedroid.logging.setup`

### Test File: `tests/unit/test_retry.py`
- `test_succeeds_first_try_no_retry`
- `test_retries_on_failure_then_succeeds`
- `test_exhausts_all_attempts`
- `test_raises_retry_exhausted_with_last_exception`
- `test_respects_max_attempts_config`
- `test_exponential_backoff_timing`
- `test_jitter_adds_randomness`
- `test_max_delay_caps_backoff`
- `test_only_retries_specified_exceptions`
- `test_non_retryable_exception_raises_immediately`
- `test_decorator_form`
- `test_config_values_not_hardcoded`

---

## Phase 3: Resilient ESP32 Driver Wrapper (`src/mousedroid/resilience/resilient_driver.py`)

### Design
- Wraps any `ESP32CommProtocol` with circuit breaker + retry
- Implements `ESP32CommProtocol` itself (transparent to orchestrator)
- Decorator pattern — no changes to existing drivers
- Auto-reconnection: on circuit open, attempts reconnect before retry
- Separate circuit breakers per operation type (command vs query)

### API
```python
class ResilientESP32Driver:
    """ESP32 driver wrapper with circuit breaker and retry.

    Implements ESP32CommProtocol — drop-in replacement.
    """
    def __init__(
        self,
        inner: ESP32CommProtocol,
        retry_cfg: RetryConfig,
        cb_cfg: CircuitBreakerConfig,
    ) -> None: ...

    # All ESP32CommProtocol methods delegated through CB + retry
    async def connect(self) -> None: ...
    async def send_velocity(self, vx: float, vy: float, omega: float) -> None: ...
    async def read_encoders(self) -> EncoderReading: ...
    async def get_battery_voltage(self) -> float: ...
    async def emergency_stop(self) -> None: ...
    async def disconnect(self) -> None: ...

    @property
    def circuit_state(self) -> CircuitState: ...
    @property
    def stats(self) -> dict[str, Any]: ...
```

### Reuses
- `ESP32CommProtocol` from `mousedroid.comms.protocol`
- `CircuitBreaker` from Phase 1
- `retry_async` from Phase 2
- Both config objects from schema

### Test File: `tests/unit/test_resilient_driver.py`
- `test_delegates_to_inner_driver`
- `test_retries_send_velocity_on_failure`
- `test_circuit_breaker_opens_after_threshold`
- `test_circuit_open_returns_fallback_for_queries`
- `test_emergency_stop_bypasses_circuit_breaker` (safety-critical)
- `test_reconnect_on_sustained_failure`
- `test_stats_tracking`
- `test_protocol_conformance` (isinstance check)

---

## Phase 4: Sensor Staleness Detection (modify `safety/monitor.py`)

### Design
- Track per-sensor last-valid timestamp in safety monitor
- Compare against `cfg.sensor_stale_s` threshold
- Stale sensors reduce `valid_sensor_count` → can trigger emergency
- Move `_MAX_LOOP_TIME_MS` to `SafetyConfig.max_loop_time_ms`
- Backwards compatible: new config field has default matching current hardcoded value

### Changes

**`config/schema.py`** — Add to `SafetyConfig`:
```python
max_loop_time_ms: float = Field(200.0, gt=0, description="Max loop time before emergency (ms)")
```

**`safety/monitor.py`** — Modify `MouseDroidSafetyMonitor`:
```python
class MouseDroidSafetyMonitor:
    def __init__(self, cfg: SafetyConfig) -> None:
        self._cfg = cfg
        self._last_valid_timestamps: dict[int, float] = {}

    def evaluate(self, observation, loop_time_ms) -> SafetyContext:
        # ... existing checks ...
        # NEW: update per-sensor timestamps from valid_mask
        # NEW: check staleness against cfg.sensor_stale_s
        # CHANGE: use cfg.max_loop_time_ms instead of _MAX_LOOP_TIME_MS
```

### Reuses
- Existing `SafetyConfig.sensor_stale_s` (already defined, never used)
- Existing `valid_mask` from `ObservationProtocol`
- Existing `ObservationProtocol.timestamp` for current time reference

### Test File: Extend `tests/unit/test_safety_monitor.py`
- `test_stale_sensor_reduces_valid_count`
- `test_fresh_sensor_not_flagged_stale`
- `test_staleness_threshold_from_config`
- `test_max_loop_time_from_config`
- `test_backwards_compatible_default_loop_time`

---

## Phase 5: Factory + Module Wiring

### `src/mousedroid/resilience/__init__.py`
```python
from mousedroid.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
from mousedroid.resilience.retry import RetryExhaustedError, retry_async, with_retry
from mousedroid.resilience.resilient_driver import ResilientESP32Driver

__all__ = [
    "CircuitBreaker", "CircuitOpenError", "CircuitState",
    "RetryExhaustedError", "retry_async", "with_retry",
    "ResilientESP32Driver",
]
```

### `factory.py` — Add:
```python
def build_esp32_driver(cfg: Settings) -> ESP32CommProtocol:
    # ... existing driver selection ...
    # NEW: wrap with resilient driver
    from mousedroid.resilience.resilient_driver import ResilientESP32Driver
    return ResilientESP32Driver(inner=driver, retry_cfg=cfg.retry, cb_cfg=cfg.circuit_breaker)
```

### Test File: Extend `tests/unit/test_factory.py`
- `test_build_esp32_returns_resilient_wrapper`
- `test_build_esp32_mock_still_works`

---

## Phase 6: Integration Test

### `tests/integration/test_self_healing_orchestrator.py`
- `test_orchestrator_survives_esp32_transient_failures`
- `test_orchestrator_emergency_stops_on_sustained_failure`
- `test_sensor_staleness_triggers_safety`
- `test_full_tick_with_resilient_driver`

---

## Implementation Order
1. Circuit breaker (no deps)
2. Retry (no deps)
3. Resilient driver (depends on 1+2)
4. Sensor staleness (independent)
5. Factory wiring (depends on 3)
6. Integration tests (depends on all)

All phases can have 1+2 and 4 done in parallel, then 3→5→6 sequentially.

---

## Non-Goals (deferred to future)
- World model state recovery (SurpriseConfig wiring)
- Background health monitor integration
- Serial↔WiFi automatic failover
- Watchdog/heartbeat subsystem
- Auto-throttling on GPU overheating
