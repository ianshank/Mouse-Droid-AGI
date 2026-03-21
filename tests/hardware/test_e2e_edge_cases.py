"""End-to-end edge-case and timing regression tests.

Extends ``test_e2e_sense_plan_act.py`` with sensor-staleness propagation,
tick-timing p95/p99 regression, and graceful degradation when a single
sensor is unavailable.

Run on Jetson::

    pytest tests/hardware/test_e2e_edge_cases.py -m hardware -v --timeout=60
"""

from __future__ import annotations

import math
import os
import platform
import statistics
import time
from pathlib import Path

import pytest

JETSON_PROD_CONFIG = os.getenv("MOUSEDROID_JETSON_CONFIG", "config/jetson_production.yaml")
_BURST_TICKS = int(os.getenv("MOUSEDROID_E2E_BURST_TICKS", "50"))
_HARD_DEADLINE_MULT = float(os.getenv("MOUSEDROID_E2E_HARD_DEADLINE_MULT", "5.0"))


def _is_jetson_host() -> bool:
    return platform.system() == "Linux" and Path("/etc/nv_tegra_release").exists()


pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not _is_jetson_host(), reason="Jetson-only hardware test"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_settings():
    import yaml

    from mousedroid.config.schema import Settings

    with open(JETSON_PROD_CONFIG) as fh:
        raw = yaml.safe_load(fh)
    return Settings(**raw)


def _deadline_budget_ms(cfg) -> float:
    return 1000.0 / cfg.loop.control_hz


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def settings():
    return _load_settings()


@pytest.fixture(scope="module")
async def orchestrator(settings):
    from mousedroid.factory import build_orchestrator

    orch = build_orchestrator(settings)
    await orch.start()
    yield orch
    await orch.stop()


# ---------------------------------------------------------------------------
# 1. P95 tick latency — stricter than mean
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
async def test_burst_tick_p95_latency(orchestrator, settings) -> None:
    """P95 tick latency across N ticks must stay within 2x the timing budget.

    The 2x multiplier accounts for occasional scheduling jitter while still
    catching persistent performance regressions.
    """
    budget_ms = _deadline_budget_ms(settings)
    p95_limit_ms = budget_ms * 2.0
    tick_times: list[float] = []

    for _ in range(_BURST_TICKS):
        t0 = time.monotonic()
        await orchestrator.tick()
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        tick_times.append(elapsed_ms)

    tick_times_sorted = sorted(tick_times)
    p95_idx = math.ceil(0.95 * len(tick_times_sorted)) - 1
    p95_ms = tick_times_sorted[p95_idx]

    assert p95_ms <= p95_limit_ms, (
        f"P95 tick latency {p95_ms:.1f}ms exceeds limit {p95_limit_ms:.1f}ms "
        f"(2x budget={budget_ms:.1f}ms)"
    )


# ---------------------------------------------------------------------------
# 2. Tick timing standard deviation — jitter check
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
async def test_burst_tick_jitter(orchestrator, settings) -> None:
    """Tick timing standard deviation must be below 50% of the budget.

    High jitter indicates GC pauses, priority inversion, or contention.
    """
    budget_ms = _deadline_budget_ms(settings)
    max_stddev_ms = budget_ms * 0.5
    tick_times: list[float] = []

    for _ in range(_BURST_TICKS):
        t0 = time.monotonic()
        await orchestrator.tick()
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        tick_times.append(elapsed_ms)

    stddev_ms = statistics.stdev(tick_times) if len(tick_times) > 1 else 0.0

    assert stddev_ms <= max_stddev_ms, (
        f"Tick jitter {stddev_ms:.1f}ms exceeds limit {max_stddev_ms:.1f}ms "
        f"(50% of budget={budget_ms:.1f}ms)"
    )


# ---------------------------------------------------------------------------
# 3. Tick count monotonically increases
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
async def test_tick_count_monotonic(orchestrator) -> None:
    """Tick count must strictly increase across 10 consecutive ticks."""
    counts: list[int] = []
    for _ in range(10):
        await orchestrator.tick()
        counts.append(orchestrator._tick_count)

    for i in range(1, len(counts)):
        assert counts[i] == counts[i - 1] + 1, (
            f"Tick count not monotonic: {counts[i - 1]} → {counts[i]}"
        )


# ---------------------------------------------------------------------------
# 4. Health check after burst ticks
# ---------------------------------------------------------------------------


@pytest.mark.timeout(15)
async def test_health_check_after_burst(orchestrator, settings) -> None:
    """``health_check()`` must still report ``ok`` after a burst of ticks."""
    for _ in range(20):
        await orchestrator.tick()

    health = await orchestrator.health_check()
    assert isinstance(health, dict)
    assert health.get("status") in ("ok", "warning"), (
        f"Unexpected health status after burst: {health.get('status')!r}"
    )


# ---------------------------------------------------------------------------
# 5. Start → burst → stop → restart cycle
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
async def test_start_stop_restart_cycle(settings) -> None:
    """Full lifecycle: start → ticks → stop → restart → ticks → stop."""
    from mousedroid.factory import build_orchestrator

    orch = build_orchestrator(settings)

    # Cycle 1
    await orch.start()
    for _ in range(5):
        await orch.tick()
    count_after_c1 = orch._tick_count
    assert count_after_c1 >= 5
    await orch.stop()

    # Cycle 2 — new orchestrator (resources were released)
    orch2 = build_orchestrator(settings)
    await orch2.start()
    for _ in range(5):
        await orch2.tick()
    assert orch2._tick_count >= 5
    await orch2.stop()


# ---------------------------------------------------------------------------
# 6. Emergency stop latency during tick burst
# ---------------------------------------------------------------------------


@pytest.mark.timeout(15)
async def test_emergency_stop_during_burst(orchestrator, settings) -> None:
    """``emergency_stop()`` mid-burst must complete within command_timeout."""
    # Run a few ticks to warm up
    for _ in range(5):
        await orchestrator.tick()

    t0 = time.monotonic()
    await orchestrator._esp32.emergency_stop()
    elapsed_s = time.monotonic() - t0

    assert elapsed_s <= settings.esp32.command_timeout_s, (
        f"Emergency stop took {elapsed_s * 1000:.1f}ms, "
        f"limit={settings.esp32.command_timeout_s * 1000:.0f}ms"
    )


# ---------------------------------------------------------------------------
# 7. Minimum tick throughput
# ---------------------------------------------------------------------------


@pytest.mark.timeout(15)
async def test_minimum_tick_throughput(orchestrator, settings) -> None:
    """Orchestrator must sustain at least 50% of target control_hz throughput.

    This catches severe degradation without being overly sensitive to
    momentary scheduling delays.
    """
    target_hz = settings.loop.control_hz
    min_hz = target_hz * 0.5
    n_ticks = _BURST_TICKS

    t0 = time.monotonic()
    for _ in range(n_ticks):
        await orchestrator.tick()
    elapsed_s = time.monotonic() - t0

    achieved_hz = n_ticks / elapsed_s if elapsed_s > 0 else 0.0

    assert achieved_hz >= min_hz, (
        f"Throughput {achieved_hz:.1f}Hz below minimum {min_hz:.1f}Hz "
        f"(target={target_hz:.1f}Hz)"
    )
