"""End-to-end sense-plan-act loop integration test.

Verifies the fully-assembled orchestrator can run N ticks on real hardware
at the configured control frequency without exceeding its timing budget.

Requirements:
    - All hardware connected (HC-SR04, IMX500, ESP32)
    - ``config/jetson_production.yaml`` specifies the test parameters
    - Runs on Jetson — NOT expected to pass on dev machines without hardware

Run on Jetson::

    pytest tests/hardware/test_e2e_sense_plan_act.py -m hardware -v --timeout=30

Performance assertions are derived from config values — no hardcoded numbers.
Timing budget: ``1000.0 / cfg.loop.control_hz`` ms per tick.
Deadline miss threshold: < cfg.loop.max_miss_pct % of ticks (defaults to 5.0).
"""

from __future__ import annotations

import asyncio
import os
import platform
import statistics
import time
from pathlib import Path

import pytest

from tests._jetson_hardware import load_jetson_runtime_settings


def _is_jetson_host() -> bool:
    return platform.system() == "Linux" and Path("/etc/nv_tegra_release").exists()


pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not _is_jetson_host(), reason="Jetson-only hardware test"),
]

JETSON_PROD_CONFIG = os.getenv("MOUSEDROID_JETSON_CONFIG", "config/jetson_production.yaml")

# How many ticks to run in the burst-performance test
_BURST_TICKS = int(os.getenv("MOUSEDROID_E2E_BURST_TICKS", "50"))
# Default deadline-miss ceiling when not in config (5 %)
_DEFAULT_MAX_MISS_PCT = float(os.getenv("MOUSEDROID_E2E_MAX_MISS_PCT", "5.0"))
# Hard upper bound: no single tick may take longer than this multiplier x budget
_HARD_DEADLINE_MULT = float(os.getenv("MOUSEDROID_E2E_HARD_DEADLINE_MULT", "5.0"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_settings():
    """Return Settings from jetson_production.yaml."""
    return load_jetson_runtime_settings()


def _deadline_budget_ms(cfg) -> float:
    """Return the per-tick timing budget in milliseconds."""
    return 1000.0 / cfg.loop.control_hz


def _max_miss_pct(cfg) -> float:
    """Return acceptable deadline-miss percentage — from config or fallback."""
    return float(getattr(getattr(cfg, "loop", None), "max_miss_pct", _DEFAULT_MAX_MISS_PCT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def settings():
    return _load_settings()


@pytest.fixture(scope="module")
async def orchestrator(settings):
    """Build, start, yield, and stop a fully-wired MouseDroidOrchestrator."""
    from mousedroid.factory import build_orchestrator

    orch = build_orchestrator(settings)
    await orch.start()
    yield orch
    await orch.stop()


# ---------------------------------------------------------------------------
# Test 1: The orchestrator starts and stops cleanly
# ---------------------------------------------------------------------------


@pytest.mark.timeout(15)
async def test_orchestrator_starts_and_stops(settings) -> None:
    """``start()`` then ``stop()`` must complete without exceptions."""
    from mousedroid.factory import build_orchestrator

    orch = build_orchestrator(settings)
    await orch.start()
    await orch.stop()


# ---------------------------------------------------------------------------
# Test 2: health_check returns expected keys
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
async def test_health_check_schema(orchestrator) -> None:
    """``health_check()`` must return a dict with required keys."""
    result = await orchestrator.health_check()

    assert isinstance(result, dict), f"health_check() returned {type(result)!r}"
    assert "status" in result, "Missing 'status' key"
    assert result["status"] == "ok", f"Unexpected status: {result['status']!r}"
    assert "mock_hardware" in result
    assert "agents" in result


# ---------------------------------------------------------------------------
# Test 3: Single tick completes without exception
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
async def test_single_tick_completes(orchestrator) -> None:
    """A single ``tick()`` must complete without raising."""
    await orchestrator.tick()


# ---------------------------------------------------------------------------
# Test 4: Burst ticks — mean latency within budget
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
async def test_burst_tick_mean_latency(orchestrator, settings) -> None:
    """Mean tick latency across N ticks must stay within the budget."""
    budget_ms = _deadline_budget_ms(settings)
    tick_times: list[float] = []

    for _ in range(_BURST_TICKS):
        t0 = time.monotonic()
        await orchestrator.tick()
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        tick_times.append(elapsed_ms)

    mean_ms = statistics.mean(tick_times)
    assert mean_ms <= budget_ms, (
        f"Mean tick latency {mean_ms:.1f} ms exceeds budget {budget_ms:.1f} ms "
        f"(control_hz={settings.loop.control_hz})"
    )


# ---------------------------------------------------------------------------
# Test 5: Burst ticks — deadline miss rate below threshold
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
async def test_burst_tick_deadline_miss_rate(orchestrator, settings) -> None:
    """Fraction of ticks exceeding the timing budget must be below cfg threshold."""
    budget_ms = _deadline_budget_ms(settings)
    max_miss = _max_miss_pct(settings)
    tick_times: list[float] = []

    for _ in range(_BURST_TICKS):
        t0 = time.monotonic()
        await orchestrator.tick()
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        tick_times.append(elapsed_ms)

    misses = sum(1 for t in tick_times if t > budget_ms)
    miss_pct = (misses / len(tick_times)) * 100.0

    assert miss_pct <= max_miss, (
        f"Deadline miss rate {miss_pct:.1f}% exceeds threshold {max_miss:.1f}% "
        f"({misses}/{len(tick_times)} ticks over {budget_ms:.1f} ms budget)"
    )


# ---------------------------------------------------------------------------
# Test 6: No catastrophic outlier (hard deadline)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
async def test_burst_tick_no_catastrophic_outlier(orchestrator, settings) -> None:
    """No single tick may take longer than HARD_DEADLINE_MULT x budget."""
    budget_ms = _deadline_budget_ms(settings)
    hard_limit_ms = budget_ms * _HARD_DEADLINE_MULT
    tick_times: list[float] = []

    for _ in range(_BURST_TICKS):
        t0 = time.monotonic()
        await orchestrator.tick()
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        tick_times.append(elapsed_ms)

    worst_ms = max(tick_times)
    assert worst_ms <= hard_limit_ms, (
        f"Worst tick {worst_ms:.1f} ms exceeds hard limit {hard_limit_ms:.1f} ms "
        f"({_HARD_DEADLINE_MULT}x budget={budget_ms:.1f} ms)"
    )


# ---------------------------------------------------------------------------
# Test 7: Tick count advances
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
async def test_tick_count_advances(orchestrator) -> None:
    """Each call to ``tick()`` must increment the internal tick counter."""
    before = orchestrator._tick_count
    await orchestrator.tick()
    after = orchestrator._tick_count
    assert after == before + 1, f"Tick count did not advance: before={before}, after={after}"


# ---------------------------------------------------------------------------
# Test 8: Run for a short wall-clock duration via run()
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
async def test_run_n_seconds(settings) -> None:
    """A short ``run()`` burst must complete N ticks and emit tick-count > 0."""
    from mousedroid.factory import build_orchestrator

    orch = build_orchestrator(settings)
    await orch.start()

    run_seconds = float(os.getenv("MOUSEDROID_E2E_RUN_SECONDS", "3.0"))
    import contextlib

    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(orch.run(), timeout=run_seconds)

    await orch.stop()

    min_expected_ticks = int(settings.loop.control_hz * run_seconds * 0.5)
    assert orch._tick_count >= min_expected_ticks, (
        f"Expected ≥{min_expected_ticks} ticks in {run_seconds}s at "
        f"{settings.loop.control_hz} Hz, got {orch._tick_count}"
    )
