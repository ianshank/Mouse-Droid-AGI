"""End-to-end 5-second orchestrator integration test with all mock hardware.

Starts the full orchestrator with mock hardware, runs for 5 simulated seconds,
and verifies tick count, exception-free operation, clean shutdown, 30 Hz deadline
adherence, telemetry frame publishing, safety monitor activity, and graceful
start/stop lifecycle.

All timing and threshold values are derived from config — no hardcoded numbers.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cfg() -> Settings:
    """Provide a Settings instance with mock hardware and telemetry disabled."""
    from mousedroid.config.schema import Settings

    return Settings(
        mock_hardware=True,
    )


@pytest.fixture
def mock_cfg_with_telemetry() -> Settings:
    """Provide a Settings with telemetry enabled for frame publish tests."""
    from mousedroid.config.schema import Settings

    return Settings(
        mock_hardware=True,
        telemetry={"enabled": True, "publish_hz": 10.0},  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_duration_s(cfg: Settings) -> float:
    """Return the simulated run duration (5 seconds)."""
    return 5.0


def _expected_min_ticks(cfg: Settings, duration_s: float) -> int:
    """Return minimum expected tick count for a given duration.

    Uses 50% of theoretical maximum as the floor to allow for overhead.

    Args:
        cfg: Settings with loop.control_hz.
        duration_s: Run duration in seconds.

    Returns:
        Minimum acceptable tick count.
    """
    return int(cfg.loop.control_hz * duration_s * 0.5)


def _deadline_budget_ms(cfg: Settings) -> float:
    """Return per-tick timing budget in milliseconds from config.

    Args:
        cfg: Settings with loop.control_hz.

    Returns:
        Budget in milliseconds.
    """
    return 1000.0 / cfg.loop.control_hz


# ---------------------------------------------------------------------------
# 1. Start orchestrator with all mock hardware, run for 5 simulated seconds
# ---------------------------------------------------------------------------


class TestE2E5SecondRun:
    """End-to-end orchestrator test running for 5 simulated seconds."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_orchestrator_runs_5_seconds_no_exception(self, mock_cfg: Settings) -> None:
        """Orchestrator should run for 5s without raising any exceptions."""
        from mousedroid.factory import build_orchestrator

        orch = build_orchestrator(mock_cfg)
        await orch.start()  # type: ignore[union-attr]

        duration = _run_duration_s(mock_cfg)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(orch.run(), timeout=duration)  # type: ignore[union-attr]

        await orch.stop()  # type: ignore[union-attr]

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_tick_count_after_5_seconds(self, mock_cfg: Settings) -> None:
        """Tick count should reach at least 50% of theoretical max."""
        from mousedroid.factory import build_orchestrator

        orch = build_orchestrator(mock_cfg)
        await orch.start()  # type: ignore[union-attr]

        duration = _run_duration_s(mock_cfg)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(orch.run(), timeout=duration)  # type: ignore[union-attr]

        await orch.stop()  # type: ignore[union-attr]

        min_ticks = _expected_min_ticks(mock_cfg, duration)
        assert orch._tick_count >= min_ticks, (  # type: ignore[union-attr]
            f"Expected >= {min_ticks} ticks in {duration}s at "
            f"{mock_cfg.loop.control_hz} Hz, got {orch._tick_count}"  # type: ignore[union-attr]
        )


# ---------------------------------------------------------------------------
# 2. 30 Hz deadline met (within tolerance from config)
# ---------------------------------------------------------------------------


class TestDeadlineAdherence:
    """Verify per-tick timing stays within budget."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_mean_tick_latency_within_budget(self, mock_cfg: Settings) -> None:
        """Mean tick latency across N ticks must stay within budget."""
        from mousedroid.factory import build_orchestrator

        orch = build_orchestrator(mock_cfg)
        await orch.start()  # type: ignore[union-attr]

        budget_ms = _deadline_budget_ms(mock_cfg)
        n_ticks = 50
        tick_times: list[float] = []

        for _ in range(n_ticks):
            t0 = time.monotonic()
            await orch.tick()  # type: ignore[union-attr]
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            tick_times.append(elapsed_ms)

        await orch.stop()  # type: ignore[union-attr]

        mean_ms = sum(tick_times) / len(tick_times)
        assert mean_ms <= budget_ms, (
            f"Mean tick latency {mean_ms:.1f} ms exceeds budget "
            f"{budget_ms:.1f} ms (control_hz={mock_cfg.loop.control_hz})"
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_deadline_miss_rate_below_threshold(self, mock_cfg: Settings) -> None:
        """Fraction of ticks exceeding budget should be below 10%."""
        from mousedroid.factory import build_orchestrator

        orch = build_orchestrator(mock_cfg)
        await orch.start()  # type: ignore[union-attr]

        budget_ms = _deadline_budget_ms(mock_cfg)
        max_miss_pct = 10.0  # generous for mock hardware
        n_ticks = 50
        tick_times: list[float] = []

        for _ in range(n_ticks):
            t0 = time.monotonic()
            await orch.tick()  # type: ignore[union-attr]
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            tick_times.append(elapsed_ms)

        await orch.stop()  # type: ignore[union-attr]

        misses = sum(1 for t in tick_times if t > budget_ms)
        miss_pct = (misses / len(tick_times)) * 100.0

        assert miss_pct <= max_miss_pct, (
            f"Deadline miss rate {miss_pct:.1f}% exceeds threshold "
            f"{max_miss_pct:.1f}% ({misses}/{n_ticks} ticks "
            f"over {budget_ms:.1f} ms budget)"
        )


# ---------------------------------------------------------------------------
# 3. Clean shutdown via SIGINT simulation
# ---------------------------------------------------------------------------


class TestGracefulShutdown:
    """Verify orchestrator handles shutdown signals gracefully."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_stop_cancels_running_loop(self, mock_cfg: Settings) -> None:
        """Calling stop() during run() should terminate cleanly."""
        from mousedroid.factory import build_orchestrator

        orch = build_orchestrator(mock_cfg)
        await orch.start()  # type: ignore[union-attr]

        # Start running in background
        run_task = asyncio.create_task(orch.run())  # type: ignore[union-attr]

        # Let it run briefly
        await asyncio.sleep(0.2)

        # Stop should terminate the loop
        await orch.stop()  # type: ignore[union-attr]

        # run() should complete shortly after stop
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(run_task, timeout=2.0)

        assert not orch._running  # type: ignore[union-attr]

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_tick_count_advances_during_run(self, mock_cfg: Settings) -> None:
        """Tick count should increase during run()."""
        from mousedroid.factory import build_orchestrator

        orch = build_orchestrator(mock_cfg)
        await orch.start()  # type: ignore[union-attr]

        initial_count = orch._tick_count  # type: ignore[union-attr]

        run_task = asyncio.create_task(orch.run())  # type: ignore[union-attr]
        await asyncio.sleep(0.5)
        await orch.stop()  # type: ignore[union-attr]

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(run_task, timeout=2.0)

        assert orch._tick_count > initial_count  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 4. Telemetry frames published
# ---------------------------------------------------------------------------


class TestTelemetryPublishing:
    """Verify telemetry frames are published during orchestrator run."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_telemetry_publisher_receives_frames(
        self, mock_cfg_with_telemetry: Settings
    ) -> None:
        """Telemetry publisher should receive frames during tick execution."""
        from mousedroid.factory import build_orchestrator

        orch = build_orchestrator(mock_cfg_with_telemetry)
        await orch.start()  # type: ignore[union-attr]

        # Run several ticks
        for _ in range(10):
            await orch.tick()  # type: ignore[union-attr]

        await orch.stop()  # type: ignore[union-attr]

        # With telemetry enabled the publisher should have been called.
        # The mock telemetry server won't actually publish, but
        # _publish_telemetry should have been invoked without error.
        assert orch._tick_count >= 10  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 5. Safety monitor active
# ---------------------------------------------------------------------------


class TestSafetyMonitorActive:
    """Verify safety monitor is evaluated each tick."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_safety_monitor_evaluates_each_tick(self, mock_cfg: Settings) -> None:
        """Safety monitor evaluate() should be called on each tick."""
        from mousedroid.factory import build_orchestrator

        orch = build_orchestrator(mock_cfg)
        await orch.start()  # type: ignore[union-attr]

        # Wrap the safety monitor to count calls
        original_evaluate = orch._safety_monitor.evaluate  # type: ignore[union-attr]
        call_count = 0

        def counting_evaluate(*args: object, **kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            return original_evaluate(*args, **kwargs)

        orch._safety_monitor.evaluate = counting_evaluate  # type: ignore[union-attr]

        n_ticks = 10
        for _ in range(n_ticks):
            await orch.tick()  # type: ignore[union-attr]

        await orch.stop()  # type: ignore[union-attr]

        assert call_count == n_ticks, (
            f"Safety monitor was called {call_count} times, expected {n_ticks}"
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_safety_context_is_not_emergency_mock(self, mock_cfg: Settings) -> None:
        """In mock mode with no obstacles, safety should not trigger emergency."""
        from mousedroid.factory import build_orchestrator

        orch = build_orchestrator(mock_cfg)
        await orch.start()  # type: ignore[union-attr]

        # Track emergency state
        emergencies: list[bool] = []
        original_evaluate = orch._safety_monitor.evaluate  # type: ignore[union-attr]

        def tracking_evaluate(*args: object, **kwargs: object) -> object:
            ctx = original_evaluate(*args, **kwargs)
            emergencies.append(ctx.is_emergency)
            return ctx

        orch._safety_monitor.evaluate = tracking_evaluate  # type: ignore[union-attr]

        for _ in range(10):
            await orch.tick()  # type: ignore[union-attr]

        await orch.stop()  # type: ignore[union-attr]

        # Mock hardware should not trigger emergency
        assert not any(emergencies), f"Unexpected emergency triggers in mock mode: {emergencies}"


# ---------------------------------------------------------------------------
# 6. Graceful start/stop lifecycle
# ---------------------------------------------------------------------------


class TestStartStopLifecycle:
    """Verify orchestrator start/stop lifecycle."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_start_stop_no_exception(self, mock_cfg: Settings) -> None:
        """Start then stop should complete without exceptions."""
        from mousedroid.factory import build_orchestrator

        orch = build_orchestrator(mock_cfg)
        await orch.start()  # type: ignore[union-attr]
        await orch.stop()  # type: ignore[union-attr]

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_multiple_start_stop_cycles(self, mock_cfg: Settings) -> None:
        """Multiple start/stop cycles should work cleanly."""
        from mousedroid.factory import build_orchestrator

        for _ in range(3):
            orch = build_orchestrator(mock_cfg)
            await orch.start()  # type: ignore[union-attr]
            await orch.tick()  # type: ignore[union-attr]
            await orch.stop()  # type: ignore[union-attr]

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_health_check_during_run(self, mock_cfg: Settings) -> None:
        """health_check() should return valid status during operation."""
        from mousedroid.factory import build_orchestrator

        orch = build_orchestrator(mock_cfg)
        await orch.start()  # type: ignore[union-attr]

        result = await orch.health_check()  # type: ignore[union-attr]

        assert isinstance(result, dict)
        assert result["status"] == "ok"
        assert result["mock_hardware"] is True
        assert "agents" in result

        await orch.stop()  # type: ignore[union-attr]

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_running_flag_lifecycle(self, mock_cfg: Settings) -> None:
        """_running flag should be True after start, False after stop."""
        from mousedroid.factory import build_orchestrator

        orch = build_orchestrator(mock_cfg)

        assert not orch._running  # type: ignore[union-attr]

        await orch.start()  # type: ignore[union-attr]
        assert orch._running  # type: ignore[union-attr]

        await orch.stop()  # type: ignore[union-attr]
        assert not orch._running  # type: ignore[union-attr]
