"""Loop-overrun debounce, warm-up grace, and the recovery-path dedupe.

``_evaluate_loop_timing`` exists because feeding the safety monitor a *real*
tick duration (rather than the sensor-read segment it receives today) would
otherwise emergency-stop the rover at every boot: a Jetson's first ticks pay
lazy CUDA context creation and TensorRT/ONNX kernel warm-up and routinely
exceed ``max_loop_time_ms``.

The dedupe matters just as much. ``orchestrator.tick`` calls ``evaluate``
twice when sensor recovery fires, so a naive streak counter would advance
twice per tick and trip at half the configured threshold — turning a debounce
into a shorter fuse.
"""

from __future__ import annotations

import numpy as np
import pytest

from mousedroid.config.schema.reward_safety import SafetyConfig
from mousedroid.safety.monitor import MouseDroidSafetyMonitor
from mousedroid.sensing.bundle import MouseDroidObservationBundle

_OVER_BUDGET_MS = 300.0  # > the 200.0 default max_loop_time_ms
_IN_BUDGET_MS = 5.0


def _observation() -> MouseDroidObservationBundle:
    """A bundle that is safe on every axis except whatever the test varies."""
    return MouseDroidObservationBundle(
        _timestamp=0.0,
        _vision_features=np.zeros(8, dtype=np.float32),
        _distance_m=1.5,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _audio_chunk=np.zeros(16, dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )


def _emergencies(monitor: MouseDroidSafetyMonitor, ticks: int, ms: float) -> list[bool]:
    return [monitor.evaluate(_observation(), ms, tick_index=i).is_emergency for i in range(ticks)]


class TestDefaultsPreserveHistoricalBehaviour:
    """At shipped defaults the monitor must behave exactly as it always did."""

    def test_single_overrun_trips_immediately(self) -> None:
        monitor = MouseDroidSafetyMonitor(SafetyConfig())
        assert monitor.evaluate(_observation(), _OVER_BUDGET_MS).is_emergency is True

    def test_in_budget_tick_does_not_trip(self) -> None:
        monitor = MouseDroidSafetyMonitor(SafetyConfig())
        assert monitor.evaluate(_observation(), _IN_BUDGET_MS).is_emergency is False

    def test_tick_index_is_optional(self) -> None:
        """Callers that predate the debounce must keep working unchanged."""
        monitor = MouseDroidSafetyMonitor(SafetyConfig())
        assert monitor.evaluate(_observation(), _OVER_BUDGET_MS).is_emergency is True


class TestDebounce:
    """``loop_overrun_consecutive_ticks`` requires a sustained overrun."""

    def test_trips_only_on_the_nth_consecutive_overrun(self) -> None:
        monitor = MouseDroidSafetyMonitor(SafetyConfig(loop_overrun_consecutive_ticks=3))
        assert _emergencies(monitor, 4, _OVER_BUDGET_MS) == [False, False, True, True]

    def test_an_in_budget_tick_resets_the_streak(self) -> None:
        """A recovered loop must not carry its old streak into the next spike."""
        monitor = MouseDroidSafetyMonitor(SafetyConfig(loop_overrun_consecutive_ticks=3))
        assert monitor.evaluate(_observation(), _OVER_BUDGET_MS, tick_index=0).is_emergency is False
        assert monitor.evaluate(_observation(), _OVER_BUDGET_MS, tick_index=1).is_emergency is False
        assert monitor.evaluate(_observation(), _IN_BUDGET_MS, tick_index=2).is_emergency is False
        # Streak restarts from zero, so two more overruns are still not enough.
        assert monitor.evaluate(_observation(), _OVER_BUDGET_MS, tick_index=3).is_emergency is False
        assert monitor.evaluate(_observation(), _OVER_BUDGET_MS, tick_index=4).is_emergency is False
        assert monitor.evaluate(_observation(), _OVER_BUDGET_MS, tick_index=5).is_emergency is True


class TestWarmupGrace:
    """``loop_overrun_warmup_ticks`` covers first-inference cost."""

    def test_overruns_inside_the_window_do_not_trip(self) -> None:
        monitor = MouseDroidSafetyMonitor(SafetyConfig(loop_overrun_warmup_ticks=2))
        assert _emergencies(monitor, 4, _OVER_BUDGET_MS) == [False, False, True, True]

    def test_zero_warmup_is_the_default_and_trips_on_tick_zero(self) -> None:
        monitor = MouseDroidSafetyMonitor(SafetyConfig())
        assert monitor.evaluate(_observation(), _OVER_BUDGET_MS, tick_index=0).is_emergency is True


class TestRecoveryPathDedupe:
    """Repeated ``evaluate`` calls within one tick must count once."""

    def test_same_tick_index_does_not_advance_the_streak(self) -> None:
        """The orchestrator evaluates twice when sensor recovery fires.

        Without the dedupe a `consecutive_ticks=3` debounce would trip after
        two real ticks, silently halving the operator's configured fuse.
        """
        monitor = MouseDroidSafetyMonitor(SafetyConfig(loop_overrun_consecutive_ticks=3))
        results = [
            monitor.evaluate(_observation(), _OVER_BUDGET_MS, tick_index=0).is_emergency
            for _ in range(5)
        ]
        assert results == [False] * 5, (
            "five evaluations of the SAME tick advanced the overrun streak; the "
            "sensor-recovery path calls evaluate twice per tick, so this would "
            "trip the emergency stop at half the configured threshold"
        )

    def test_distinct_tick_indices_do_advance(self) -> None:
        monitor = MouseDroidSafetyMonitor(SafetyConfig(loop_overrun_consecutive_ticks=3))
        assert _emergencies(monitor, 3, _OVER_BUDGET_MS)[-1] is True

    def test_none_tick_index_counts_every_call(self) -> None:
        """``None`` means 'caller does not track ticks' — today's semantics."""
        monitor = MouseDroidSafetyMonitor(SafetyConfig(loop_overrun_consecutive_ticks=2))
        assert monitor.evaluate(_observation(), _OVER_BUDGET_MS).is_emergency is False
        assert monitor.evaluate(_observation(), _OVER_BUDGET_MS).is_emergency is True


class TestCombined:
    """Warm-up and debounce compose without either being bypassed."""

    @pytest.mark.parametrize(
        ("warmup", "consecutive", "expected_first_trip_tick"),
        [(0, 1, 0), (2, 1, 2), (0, 3, 2), (2, 3, 2)],
    )
    def test_first_trip_tick(
        self, warmup: int, consecutive: int, expected_first_trip_tick: int
    ) -> None:
        monitor = MouseDroidSafetyMonitor(
            SafetyConfig(
                loop_overrun_warmup_ticks=warmup,
                loop_overrun_consecutive_ticks=consecutive,
            )
        )
        results = _emergencies(monitor, 6, _OVER_BUDGET_MS)
        assert results.index(True) == expected_first_trip_tick
