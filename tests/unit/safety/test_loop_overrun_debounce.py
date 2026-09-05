"""Loop-overrun debounce, warm-up grace, and the recovery-path dedupe.

``_evaluate_loop_timing`` exists because the safety monitor is now handed each
tick's *real* measured duration rather than the sensor-read segment it used to
receive. Without these guards that would emergency-stop the rover at every
boot: a Jetson's first ticks pay lazy CUDA context creation and TensorRT/ONNX
kernel warm-up and routinely exceed ``max_loop_time_ms``. This is not
hypothetical — ``tests/integration/test_e2e_5sec_run.py`` tripped the interlock
on a single 1.3 s MCTS plan before the guards were given non-zero defaults.

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


def _armed(*, warmup: int = 0, consecutive: int = 1) -> SafetyConfig:
    """A config with both boot guards set explicitly, armed from tick zero.

    The shipped defaults carry a warm-up window and an overrun debounce (see
    :class:`TestShippedDefaults`). A test exercising ONE of those mechanisms
    must pin the other rather than inherit it, or it ends up asserting against
    a default that is itself under test — and silently changes meaning the
    next time the default is retuned.
    """
    return SafetyConfig(
        loop_overrun_warmup_ticks=warmup,
        loop_overrun_consecutive_ticks=consecutive,
    )


class TestShippedDefaults:
    """At shipped defaults the interlock must be armed but not hair-trigger.

    Both halves matter. Defaults that never trip are a disabled safety check;
    defaults that trip on one sample emergency-stop the rover for a single
    slow MCTS plan, which ``tests/integration/test_e2e_5sec_run.py`` shows is
    an ordinary event on a loaded machine (1.3 s ticks against a 200 ms
    ceiling).
    """

    def test_a_single_overrun_does_not_trip(self) -> None:
        """One slow tick is a spike, not a control-loop failure."""
        monitor = MouseDroidSafetyMonitor(SafetyConfig())
        assert monitor.evaluate(_observation(), _OVER_BUDGET_MS, tick_index=0).is_emergency is False

    def test_a_sustained_overrun_still_trips(self) -> None:
        """The anti-cheat half: the interlock must not be off by default."""
        cfg = SafetyConfig()
        monitor = MouseDroidSafetyMonitor(cfg)
        # Spend the warm-up grace with healthy ticks, then sustain an overrun.
        for tick in range(cfg.loop_overrun_warmup_ticks):
            monitor.evaluate(_observation(), _IN_BUDGET_MS, tick_index=tick)
        start = cfg.loop_overrun_warmup_ticks
        tripped = [
            monitor.evaluate(_observation(), _OVER_BUDGET_MS, tick_index=start + i).is_emergency
            for i in range(cfg.loop_overrun_consecutive_ticks)
        ]
        assert tripped[-1] is True, (
            f"{cfg.loop_overrun_consecutive_ticks} consecutive ticks at "
            f"{_OVER_BUDGET_MS} ms against a {cfg.max_loop_time_ms} ms ceiling "
            "must emergency-stop; defaults that cannot trip are a disabled check"
        )

    def test_in_budget_tick_does_not_trip(self) -> None:
        monitor = MouseDroidSafetyMonitor(SafetyConfig())
        assert monitor.evaluate(_observation(), _IN_BUDGET_MS).is_emergency is False

    def test_tick_index_is_optional(self) -> None:
        """Callers that predate the debounce must keep working unchanged."""
        monitor = MouseDroidSafetyMonitor(SafetyConfig(loop_overrun_warmup_ticks=0))
        # No tick_index => every call counts, which is the pre-debounce
        # semantic; the streak still has to reach the configured length.
        results = [monitor.evaluate(_observation(), _OVER_BUDGET_MS).is_emergency for _ in range(3)]
        assert results[-1] is True


class TestDebounce:
    """``loop_overrun_consecutive_ticks`` requires a sustained overrun."""

    def test_trips_only_on_the_nth_consecutive_overrun(self) -> None:
        monitor = MouseDroidSafetyMonitor(_armed(consecutive=3))
        assert _emergencies(monitor, 4, _OVER_BUDGET_MS) == [False, False, True, True]

    def test_an_in_budget_tick_resets_the_streak(self) -> None:
        """A recovered loop must not carry its old streak into the next spike."""
        monitor = MouseDroidSafetyMonitor(_armed(consecutive=3))
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
        monitor = MouseDroidSafetyMonitor(_armed(warmup=2))
        assert _emergencies(monitor, 4, _OVER_BUDGET_MS) == [False, False, True, True]

    def test_zero_warmup_trips_on_tick_zero(self) -> None:
        """``warmup=0`` must arm the interlock from the very first tick.

        Not the shipped default any more — see :class:`TestShippedDefaults` —
        but it stays a supported setting for a rig with no lazy inference
        cost, so the zero case still needs a pin.
        """
        monitor = MouseDroidSafetyMonitor(_armed(warmup=0, consecutive=1))
        assert monitor.evaluate(_observation(), _OVER_BUDGET_MS, tick_index=0).is_emergency is True


class TestRecoveryPathDedupe:
    """Repeated ``evaluate`` calls within one tick must count once."""

    def test_same_tick_index_does_not_advance_the_streak(self) -> None:
        """The orchestrator evaluates twice when sensor recovery fires.

        Without the dedupe a `consecutive_ticks=3` debounce would trip after
        two real ticks, silently halving the operator's configured fuse.
        """
        monitor = MouseDroidSafetyMonitor(_armed(consecutive=3))
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
        monitor = MouseDroidSafetyMonitor(_armed(consecutive=3))
        assert _emergencies(monitor, 3, _OVER_BUDGET_MS)[-1] is True

    def test_none_tick_index_counts_every_call(self) -> None:
        """``None`` means 'caller does not track ticks': every call counts."""
        monitor = MouseDroidSafetyMonitor(_armed(consecutive=2))
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
