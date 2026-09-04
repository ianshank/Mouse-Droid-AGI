"""AQA pins for the tick-instrumentation config surface.

Backgrounds the defect these fields exist to fix: ``orchestrator.tick``
computes ``loop_time_ms`` immediately after ``sensor_manager.read_all()`` and
never recomputes it, then feeds that value to the safety monitor's
loop-overrun interlock *and* to the Prometheus gauge and histogram. Planning,
inference, actuation and telemetry all happen after the measurement, so the
phases most likely to blow the 33.3 ms budget are invisible to both.

Fixing that means feeding the monitor a real tick duration, which needs a
warm-up grace (a Jetson's first tick pays lazy CUDA context creation and
TensorRT warm-up and routinely exceeds 200 ms) and a consecutive-overrun
debounce. This file pins the shape of those knobs; the companion
``*_backwards_compat.py`` pins that they change nothing by default.
"""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from mousedroid.config.schema import Settings
from mousedroid.config.schema._primitives import TickPhaseLiteral
from mousedroid.config.schema.reward_safety import SafetyConfig
from mousedroid.config.schema.telemetry import MetricsConfig

_NEW_SAFETY_FIELDS = (
    "max_loop_time_factor",
    "loop_overrun_consecutive_ticks",
    "loop_overrun_warmup_ticks",
    "loop_soft_budget_factor",
)
_NEW_METRICS_FIELDS = ("track_tick_phases", "tick_phase_buckets_ms")


class TestFieldHygiene:
    """House rule: every new field is documented and defaulted."""

    @pytest.mark.parametrize("name", _NEW_SAFETY_FIELDS)
    def test_safety_field_documented(self, name: str) -> None:
        info = SafetyConfig.model_fields[name]
        assert info.description, f"SafetyConfig.{name} must carry a description"
        assert not info.is_required(), f"SafetyConfig.{name} must default (invariant 6)"

    @pytest.mark.parametrize("name", _NEW_METRICS_FIELDS)
    def test_metrics_field_documented(self, name: str) -> None:
        info = MetricsConfig.model_fields[name]
        assert info.description, f"MetricsConfig.{name} must carry a description"
        assert not info.is_required(), f"MetricsConfig.{name} must default (invariant 6)"


class TestBounds:
    """The knobs must reject values that would disable the interlock."""

    def test_factor_must_exceed_one(self) -> None:
        """A factor <= 1 would trip the e-stop on a tick that met its budget."""
        with pytest.raises(ValidationError):
            SafetyConfig(max_loop_time_factor=1.0)

    def test_consecutive_ticks_must_be_at_least_one(self) -> None:
        """Zero would mean 'never trip', silently disabling the interlock."""
        with pytest.raises(ValidationError):
            SafetyConfig(loop_overrun_consecutive_ticks=0)

    def test_warmup_ticks_may_be_zero_but_not_negative(self) -> None:
        assert SafetyConfig(loop_overrun_warmup_ticks=0).loop_overrun_warmup_ticks == 0
        with pytest.raises(ValidationError):
            SafetyConfig(loop_overrun_warmup_ticks=-1)

    def test_soft_budget_factor_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            SafetyConfig(loop_soft_budget_factor=0.0)


class TestDerivation:
    """``max_loop_time_factor`` must track ``loop.control_hz``."""

    def test_factor_derives_the_historical_threshold_at_30hz(self) -> None:
        """factor=6.0 at 30 Hz reproduces the historical literal 200 ms.

        That 6x relationship was undocumented folklore; this pins it as an
        explicit, stated rule.
        """
        settings = Settings(mock_hardware=True, safety={"max_loop_time_factor": 6.0})
        assert settings.safety.max_loop_time_ms == pytest.approx(200.0)

    @pytest.mark.parametrize(
        ("control_hz", "expected_ms"),
        [(30.0, 200.0), (10.0, 600.0), (60.0, 100.0)],
    )
    def test_derived_threshold_scales_with_control_hz(
        self, control_hz: float, expected_ms: float
    ) -> None:
        """The whole point: the threshold follows the tick rate."""
        settings = Settings(
            mock_hardware=True,
            loop={"control_hz": control_hz},
            safety={"max_loop_time_factor": 6.0},
        )
        assert settings.safety.max_loop_time_ms == pytest.approx(expected_ms)

    def test_literal_threshold_is_used_when_no_factor_is_set(self) -> None:
        settings = Settings(
            mock_hardware=True,
            loop={"control_hz": 10.0},
            safety={"max_loop_time_ms": 123.0},
        )
        assert settings.safety.max_loop_time_ms == pytest.approx(123.0)


class TestTickPhases:
    """The phase label set must be bounded and straddle the budget."""

    def test_phases_are_a_closed_set(self) -> None:
        phases = get_args(TickPhaseLiteral)
        assert len(phases) == len(set(phases)), "duplicate phase label"
        assert len(phases) == 8, f"expected the 8 tiling phases, got {phases}"

    def test_phases_tile_the_tick_in_execution_order(self) -> None:
        """Order is meaningful: it is the order the tick executes them."""
        assert get_args(TickPhaseLiteral) == (
            "sense",
            "safety",
            "world_model",
            "plan",
            "act",
            "learn",
            "telemetry",
            "post",
        )

    def test_buckets_straddle_the_30hz_budget(self) -> None:
        """A boundary must sit ON 33.3 so le="33.3" is the in-budget fraction."""
        buckets = MetricsConfig().tick_phase_buckets_ms
        finite = [b for b in buckets if b != float("inf")]
        assert min(finite) < 33.3 < max(finite)
        assert any(b == pytest.approx(33.3) for b in finite), (
            "a bucket boundary must land exactly on the 33.3 ms period, or the "
            "in-budget fraction has to be interpolated"
        )

    def test_bucket_validator_covers_the_new_field(self) -> None:
        """Ascending/positive validation must apply to the new bucket tuple."""
        with pytest.raises(ValidationError):
            MetricsConfig(tick_phase_buckets_ms=(5.0, 1.0, 10.0))
