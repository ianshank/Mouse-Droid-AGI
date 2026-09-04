"""AQA pins: safety thresholds that are jointly absurd must be rejected.

``SafetyConfig`` carries 59 fields and, until now, zero cross-field validators
— in the one module where a silently-wrong combination is most costly. Every
field bound is per-field, so a swapped pair passes validation and only
misbehaves at runtime: with ``gpu_warn_temp_c`` above ``gpu_critical_temp_c``
the monitor warns at the higher temperature and criticals at the lower, and
nothing tells the operator the config is inverted.

``UltrasonicConfig`` had the same shape of gap on its GPIO pins: bare ``int``
fields accepting ``-5``, ``9999``, or the same pin twice. The symptom on
hardware is a sensor that never returns a reading, which reads as a dead
sensor rather than a config error.

Companion file: ``test_safety_threshold_ordering_backwards_compat.py`` pins
that every shipped overlay still loads unchanged.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mousedroid.config.schema.hardware import UltrasonicConfig
from mousedroid.config.schema.reward_safety import SafetyConfig


class TestGPUThresholdOrdering:
    """``gpu_warn_temp_c`` must sit below ``gpu_critical_temp_c``."""

    def test_swapped_gpu_thresholds_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"gpu_warn_temp_c.*must be below"):
            SafetyConfig(gpu_warn_temp_c=90.0, gpu_critical_temp_c=75.0)

    def test_equal_gpu_thresholds_are_rejected(self) -> None:
        """Equal thresholds give the monitor no band to warn in."""
        with pytest.raises(ValidationError, match=r"gpu_warn_temp_c.*must be below"):
            SafetyConfig(gpu_warn_temp_c=85.0, gpu_critical_temp_c=85.0)

    def test_correctly_ordered_gpu_thresholds_are_accepted(self) -> None:
        cfg = SafetyConfig(gpu_warn_temp_c=70.0, gpu_critical_temp_c=95.0)
        assert cfg.gpu_warn_temp_c < cfg.gpu_critical_temp_c


class TestBatteryThresholdOrdering:
    """Battery thresholds ascend: implausible-floor < critical < warn."""

    def test_critical_above_warn_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"battery_critical_v.*must be below"):
            SafetyConfig(battery_critical_v=11.0, battery_warn_v=10.5)

    def test_implausible_floor_above_critical_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"battery_implausible_below_v.*must be below"):
            SafetyConfig(battery_implausible_below_v=10.0, battery_critical_v=9.5)

    @pytest.mark.parametrize(
        ("disabled_field", "overrides"),
        [
            ("battery_warn_v", {"battery_warn_v": 0.0}),
            ("battery_critical_v", {"battery_critical_v": 0.0}),
            ("battery_implausible_below_v", {"battery_implausible_below_v": 0.0}),
        ],
    )
    def test_a_disabled_threshold_is_skipped_not_ordered(
        self, disabled_field: str, overrides: dict[str, float]
    ) -> None:
        """``0 disables`` is documented on each field and must keep working.

        Forcing a disabled threshold into the ordering would make switching one
        off unsatisfiable, breaking existing YAML that does exactly that.
        """
        cfg = SafetyConfig(**overrides)
        assert getattr(cfg, disabled_field) == 0.0

    def test_all_battery_thresholds_disabled_is_accepted(self) -> None:
        cfg = SafetyConfig(
            battery_warn_v=0.0,
            battery_critical_v=0.0,
            battery_implausible_below_v=0.0,
        )
        assert cfg.battery_warn_v == 0.0


class TestUltrasonicPinBounds:
    """GPIO pins must be legal BCM numbers, and must differ from each other."""

    @pytest.mark.parametrize("bad_pin", [-5, -1, 28, 9999])
    def test_out_of_range_pins_are_rejected(self, bad_pin: int) -> None:
        with pytest.raises(ValidationError):
            UltrasonicConfig(trigger_pin=bad_pin, echo_pin=24)

    def test_identical_nonzero_pins_are_rejected(self) -> None:
        """A trigger and echo on one pin yields a sensor that never answers."""
        with pytest.raises(ValidationError, match="must differ"):
            UltrasonicConfig(trigger_pin=23, echo_pin=23)

    def test_the_zero_zero_unwired_sentinel_is_accepted(self) -> None:
        """``0/0`` means "no ultrasonic wired" and must keep loading.

        Mock-mode fixtures and factory tests have long used this pair. Treating
        it as a duplicate-pin misconfiguration would break existing configs
        (CLAUDE.md invariant 6) while catching nothing real — two pins at 0 is
        an absence, not a collision.
        """
        cfg = UltrasonicConfig(trigger_pin=0, echo_pin=0)
        assert cfg.trigger_pin == cfg.echo_pin == 0

    def test_valid_distinct_pins_are_accepted(self) -> None:
        cfg = UltrasonicConfig(trigger_pin=23, echo_pin=24)
        assert cfg.trigger_pin != cfg.echo_pin

    @pytest.mark.parametrize("edge_pin", [0, 27])
    def test_bcm_range_is_inclusive_at_both_ends(self, edge_pin: int) -> None:
        other = 27 if edge_pin == 0 else 1
        cfg = UltrasonicConfig(trigger_pin=edge_pin, echo_pin=other)
        assert cfg.trigger_pin == edge_pin
