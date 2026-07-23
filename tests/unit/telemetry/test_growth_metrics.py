"""Unit tests for the growth-pillar distillation Prometheus counter.

Pins the pure-add contract: the family is omitted from ``/metrics`` until the
first write, gated by ``track_growth_distillation``, drops out-of-set labels, and
is seeded by ``generate_metrics_sample`` (promtool contract).
"""

from __future__ import annotations

from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry.metrics import MetricsRegistry, generate_metrics_sample

_FAMILY = "growth_distillations_total"


def _registry(*, track: bool = True) -> MetricsRegistry:
    return MetricsRegistry(MetricsConfig(track_growth_distillation=track))


def test_family_absent_until_written() -> None:
    """No growth family renders before the first increment."""
    assert _FAMILY not in _registry().render_prometheus()


def test_family_present_after_write() -> None:
    """A single valid increment surfaces the family with its outcome label."""
    reg = _registry()
    reg.inc_growth_distilled("completed")
    out = reg.render_prometheus()
    assert _FAMILY in out
    assert 'outcome="completed"' in out


def test_track_flag_off_suppresses_family() -> None:
    """``track_growth_distillation=False`` suppresses the family entirely."""
    reg = _registry(track=False)
    reg.inc_growth_distilled("completed")
    assert _FAMILY not in reg.render_prometheus()


def test_invalid_outcome_dropped() -> None:
    """An out-of-set outcome is dropped (no new time series)."""
    reg = _registry()
    reg.inc_growth_distilled("bogus")
    assert _FAMILY not in reg.render_prometheus()


def test_nonpositive_amount_is_noop() -> None:
    """A non-positive amount does not open the family."""
    reg = _registry()
    reg.inc_growth_distilled("completed", amount=0)
    assert _FAMILY not in reg.render_prometheus()


def test_generate_metrics_sample_seeds_family() -> None:
    """The promtool sample seeds every valid outcome series."""
    sample = generate_metrics_sample()
    assert _FAMILY in sample
    assert 'outcome="completed"' in sample
    assert 'outcome="skipped_no_batch"' in sample
