"""Unit tests for commentary metric families (pure-add + label hygiene)."""

from __future__ import annotations

from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry.metrics import MetricsRegistry


def _reg() -> MetricsRegistry:
    return MetricsRegistry(MetricsConfig())


def test_families_absent_until_first_write() -> None:
    out = _reg().render_prometheus()
    assert "commentary_emitted" not in out
    assert "commentary_suppressed" not in out
    assert "commentary_compose_seconds" not in out


def test_emitted_and_novelty_render_after_write() -> None:
    reg = _reg()
    reg.inc_commentary_emitted()
    reg.set_commentary_novelty(1.5)
    out = reg.render_prometheus()
    assert "commentary_emitted_total 1" in out
    assert "commentary_novelty 1.5" in out


def test_considered_renders_after_write() -> None:
    reg = _reg()
    reg.inc_commentary_considered()
    assert "commentary_considered_total 1" in reg.render_prometheus()


def test_suppressed_label_and_render() -> None:
    reg = _reg()
    reg.inc_commentary_suppressed("cooldown")
    out = reg.render_prometheus()
    assert 'commentary_suppressed_total{reason="cooldown"} 1' in out


def test_suppressed_invalid_reason_dropped() -> None:
    reg = _reg()
    reg.inc_commentary_suppressed("totally_made_up")
    assert "commentary_suppressed" not in reg.render_prometheus()


def test_compose_histogram_drops_non_finite() -> None:
    reg = _reg()
    reg.observe_commentary_compose_seconds(float("nan"))
    reg.observe_commentary_compose_seconds(float("inf"))
    reg.observe_commentary_compose_seconds(-1.0)
    assert "commentary_compose_seconds" not in reg.render_prometheus()
    reg.observe_commentary_compose_seconds(0.05)
    assert "commentary_compose_seconds_count 1" in reg.render_prometheus()


def test_non_positive_increments_are_noops() -> None:
    reg = _reg()
    reg.inc_commentary_emitted(0)
    reg.inc_commentary_considered(-1)
    reg.inc_commentary_suppressed("cooldown", 0)
    reg.inc_commentary_recognitions(0)
    reg.inc_commentary_referents_stored(-2)
    assert "commentary" not in reg.render_prometheus()


def test_recognition_families_render_after_write() -> None:
    reg = _reg()
    assert "commentary_recognitions" not in reg.render_prometheus()
    reg.inc_commentary_recognitions()
    reg.inc_commentary_referents_stored(3)
    out = reg.render_prometheus()
    assert "commentary_recognitions_total 1" in out
    assert "commentary_referents_stored_total 3" in out


def test_recognition_cooldown_is_a_valid_suppress_reason() -> None:
    reg = _reg()
    reg.inc_commentary_suppressed("recognition_cooldown")
    assert 'reason="recognition_cooldown"' in reg.render_prometheus()


def test_novelty_nan_skipped() -> None:
    reg = _reg()
    reg.inc_commentary_emitted()
    reg.set_commentary_novelty(float("nan"))
    out = reg.render_prometheus()
    # Gauge stays at its 0.0 default (NaN skipped), still renders alongside emitted.
    assert "commentary_novelty 0" in out
