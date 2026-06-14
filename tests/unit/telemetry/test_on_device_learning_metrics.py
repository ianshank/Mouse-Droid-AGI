"""Unit tests for the on-device-learning revert counter (Phase 6 WS1).

Mirrors the PR-#115 LLM-counter pure-add pattern: family absent until first
write, gated behind ``track_on_device_learning``, low-cardinality ``reason``
label guarded against a module-level frozenset (out-of-set values dropped).
"""

from __future__ import annotations

from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry import metrics as metrics_mod
from mousedroid.telemetry.metrics import MetricsRegistry, generate_metrics_sample


def _registry(**overrides: object) -> MetricsRegistry:
    return MetricsRegistry(MetricsConfig(**overrides))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Pure-add: family absent until first write
# --------------------------------------------------------------------------- #
def test_family_absent_until_written() -> None:
    assert "on_device_learning_reverted_total" not in _registry().render_prometheus()


# --------------------------------------------------------------------------- #
# Counter increments per reason
# --------------------------------------------------------------------------- #
def test_counter_renders_per_reason() -> None:
    reg = _registry()
    reg.inc_on_device_learning_reverted("regression_bound")
    reg.inc_on_device_learning_reverted("regression_bound")
    reg.inc_on_device_learning_reverted("integrity_mismatch")
    out = reg.render_prometheus()
    assert 'mousedroid_on_device_learning_reverted_total{reason="regression_bound"} 2' in out
    assert 'mousedroid_on_device_learning_reverted_total{reason="integrity_mismatch"} 1' in out


def test_counter_namespaced() -> None:
    reg = _registry(namespace="rover")
    reg.inc_on_device_learning_reverted("exception")
    assert "rover_on_device_learning_reverted_total" in reg.render_prometheus()


def test_counter_noop_on_nonpositive() -> None:
    reg = _registry()
    reg.inc_on_device_learning_reverted("regression_bound", amount=0)
    assert "on_device_learning_reverted_total" not in reg.render_prometheus()


# --------------------------------------------------------------------------- #
# Label-cardinality hygiene: out-of-set reason dropped (no new series)
# --------------------------------------------------------------------------- #
def test_counter_drops_invalid_reason() -> None:
    reg = _registry()
    reg.inc_on_device_learning_reverted("mission text leaked here")  # free text
    assert "on_device_learning_reverted_total" not in reg.render_prometheus()


def test_counter_accepts_full_valid_reason_set() -> None:
    reg = _registry()
    for reason in ("regression_bound", "integrity_mismatch", "exception"):
        reg.inc_on_device_learning_reverted(reason)
    assert reg.render_prometheus().count("on_device_learning_reverted_total{") == 3


def test_reason_constant_set_is_pinned() -> None:
    assert (
        frozenset({"regression_bound", "integrity_mismatch", "exception"})
        == metrics_mod._ON_DEVICE_REVERT_REASONS
    )


# --------------------------------------------------------------------------- #
# track_on_device_learning flag gates the family
# --------------------------------------------------------------------------- #
def test_track_flag_off_suppresses_family() -> None:
    reg = _registry(track_on_device_learning=False)
    reg.inc_on_device_learning_reverted("regression_bound")
    assert "on_device_learning_reverted_total" not in reg.render_prometheus()


# --------------------------------------------------------------------------- #
# promtool seeding contract
# --------------------------------------------------------------------------- #
def test_generate_metrics_sample_seeds_family() -> None:
    assert "on_device_learning_reverted_total" in generate_metrics_sample()
