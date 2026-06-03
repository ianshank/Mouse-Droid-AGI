"""Unit tests for the LLM-gateway observability metric families.

Exercises the MetricsRegistry helpers added for the deliberative Claude tier:
token-usage counter, round-trip latency histogram, per-tier served counter,
and the latency-budget-exceeded counter. Pure registry-level tests (no gateway
/ SDK) following the render-assertion style of ``test_metrics_pr4.py``.
"""

from __future__ import annotations

import math

import pytest

from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry.metrics import MetricsRegistry


def _registry(**overrides: object) -> MetricsRegistry:
    return MetricsRegistry(MetricsConfig(**overrides))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Pure-add: families absent until first write
# --------------------------------------------------------------------------- #
def test_families_absent_until_written() -> None:
    """A pristine registry renders none of the LLM-gateway families."""
    out = _registry().render_prometheus()
    for name in (
        "llm_tokens_total",
        "llm_gateway_latency_ms",
        "llm_gateway_served_total",
        "llm_latency_budget_exceeded_total",
    ):
        assert name not in out


# --------------------------------------------------------------------------- #
# Token-usage counter
# --------------------------------------------------------------------------- #
def test_token_counter_renders_per_direction() -> None:
    reg = _registry()
    reg.inc_llm_tokens("claude-haiku-4-5", "input", 120)
    reg.inc_llm_tokens("claude-haiku-4-5", "output", 40)
    out = reg.render_prometheus()
    assert 'mousedroid_llm_tokens_total{model="claude-haiku-4-5",token_type="input"} 120' in out
    assert 'mousedroid_llm_tokens_total{model="claude-haiku-4-5",token_type="output"} 40' in out


@pytest.mark.parametrize("amount", [0, -5, None])
def test_token_counter_noop_on_nonpositive_or_none(amount: int | None) -> None:
    reg = _registry()
    reg.inc_llm_tokens("m", "input", amount)
    assert "llm_tokens_total" not in reg.render_prometheus()


def test_token_counter_namespaced() -> None:
    reg = _registry(namespace="rover")
    reg.inc_llm_tokens("m", "input", 1)
    assert "rover_llm_tokens_total" in reg.render_prometheus()


# --------------------------------------------------------------------------- #
# Latency histogram
# --------------------------------------------------------------------------- #
def test_latency_histogram_records_and_buckets() -> None:
    reg = _registry()
    reg.observe_llm_gateway_latency_ms(180.0)
    out = reg.render_prometheus()
    assert "mousedroid_llm_gateway_latency_ms_count 1" in out
    assert 'mousedroid_llm_gateway_latency_ms_bucket{le="+Inf"} 1' in out
    assert "mousedroid_llm_gateway_latency_ms_sum 180" in out


def test_latency_histogram_omitted_when_zero_count() -> None:
    """The histogram is NOT rendered until an observation lands (count>0 guard)."""
    assert "llm_gateway_latency_ms" not in _registry().render_prometheus()


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_latency_histogram_drops_invalid_samples(bad: float) -> None:
    reg = _registry()
    reg.observe_llm_gateway_latency_ms(bad)
    assert "llm_gateway_latency_ms" not in reg.render_prometheus()


# --------------------------------------------------------------------------- #
# Per-tier served counter
# --------------------------------------------------------------------------- #
def test_served_counter_labels() -> None:
    reg = _registry()
    reg.inc_llm_gateway_served("primary", "ok")
    reg.inc_llm_gateway_served("primary", "ok")
    reg.inc_llm_gateway_served("secondary", "degraded")
    out = reg.render_prometheus()
    assert 'mousedroid_llm_gateway_served_total{tier="primary",outcome="ok"} 2' in out
    assert 'mousedroid_llm_gateway_served_total{tier="secondary",outcome="degraded"} 1' in out


def test_served_counter_noop_on_nonpositive() -> None:
    reg = _registry()
    reg.inc_llm_gateway_served("primary", "ok", amount=0)
    assert "llm_gateway_served_total" not in reg.render_prometheus()


# --------------------------------------------------------------------------- #
# Budget-exceeded counter
# --------------------------------------------------------------------------- #
def test_budget_counter() -> None:
    reg = _registry()
    reg.inc_llm_latency_budget_exceeded("claude-haiku-4-5")
    assert (
        'mousedroid_llm_latency_budget_exceeded_total{model="claude-haiku-4-5"} 1'
        in reg.render_prometheus()
    )


# --------------------------------------------------------------------------- #
# track_llm_gateway flag gates all four families
# --------------------------------------------------------------------------- #
def test_track_flag_off_suppresses_all_families() -> None:
    reg = _registry(track_llm_gateway=False)
    reg.inc_llm_tokens("m", "input", 5)
    reg.observe_llm_gateway_latency_ms(100.0)
    reg.inc_llm_gateway_served("primary", "ok")
    reg.inc_llm_latency_budget_exceeded("m")
    out = reg.render_prometheus()
    for name in (
        "llm_tokens_total",
        "llm_gateway_latency_ms",
        "llm_gateway_served_total",
        "llm_latency_budget_exceeded_total",
    ):
        assert name not in out


# --------------------------------------------------------------------------- #
# Label-cardinality hygiene: out-of-set values are dropped (not new series)
# --------------------------------------------------------------------------- #
def test_token_counter_drops_invalid_token_type() -> None:
    reg = _registry()
    reg.inc_llm_tokens("claude-haiku-4-5", "inpt", 10)  # typo
    assert "llm_tokens_total" not in reg.render_prometheus()


def test_served_counter_drops_invalid_labels() -> None:
    reg = _registry()
    reg.inc_llm_gateway_served("tertiary", "ok")  # invalid tier
    reg.inc_llm_gateway_served("primary", "kinda-ok")  # invalid outcome
    assert "llm_gateway_served_total" not in reg.render_prometheus()


def test_served_counter_accepts_full_valid_grid() -> None:
    reg = _registry()
    for tier in ("primary", "secondary"):
        for outcome in ("ok", "degraded"):
            reg.inc_llm_gateway_served(tier, outcome)
    assert reg.render_prometheus().count("llm_gateway_served_total{") == 4
