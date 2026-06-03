"""Regression: LLM-gateway observability is purely additive + backwards-compatible.

Guards the CLAUDE.md invariant "New config fields MUST have defaults; existing
YAML files must load unchanged" and the plan's byte-identical-when-unused
contract: a registry that never serves an LLM translation renders none of the
new families, and ``build_llm_gateway(cfg)`` (no ``metrics`` kwarg) still works.
"""

from __future__ import annotations

import math

from mousedroid.config.schema import MetricsConfig, Settings
from mousedroid.factory import build_llm_gateway
from mousedroid.llm_gateway.anthropic_gateway import AnthropicLLMGateway
from mousedroid.telemetry.metrics import MetricsRegistry


def test_new_metrics_fields_have_safe_defaults() -> None:
    cfg = MetricsConfig()
    assert cfg.track_llm_gateway is True
    buckets = cfg.llm_gateway_latency_buckets_ms
    assert isinstance(buckets, tuple)
    assert buckets[-1] == math.inf  # +Inf overflow bucket
    assert all(isinstance(b, float) for b in buckets)


def test_pre_feature_metrics_yaml_loads_unchanged() -> None:
    """A ``metrics:`` block predating these fields still validates."""
    legacy = MetricsConfig.model_validate({"enabled": True, "path": "/metrics"})
    assert legacy.track_llm_gateway is True  # default filled in
    assert legacy.llm_gateway_latency_buckets_ms[-1] == math.inf


def test_metrics_endpoint_byte_identical_when_llm_unused() -> None:
    """A registry with no LLM writes renders none of the new families."""
    out = MetricsRegistry(MetricsConfig()).render_prometheus()
    for name in (
        "llm_tokens_total",
        "llm_gateway_latency_ms",
        "llm_gateway_served_total",
        "llm_latency_budget_exceeded_total",
    ):
        assert name not in out


def test_build_llm_gateway_without_metrics_kwarg_still_works() -> None:
    cfg = Settings(mock_hardware=True)
    cfg.llm.backend = "anthropic"
    cfg.llm.model_name = "claude-haiku-4-5"
    gw = build_llm_gateway(cfg)  # legacy call site — no metrics
    assert isinstance(gw, AnthropicLLMGateway)
    assert gw._metrics is None


def test_track_flag_off_yaml_round_trips() -> None:
    cfg = MetricsConfig.model_validate({"track_llm_gateway": False})
    assert cfg.track_llm_gateway is False
    reg = MetricsRegistry(cfg)
    reg.inc_llm_tokens("m", "input", 5)
    assert "llm_tokens_total" not in reg.render_prometheus()
