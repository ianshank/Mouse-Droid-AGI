"""Smoke: sub-second sanity for LLM-gateway observability.

Cheapest regression net — imports the touched modules, round-trips the new
``MetricsConfig`` fields through YAML, and confirms an in-process writer renders
all four family names. Selected first in CI via ``-m smoke``.
"""

from __future__ import annotations

import math

import pytest
import yaml

from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry.metrics import MetricsRegistry

pytestmark = pytest.mark.smoke


def test_modules_import() -> None:
    import mousedroid.factory
    import mousedroid.llm_gateway.anthropic_gateway
    import mousedroid.llm_gateway.fallback_gateway
    import mousedroid.telemetry.metrics  # noqa: F401


def test_metrics_fields_round_trip_through_yaml() -> None:
    raw = yaml.safe_load(
        yaml.safe_dump(
            {
                "track_llm_gateway": False,
                "llm_gateway_latency_buckets_ms": [10.0, 100.0, 1000.0, float("inf")],
            }
        )
    )
    cfg = MetricsConfig.model_validate(raw)
    assert cfg.track_llm_gateway is False
    assert cfg.llm_gateway_latency_buckets_ms[-1] == math.inf


def test_writer_renders_all_four_families() -> None:
    reg = MetricsRegistry(MetricsConfig())
    reg.inc_llm_tokens("claude-haiku-4-5", "input", 7)
    reg.observe_llm_gateway_latency_ms(123.0)
    reg.inc_llm_gateway_served("primary", "ok")
    reg.inc_llm_latency_budget_exceeded("claude-haiku-4-5")
    out = reg.render_prometheus()
    for name in (
        "mousedroid_llm_tokens_total",
        "mousedroid_llm_gateway_latency_ms",
        "mousedroid_llm_gateway_served_total",
        "mousedroid_llm_latency_budget_exceeded_total",
    ):
        assert name in out
