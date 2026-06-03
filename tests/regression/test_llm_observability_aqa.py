"""AQA: schema-field hygiene + protocol conformance for LLM-gateway observability.

Architectural-quality assertions that lock the contracts a future refactor
could silently break: field descriptions/defaults, registry helper names
(rename guard), the keyword-only ``metrics`` parameter, protocol conformance of
the instrumented gateway, and the fixed low-cardinality label value sets.
"""

from __future__ import annotations

import inspect

from mousedroid.config.schema import MetricsConfig, Settings
from mousedroid.factory import build_llm_gateway
from mousedroid.llm_gateway.anthropic_gateway import AnthropicLLMGateway
from mousedroid.llm_gateway.protocol import LLMGatewayProtocol
from mousedroid.telemetry.metrics import MetricsRegistry


def test_metrics_fields_documented() -> None:
    fields = MetricsConfig.model_fields
    for name in ("track_llm_gateway", "llm_gateway_latency_buckets_ms"):
        assert name in fields
        assert fields[name].description, f"{name} must carry an operator description"


def test_registry_helpers_exist_rename_guard() -> None:
    for helper in (
        "inc_llm_tokens",
        "observe_llm_gateway_latency_ms",
        "inc_llm_gateway_served",
        "inc_llm_latency_budget_exceeded",
    ):
        assert callable(getattr(MetricsRegistry, helper, None)), helper


def test_anthropic_gateway_satisfies_protocol() -> None:
    cfg = Settings(mock_hardware=True)
    cfg.llm.backend = "anthropic"
    cfg.llm.model_name = "claude-haiku-4-5"
    gw = build_llm_gateway(cfg, metrics=MetricsRegistry(MetricsConfig()))
    assert isinstance(gw, AnthropicLLMGateway)
    assert isinstance(gw, LLMGatewayProtocol)


def test_build_llm_gateway_metrics_keyword_only_none_default() -> None:
    param = inspect.signature(build_llm_gateway).parameters["metrics"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is None


def test_served_counter_label_value_sets_fixed() -> None:
    """Render reflects only the fixed low-cardinality tier/outcome enums."""
    reg = MetricsRegistry(MetricsConfig())
    for tier in ("primary", "secondary"):
        for outcome in ("ok", "degraded"):
            reg.inc_llm_gateway_served(tier, outcome)
    out = reg.render_prometheus()
    assert out.count("llm_gateway_served_total{") == 4  # exactly the 2x2 grid
