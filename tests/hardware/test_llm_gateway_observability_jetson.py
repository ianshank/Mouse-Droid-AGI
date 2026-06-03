"""Hardware: LLM-gateway observability against the live Claude API on the rover.

Double-gated — runs only on a Jetson host (:func:`is_jetson_host`) AND when
``ANTHROPIC_API_KEY`` is present. A real cloud translation must populate the
latency histogram and token counters; and a normal cloud round-trip must NOT
trip the 5000 ms budget (regression guard on the ``latency_target_ms``
calibration documented in CLAUDE.md). This test builds a single, non-composite
:class:`AnthropicLLMGateway`, so it deliberately does NOT assert the per-tier
served counter (``llm_gateway_served_total`` is emitted only by the
:class:`FallbackLLMGateway` composite).
"""

from __future__ import annotations

import os

import pytest

from mousedroid.config.schema import MetricsConfig, Settings
from mousedroid.factory import build_llm_gateway
from mousedroid.llm_gateway.anthropic_gateway import AnthropicLLMGateway
from mousedroid.telemetry.metrics import MetricsRegistry
from tests._jetson_hardware import is_jetson_host

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not is_jetson_host(), reason="Jetson-only hardware test"),
    pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"),
]


def _live_cfg() -> Settings:
    cfg = Settings()
    cfg.llm.enabled = True
    cfg.llm.backend = "anthropic"
    cfg.llm.model_name = os.getenv("MOUSEDROID_LLM_MODEL", "claude-haiku-4-5")
    cfg.llm.latency_target_ms = 5000.0  # documented cloud round-trip budget
    return cfg


@pytest.mark.asyncio
async def test_live_translate_records_observability() -> None:
    cfg = _live_cfg()
    reg = MetricsRegistry(MetricsConfig())
    gateway = build_llm_gateway(cfg, metrics=reg)
    assert isinstance(gateway, AnthropicLLMGateway)
    await gateway.start()
    try:
        await gateway.translate_mission("move forward slowly")
    finally:
        await gateway.stop()

    out = reg.render_prometheus()
    assert "mousedroid_llm_gateway_latency_ms_count 1" in out
    assert "mousedroid_llm_tokens_total" in out
    # A healthy cloud round-trip must stay within the 5000 ms budget — the
    # budget counter (recorded only when elapsed_ms exceeds the target) is
    # therefore absent. (Single non-composite gateway → no served counter.)
    assert "mousedroid_llm_latency_budget_exceeded_total" not in out
