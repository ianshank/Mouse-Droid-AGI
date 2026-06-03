"""Integration: LLM-gateway observability threaded through the factory.

Verifies the shared :class:`MetricsRegistry` reaches the concrete gateway via
``build_orchestrator`` (the real wiring path the rover uses), and that a faked
end-to-end translation populates all four metric families. The ``anthropic``
SDK is faked end-to-end so no network / API key is required — mirroring
``tests/integration/test_anthropic_gateway_wiring.py``.
"""

from __future__ import annotations

import json
import types
from typing import Any

import pytest

from mousedroid.config.schema import MetricsConfig, Settings
from mousedroid.factory import build_llm_gateway, build_orchestrator
from mousedroid.llm_gateway.anthropic_gateway import AnthropicLLMGateway
from mousedroid.llm_gateway.fallback_gateway import FallbackLLMGateway
from mousedroid.telemetry.metrics import MetricsRegistry


# --------------------------------------------------------------------------- #
# Fake anthropic SDK (object response WITH usage, mirrors the live Message)
# --------------------------------------------------------------------------- #
class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(self, text: str, usage: _FakeUsage) -> None:
        self.content = [_FakeBlock(text)]
        self.usage = usage


class _FakeMessages:
    def __init__(self, text: str, usage: _FakeUsage) -> None:
        self._text = text
        self._usage = usage

    async def create(self, **_kwargs: Any) -> _FakeResponse:
        return _FakeResponse(self._text, self._usage)


def _make_sdk(reply: dict[str, float], usage: _FakeUsage) -> types.SimpleNamespace:
    text = json.dumps(reply)

    class _FakeAsyncClient:
        def __init__(self, **_kwargs: Any) -> None:
            self.messages = _FakeMessages(text, usage)

    sdk = types.SimpleNamespace()
    sdk.AsyncAnthropic = _FakeAsyncClient  # type: ignore[attr-defined]
    return sdk


def _anthropic_cfg() -> Settings:
    cfg = Settings(mock_hardware=True)
    cfg.llm.enabled = True
    cfg.llm.backend = "anthropic"
    cfg.llm.model_name = "claude-haiku-4-5"
    return cfg


# --------------------------------------------------------------------------- #
# build_orchestrator threads the shared registry into the gateway
# --------------------------------------------------------------------------- #
def test_build_orchestrator_threads_registry_into_gateway() -> None:
    orch = build_orchestrator(_anthropic_cfg())
    gateway = orch._llm_gateway
    assert isinstance(gateway, AnthropicLLMGateway)
    assert isinstance(gateway._metrics, MetricsRegistry)


def test_build_orchestrator_threads_registry_into_composite_primary() -> None:
    cfg = _anthropic_cfg()
    cfg.llm.fallback_backend = "llama_cpp"
    orch = build_orchestrator(cfg)
    gateway = orch._llm_gateway
    assert isinstance(gateway, FallbackLLMGateway)
    assert isinstance(gateway._metrics, MetricsRegistry)
    assert gateway._metrics is gateway._primary._metrics  # one shared registry


# --------------------------------------------------------------------------- #
# Faked translate populates every family through the factory-built gateway
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_faked_translate_populates_all_families() -> None:
    cfg = _anthropic_cfg()
    reg = MetricsRegistry(MetricsConfig())
    gateway = build_llm_gateway(cfg, metrics=reg)
    assert isinstance(gateway, AnthropicLLMGateway)
    gateway._sdk = _make_sdk(  # type: ignore[attr-defined]
        {"vx": 0.6, "vy": 0.0, "omega": 0.2}, _FakeUsage(120, 40)
    )
    await gateway.start()
    await gateway.translate_mission("navigate to the cantina")
    await gateway.stop()

    out = reg.render_prometheus()
    assert 'mousedroid_llm_tokens_total{model="claude-haiku-4-5",token_type="input"} 120' in out
    assert 'mousedroid_llm_tokens_total{model="claude-haiku-4-5",token_type="output"} 40' in out
    assert "mousedroid_llm_gateway_latency_ms_count 1" in out


@pytest.mark.asyncio
async def test_composite_records_primary_ok_served() -> None:
    cfg = _anthropic_cfg()
    cfg.llm.fallback_backend = "llama_cpp"
    reg = MetricsRegistry(MetricsConfig())
    gateway = build_llm_gateway(cfg, metrics=reg)
    assert isinstance(gateway, FallbackLLMGateway)
    gateway._primary._sdk = _make_sdk(  # type: ignore[attr-defined]
        {"vx": 0.9, "vy": 0.0, "omega": 0.0}, _FakeUsage(50, 10)
    )
    await gateway.start()
    await gateway.translate_mission("dash forward")
    await gateway.stop()

    out = reg.render_prometheus()
    assert 'mousedroid_llm_gateway_served_total{tier="primary",outcome="ok"} 1' in out
