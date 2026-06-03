"""E2E: LLM-gateway metrics surface on the real ``/metrics`` HTTP endpoint.

Spins up the real :class:`TelemetryServer` in-process (aiohttp ``TestServer``),
shares one :class:`MetricsRegistry` with a factory-built :class:`AnthropicLLMGateway`
(``anthropic`` SDK faked end-to-end), runs a single translation, then GETs the
server's configured ``cfg.telemetry.metrics_path`` and asserts every new family
is exposed. A *pre*-translate scrape must omit them (no cardinality leak before
first write) — the byte-identical-when-unused contract observed over HTTP.
"""

from __future__ import annotations

import asyncio
import json
import types
from typing import Any

import pytest

from mousedroid.config.schema import Settings
from mousedroid.telemetry.metrics import MetricsRegistry
from mousedroid.telemetry.protocol import TelemetryFrame

aiohttp = pytest.importorskip("aiohttp")

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from mousedroid.factory import build_llm_gateway
from mousedroid.llm_gateway.anthropic_gateway import AnthropicLLMGateway
from mousedroid.telemetry.server import TelemetryServer

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------- #
# Faked anthropic SDK (object response carrying usage)
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


def _build_server(cfg: Settings, registry: MetricsRegistry) -> TelemetryServer:
    queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=8)
    from unittest.mock import AsyncMock

    health = AsyncMock()
    health.check_health = AsyncMock(return_value={"status": "ok"})
    return TelemetryServer(
        cfg=cfg.telemetry,
        telemetry_queue=queue,
        health_monitor=health,
        metrics_registry=registry,
        metrics_path=cfg.telemetry.metrics_path,
    )


async def _anthropic_gateway(registry: MetricsRegistry) -> AnthropicLLMGateway:
    cfg = Settings(mock_hardware=True)
    cfg.llm.enabled = True
    cfg.llm.backend = "anthropic"
    cfg.llm.model_name = "claude-haiku-4-5"
    gateway = build_llm_gateway(cfg, metrics=registry)
    assert isinstance(gateway, AnthropicLLMGateway)
    gateway._sdk = _make_sdk(  # type: ignore[attr-defined]
        {"vx": 0.6, "vy": 0.0, "omega": 0.2}, _FakeUsage(120, 40)
    )
    await gateway.start()
    return gateway


async def test_metrics_endpoint_exposes_families_after_translate() -> None:
    cfg = Settings(mock_hardware=True)
    registry = MetricsRegistry(cfg.metrics)
    server = _build_server(cfg, registry)
    app = web.Application(middlewares=server._build_middlewares())
    server._register_routes(app)
    metrics_path = cfg.telemetry.metrics_path

    async with TestClient(TestServer(app)) as client:
        # Pre-translate scrape: new families must be absent.
        pre = await (await client.get(metrics_path)).text()
        for name in (
            "llm_tokens_total",
            "llm_gateway_latency_ms",
            "llm_gateway_served_total",
        ):
            assert name not in pre

        gateway = await _anthropic_gateway(registry)
        await gateway.translate_mission("navigate to the cantina")
        await gateway.stop()

        resp = await client.get(metrics_path)
        assert resp.status == 200
        body = await resp.text()

    assert 'mousedroid_llm_tokens_total{model="claude-haiku-4-5",token_type="input"} 120' in body
    assert 'mousedroid_llm_tokens_total{model="claude-haiku-4-5",token_type="output"} 40' in body
    assert "mousedroid_llm_gateway_latency_ms_count 1" in body


async def test_metrics_endpoint_exposes_budget_counter_when_slow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = Settings(mock_hardware=True)
    registry = MetricsRegistry(cfg.metrics)
    server = _build_server(cfg, registry)
    app = web.Application(middlewares=server._build_middlewares())
    server._register_routes(app)
    metrics_path = cfg.telemetry.metrics_path

    # The faked SDK round-trips faster than the monotonic clock resolution, so
    # elapsed_ms would be ~0 and never exceed the budget. Advance the gateway's
    # clock 1000 s per call so every consecutive start/end pair = 1000 s.
    from mousedroid.llm_gateway import anthropic_gateway as gw_mod

    state = {"t": 0.0}

    def _fake_monotonic() -> float:
        state["t"] += 1000.0
        return state["t"]

    monkeypatch.setattr(gw_mod.time, "monotonic", _fake_monotonic)

    async with TestClient(TestServer(app)) as client:
        llm_cfg = Settings(mock_hardware=True)
        llm_cfg.llm.enabled = True
        llm_cfg.llm.backend = "anthropic"
        llm_cfg.llm.model_name = "claude-haiku-4-5"
        llm_cfg.llm.latency_target_ms = 0.0001  # any real round-trip exceeds this
        gateway = build_llm_gateway(llm_cfg, metrics=registry)
        assert isinstance(gateway, AnthropicLLMGateway)
        gateway._sdk = _make_sdk(  # type: ignore[attr-defined]
            {"vx": 0.1, "vy": 0.0, "omega": 0.0}, _FakeUsage(10, 5)
        )
        await gateway.start()
        await gateway.translate_mission("creep forward")
        await gateway.stop()
        # Restore the real clock before the HTTP GET — the jumping monotonic
        # would otherwise break aiohttp's connection-timeout machinery.
        monkeypatch.undo()

        body = await (await client.get(metrics_path)).text()

    assert 'mousedroid_llm_latency_budget_exceeded_total{model="claude-haiku-4-5"} 1' in body
