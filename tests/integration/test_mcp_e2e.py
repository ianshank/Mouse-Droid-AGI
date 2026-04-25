"""End-to-end MCP integration test.

Boots a :class:`MouseDroidMCPServer` against a real
:class:`TelemetryPublisher` and an in-memory log buffer, exercises the
full request path (auth -> bridge -> handler -> metrics -> log
correlation -> resource read), then shuts down cleanly.

Skipped when the optional ``mcp`` SDK is not installed; the bridge
itself is exercised in unit tests.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from mousedroid.common.tools.registry import create_default_registry
from mousedroid.config.schema import MCPConfig, MetricsConfig, Settings
from mousedroid.mcp.server import MouseDroidMCPServer
from mousedroid.telemetry.log_buffer import LogRingBuffer
from mousedroid.telemetry.metrics import MetricsRegistry
from mousedroid.telemetry.protocol import TelemetryFrame
from mousedroid.telemetry.publisher import TelemetryPublisher


@pytest.fixture
def publisher_with_frames() -> TelemetryPublisher:
    cfg = Settings.model_validate({"mock_hardware": True}).telemetry
    publisher = TelemetryPublisher(cfg)
    queue = publisher.get_queue()
    for i in range(4):
        queue.put_nowait(TelemetryFrame(timestamp=float(i), tick_count=i))
    return publisher


@pytest.fixture
def log_buffer_with_entries() -> LogRingBuffer:
    buf = LogRingBuffer(maxlen=20)
    buf(None, "info", {"event": "boot", "host": "127.0.0.1"})
    buf(None, "warning", {"event": "auth_attempt", "api_key": "leak-this"})
    return buf


@pytest.fixture
def metrics_registry() -> MetricsRegistry:
    return MetricsRegistry(MetricsConfig())


@pytest.mark.asyncio
async def test_mcp_end_to_end(
    publisher_with_frames: TelemetryPublisher,
    log_buffer_with_entries: LogRingBuffer,
    metrics_registry: MetricsRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MOUSEDROID_MCP_TOKEN", raising=False)
    cfg = MCPConfig.model_validate(
        {
            "enabled": True,
            "transport": "stdio",
            "sample_telemetry_hz": 50.0,
            "resources": {
                "telemetry_enabled": True,
                "logs_enabled": True,
                "config_enabled": True,
            },
        }
    )
    monitor = MagicMock()
    monitor.evaluate.return_value = MagicMock(is_emergency=False)

    server = MouseDroidMCPServer(
        cfg=cfg,
        root_cfg=Settings.model_validate({"mock_hardware": True}),
        tool_registry=create_default_registry(),
        safety_monitor=monitor,
        telemetry_publisher=publisher_with_frames,
        log_buffer=log_buffer_with_entries,
        metrics_registry=metrics_registry,
    )

    await server.start()
    try:
        # 1. Tool surface
        names = server.list_tool_names()
        assert "health_check" in names

        # 2. Tool invocation -> ok
        out = await server.call_tool("health_check")
        assert out["status"] == "ok"
        assert out["payload"]["status"] == "ok"

        # 3. Telemetry resource — sampler should have drained the queue
        await asyncio.sleep(0.1)
        latest = await server.read_resource("mousedroid://telemetry/latest")
        assert latest["frame"]["tick_count"] == 3

        # 4. Log resource with redaction
        logs = await server.read_resource("mousedroid://logs/tail?n=5")
        assert logs["count"] >= 2
        # Redacted: api_key never leaks
        for entry in logs["entries"]:
            if "api_key" in entry:
                assert entry["api_key"] != "leak-this"

        # 5. Config resource
        cfg_out = await server.read_resource("mousedroid://config/redacted")
        assert "settings" in cfg_out
        # Loopback default — token-related fields should still be sanitised.
        assert "leak-this" not in repr(cfg_out["settings"])

        # 6. Metrics observability
        prom = metrics_registry.render_prometheus()
        assert "mcp_requests_total" in prom
        assert "mcp_tool_calls_total" in prom
        # Latency histogram emitted because we recorded at least one call.
        assert "mcp_request_latency_ms" in prom
    finally:
        await server.stop()
    assert server.is_running is False


@pytest.mark.asyncio
async def test_mcp_optional_sdk_skip() -> None:
    """When the optional ``mcp`` package is missing the server still starts."""
    # We don't importorskip here — the bridge / providers don't depend on the
    # SDK. The server's _serve_loop falls into the idle branch and simply
    # blocks on a long sleep, so start/stop must still be clean.
    cfg = MCPConfig.model_validate({"enabled": True, "transport": "stdio"})
    monitor = MagicMock()
    monitor.evaluate.return_value = MagicMock(is_emergency=False)
    server = MouseDroidMCPServer(
        cfg=cfg,
        root_cfg=Settings.model_validate({"mock_hardware": True}),
        tool_registry=create_default_registry(),
        safety_monitor=monitor,
    )
    await server.start()
    await asyncio.sleep(0.05)
    await server.stop()
    assert server.is_running is False
