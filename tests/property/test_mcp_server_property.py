"""Property-based tests for MCP Server memory-query latency evaluation.

Tests the latency-tracking + baseline-warning logic WITHOUT mocking
``time.monotonic`` (which asyncio's event loop also calls, making a
global mock fatal). Instead we test the *post-condition*: that the
metric was observed and the warning was/was-not emitted, using real
wall-clock time from a controlled async sleep.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.config.schema import BaselinesConfig, MCPConfig, Settings
from mousedroid.mcp.server import MouseDroidMCPServer


def _build_server(
    limit_ms: float,
) -> tuple[MouseDroidMCPServer, MagicMock, MagicMock]:
    """Build a server with mocked dependencies and a given baseline limit."""
    cfg = MCPConfig.model_validate({"enabled": True})
    root_cfg = Settings.model_validate({"mock_hardware": True})
    root_cfg.baselines = BaselinesConfig(max_memory_query_latency_ms=limit_ms)

    metrics_registry = MagicMock()
    tool_registry = MagicMock()
    safety_monitor = MagicMock()
    memory_provider = MagicMock()

    # Memory read returns instantly — real latency will be ~0 ms
    memory_provider.read = AsyncMock(return_value={"episodes": []})

    server = MouseDroidMCPServer(
        cfg=cfg,
        root_cfg=root_cfg,
        tool_registry=tool_registry,
        safety_monitor=safety_monitor,
        metrics_registry=metrics_registry,
    )
    server._memory = memory_provider
    return server, metrics_registry, memory_provider


@pytest.mark.asyncio
async def test_memory_query_latency_metric_is_observed() -> None:
    """The server records a latency observation on every memory read."""
    server, metrics, _ = _build_server(limit_ms=9999.0)

    await server.read_resource(
        "mousedroid://memory/episodes/recent",
        peer="stdio",
        token=None,
    )

    metrics.observe_mcp_memory_query_latency_ms.assert_called_once()
    observed_ms: float = metrics.observe_mcp_memory_query_latency_ms.call_args[0][0]
    # Real wall-clock: should be sub-millisecond for an in-memory mock
    assert 0.0 <= observed_ms < 500.0


@pytest.mark.asyncio
async def test_memory_query_no_warning_under_limit() -> None:
    """No degradation warning when latency is under the baseline limit."""
    server, _, _ = _build_server(limit_ms=9999.0)

    with patch("mousedroid.mcp.server._log") as mock_log:
        await server.read_resource(
            "mousedroid://memory/episodes/recent",
            peer="stdio",
            token=None,
        )
        mock_log.warning.assert_not_called()


@pytest.mark.asyncio
async def test_memory_query_warning_over_limit() -> None:
    """Degradation warning fires when latency exceeds the baseline limit."""
    server, _, memory = _build_server(limit_ms=0.001)  # 1 µs — will always be exceeded

    # Make the memory read take measurable time
    async def _slow_read(*args: Any, **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(0.01)  # 10 ms
        return {"episodes": []}

    memory.read = AsyncMock(side_effect=_slow_read)

    with patch("mousedroid.mcp.server._log") as mock_log:
        await server.read_resource(
            "mousedroid://memory/episodes/recent",
            peer="stdio",
            token=None,
        )
        mock_log.warning.assert_called_once()
        call_args = mock_log.warning.call_args
        assert call_args[0][0] == "mcp_memory_query_latency_degraded"


@settings(max_examples=20, deadline=None)
@given(
    limit_ms=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
)
def test_property_metric_always_non_negative(limit_ms: float) -> None:
    """Property: observed latency is always >= 0 regardless of limit."""
    server, metrics, _ = _build_server(limit_ms=limit_ms)

    asyncio.run(
        server.read_resource(
            "mousedroid://memory/episodes/recent",
            peer="stdio",
            token=None,
        )
    )

    metrics.observe_mcp_memory_query_latency_ms.assert_called_once()
    observed_ms: float = metrics.observe_mcp_memory_query_latency_ms.call_args[0][0]
    assert observed_ms >= 0.0
