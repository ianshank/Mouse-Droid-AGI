"""MCP (Model Context Protocol) server module.

Bridges the existing :class:`~mousedroid.common.tools.registry.ToolRegistry`,
:class:`~mousedroid.telemetry.protocol.TelemetryPublisherProtocol`,
:class:`~mousedroid.telemetry.log_buffer.LogRingBuffer`, and (optionally)
episodic memory to any MCP-compliant client over stdio, SSE, or
streamable HTTP.

The :class:`MCPServerProtocol` mirrors
:class:`~mousedroid.telemetry.protocol.TelemetryServerProtocol` so the
orchestrator can supervise it identically. All concrete implementations
are selected by :func:`mousedroid.factory.build_mcp_server` based on the
optional :class:`~mousedroid.config.schema.MCPConfig`.
"""

from __future__ import annotations

from mousedroid.mcp.protocol import (
    MCPRequestContext,
    MCPServerProtocol,
    MCPToolResult,
)

__all__ = [
    "MCPRequestContext",
    "MCPServerProtocol",
    "MCPToolResult",
]
