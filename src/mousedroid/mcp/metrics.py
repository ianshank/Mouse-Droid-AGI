"""Thin helpers that adapt the MCP server to the shared :class:`MetricsRegistry`.

These wrappers keep the MCP module decoupled from the underlying metric
implementation: production code calls :func:`record_request` /
:func:`record_tool_call`; tests substitute ``None`` for the registry to
disable metrics entirely. No metric names are constructed here — they
live in :mod:`mousedroid.telemetry.metrics`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.telemetry.metrics import MetricsRegistry

_log = get_logger(__name__)


def record_request(
    metrics: MetricsRegistry | None,
    *,
    latency_ms: float,
) -> None:
    """Record a single MCP request's latency and the request counter.

    Args:
        metrics: Shared metrics registry; ``None`` disables recording.
        latency_ms: End-to-end request latency (ms).
    """
    if metrics is None:
        return
    metrics.inc_mcp_request()
    metrics.observe_mcp_request_latency_ms(latency_ms)


def record_tool_call(
    metrics: MetricsRegistry | None,
    *,
    tool: str,
    result: str,
) -> None:
    """Record the outcome of a single tool invocation.

    Args:
        metrics: Shared metrics registry; ``None`` disables recording.
        tool: Tool name (matches a key in
            :class:`~mousedroid.common.tools.registry.ToolRegistry`).
        result: One of ``"ok"``, ``"denied"``, ``"refused_emergency"``,
            ``"timeout"``, ``"rate_limited"``, ``"error"``,
            ``"client_disconnected"``, ``"circuit_open"``, or
            ``"actuation_disabled"``.
    """
    if metrics is None:
        return
    metrics.inc_mcp_tool_call(tool, result)
