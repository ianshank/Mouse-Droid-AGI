"""MCP server request / tool-call / latency metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.telemetry.metrics.primitives import (
    _Counter,
    _DoubleLabeledCounter,
    _Histogram,
    _prepare_bucket_boundaries,
    _render_counter,
    _render_double_labeled_counter,
    _render_histogram,
)

if TYPE_CHECKING:
    from mousedroid.config.schema import MetricsConfig


class _McpMetricsMixin:
    """MCP request / tool-call / latency metric family."""

    # Populated by ``_CoreMetricsMixin._init_core_metrics``, which always runs
    # first from ``MetricsRegistry.__init__``.
    _cfg: MetricsConfig

    def _init_mcp_metrics(self, cfg: MetricsConfig) -> None:
        """Initialise MCP server metrics.

        Args:
            cfg: Metrics configuration with namespace and toggle flags.
        """
        ns = cfg.namespace

        self._mcp_requests = _Counter()
        self._mcp_tool_calls = _DoubleLabeledCounter()
        self._mcp_request_latency_ms = _Histogram(
            _prepare_bucket_boundaries(cfg.mcp_latency_buckets_ms)
        )
        self._mcp_memory_query_latency_ms = _Histogram(
            _prepare_bucket_boundaries(cfg.mcp_memory_query_latency_buckets_ms)
        )

        # MCP metric names — all derived from namespace
        self._name_mcp_requests = f"{ns}_mcp_requests"
        self._name_mcp_tool_calls = f"{ns}_mcp_tool_calls_total"
        self._name_mcp_request_latency = f"{ns}_mcp_request_latency_ms"
        self._name_mcp_memory_query_latency = f"{ns}_mcp_memory_query_latency_ms"

    # ------------------------------------------------------------------
    # MCP server helpers
    # ------------------------------------------------------------------

    def inc_mcp_request(self, amount: int = 1) -> None:
        """Increment the total MCP request counter (any kind of request)."""
        if self._cfg.track_mcp:
            self._mcp_requests.inc(amount)

    def inc_mcp_tool_call(self, tool: str, result: str, amount: int = 1) -> None:
        """Increment the per-tool MCP call counter.

        Args:
            tool: Tool name (e.g. ``"health_check"``).
            result: Outcome label (e.g. ``"ok"``, ``"refused_emergency"``,
                ``"denied"``, ``"rate_limited"``, ``"timeout"``,
                ``"error"``, ``"client_disconnected"``).
            amount: Increment amount (default 1).
        """
        if self._cfg.track_mcp:
            self._mcp_tool_calls.inc(tool, result, amount)

    def observe_mcp_request_latency_ms(self, value: float) -> None:
        """Record total MCP request latency in milliseconds."""
        if self._cfg.track_mcp:
            self._mcp_request_latency_ms.observe(value)

    def observe_mcp_memory_query_latency_ms(self, value: float) -> None:
        """Record memory query latency in milliseconds."""
        if self._cfg.track_openclaw_memory:
            self._mcp_memory_query_latency_ms.observe(value)

    # ------------------------------------------------------------------
    # Prometheus text exposition — family renderer
    # ------------------------------------------------------------------

    def _families_mcp(self) -> list[list[str]]:
        """MCP request / tool-call / latency families."""
        cfg = self._cfg
        out: list[list[str]] = []
        if cfg.track_mcp:
            out.append(
                _render_counter(
                    self._name_mcp_requests,
                    "Total MCP requests received",
                    self._mcp_requests.value,
                )
            )
            tool_call_snapshot = self._mcp_tool_calls.snapshot()
            if tool_call_snapshot:
                out.append(
                    _render_double_labeled_counter(
                        self._name_mcp_tool_calls,
                        "MCP tool call outcomes (labels: tool, result)",
                        "tool",
                        "result",
                        tool_call_snapshot,
                    )
                )
            mcp_buckets, mcp_sum, mcp_count = self._mcp_request_latency_ms.snapshot()
            if mcp_count > 0:
                out.append(
                    _render_histogram(
                        self._name_mcp_request_latency,
                        "MCP request end-to-end latency in milliseconds",
                        mcp_buckets,
                        mcp_sum,
                        mcp_count,
                    )
                )

        if cfg.track_openclaw_memory:
            mq_buckets, mq_sum, mq_count = self._mcp_memory_query_latency_ms.snapshot()
            if mq_count > 0:
                out.append(
                    _render_histogram(
                        self._name_mcp_memory_query_latency,
                        "MCP memory query latency for OpenClaw in milliseconds",
                        mq_buckets,
                        mq_sum,
                        mq_count,
                    )
                )
        return out
