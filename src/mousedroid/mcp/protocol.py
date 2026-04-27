"""MCP server protocol and value types.

Defines the runtime-checkable :class:`MCPServerProtocol` (mirrors
:class:`~mousedroid.telemetry.protocol.TelemetryServerProtocol`) plus the
two frozen dataclasses used to carry per-request context and tool-call
results across the bridge.

Only standard library types are imported here so this module can be
loaded without the optional ``mcp`` SDK installed — callers test
protocol conformance via :func:`isinstance` against the
``@runtime_checkable`` ``Protocol``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class MCPRequestContext:
    """Per-request context carried through the bridge.

    Attributes:
        request_id: Unique identifier (UUID4 hex) for log/metric correlation.
        peer: Best-effort client identifier (transport-specific; e.g.
            ``"stdio"`` or a remote address). May be ``"unknown"``.
        token_present: True iff the request carried a bearer token (only
            meaningful for non-stdio transports).
    """

    request_id: str
    peer: str = "unknown"
    token_present: bool = False


@dataclass(frozen=True)
class MCPToolResult:
    """Result returned by the tool bridge for a single ``call_tool`` request.

    Distinct from the raw ToolRegistry handler return value: this wraps
    the handler payload with status, latency, and error metadata so the
    server can render a uniform MCP response.

    Attributes:
        status: Outcome label — one of ``"ok"``, ``"denied"``,
            ``"refused_emergency"``, ``"timeout"``, ``"rate_limited"``,
            ``"error"``, ``"client_disconnected"``, ``"circuit_open"``,
            or ``"actuation_disabled"``.
        payload: Handler return value when ``status == "ok"``; an error
            description dict otherwise. Always JSON-serialisable.
        latency_ms: Wall-clock time spent in the bridge (including gates
            and the handler invocation).
        error: Optional human-readable error string (set when
            ``status != "ok"``).
    """

    status: str
    payload: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None


@runtime_checkable
class MCPServerProtocol(Protocol):
    """Lifecycle interface for the MCP server.

    Mirrors :class:`~mousedroid.telemetry.protocol.TelemetryServerProtocol`
    so the orchestrator can supervise both with the same
    ``await server.start()`` / ``await server.stop()`` pattern.
    """

    async def start(self) -> None:
        """Bind transport, install handlers, begin accepting clients."""
        ...

    async def stop(self) -> None:
        """Gracefully shut down: close transport, drain background tasks."""
        ...

    @property
    def is_running(self) -> bool:
        """Whether :meth:`start` has been called and :meth:`stop` has not."""
        ...

    @property
    def client_count(self) -> int:
        """Best-effort count of connected clients (0 for stdio transport)."""
        ...
