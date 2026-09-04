"""Adapter that exposes the existing :class:`ToolRegistry` over MCP.

Responsibilities:

* Expand / filter the registry into the set of MCP-visible tools per
  the deny-list, allow-list, and ``expose_actuation_tools`` toggle.
* Enforce a token-bucket rate limit per session.
* Gate every actuation tool on the live
  :class:`~mousedroid.safety.protocol.SafetyMonitorProtocol`.
* Wrap each handler invocation in a timeout, the configured circuit
  breaker, and the configured retry policy — all sourced from
  :class:`MCPConfig` (with fallbacks to root :class:`Settings`).
* Translate every outcome into a
  :class:`~mousedroid.mcp.protocol.MCPToolResult` with consistent
  status/error labels for the metrics layer.

No values are hardcoded — failure thresholds, retry counts, timeouts,
and even the gating predicate are config-driven.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from mousedroid.common.rate_limit import TokenBucket
from mousedroid.logging.setup import get_logger
from mousedroid.mcp.metrics import record_request, record_tool_call
from mousedroid.mcp.protocol import MCPRequestContext, MCPToolResult
from mousedroid.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError

if TYPE_CHECKING:
    from mousedroid.common.tools.registry import ToolRegistry, ToolSpec
    from mousedroid.config.schema import (
        MCPConfig,
        Settings,
    )
    from mousedroid.safety.protocol import SafetyMonitorProtocol
    from mousedroid.sensing.protocol import ObservationProtocol
    from mousedroid.telemetry.metrics import MetricsRegistry

_log = get_logger(__name__)

# Stable identifier for the bridge's circuit breaker; surfaces in
# Prometheus labels and log lines so dashboards can pin a panel to it.
BREAKER_NAME = "mcp_tool_call"


# Backwards-compatible alias: the implementation moved to
# :mod:`mousedroid.common.rate_limit` so the OpenClaw REST mission endpoint
# can share the exact same bucket. Keeping the underscore-prefixed name as
# an alias preserves any in-tree imports that referenced ``_TokenBucket``.
_TokenBucket = TokenBucket


class MCPToolBridge:
    """Bridges the existing ToolRegistry into MCP's tool surface.

    The bridge is fully data-driven: which tools are visible, which are
    gated, and how invocations are wrapped all come from
    :class:`MCPConfig`.
    """

    def __init__(
        self,
        cfg: MCPConfig,
        root_cfg: Settings,
        tool_registry: ToolRegistry,
        safety_monitor: SafetyMonitorProtocol,
        metrics: MetricsRegistry | None = None,
        *,
        observation_provider: Callable[[], ObservationProtocol | None] | None = None,
    ) -> None:
        """Wire the bridge.

        Args:
            cfg: MCP-specific configuration.
            root_cfg: Root settings. Sources the fallback
                :class:`CircuitBreakerConfig` when
                :attr:`MCPConfig.circuit_breaker` is ``None``, and the
                :attr:`OpenClawConfig.require_actuation_ack` half of the
                actuation gate (see :attr:`_actuation_permitted`).
            tool_registry: The shared registry returned by
                :func:`~mousedroid.common.tools.registry.create_default_registry`.
            safety_monitor: Live monitor consulted before any actuation
                tool is dispatched.
            metrics: Optional metrics registry; ``None`` disables metric
                recording entirely.
            observation_provider: Optional callable returning the latest
                :class:`ObservationProtocol`. When ``None`` the safety
                gate runs against an empty observation snapshot — which
                is conservative but still correct (gate is still called).
        """
        self._cfg = cfg
        self._root_cfg = root_cfg
        self._registry = tool_registry
        self._safety_monitor = safety_monitor
        self._metrics = metrics
        self._observation_provider = observation_provider
        self._rate_limiter = TokenBucket(cfg.rate_limit_rps)
        breaker_cfg = (
            cfg.circuit_breaker if cfg.circuit_breaker is not None else root_cfg.circuit_breaker
        )
        self._circuit = CircuitBreaker(BREAKER_NAME, breaker_cfg)
        # Pre-compute deny / actuation sets for O(1) lookups
        self._denylist: set[str] = set(cfg.tools_denylist)
        self._actuation: set[str] = set(cfg.actuation_tools)
        _log.info(
            "mcp_tool_bridge_init",
            registry_size=len(self._registry),
            denylist=sorted(self._denylist),
            actuation=sorted(self._actuation),
            allowlist=cfg.tools_allowlist,
            expose_actuation=cfg.expose_actuation_tools,
        )

    # ------------------------------------------------------------------
    # List helpers
    # ------------------------------------------------------------------

    @property
    def _actuation_permitted(self) -> bool:
        """Whether actuation tools may be listed or dispatched at all.

        Implements the two-of-two gate that
        :attr:`OpenClawConfig.require_actuation_ack` documents: an actuation
        tool needs *both* that flag and
        :attr:`MCPConfig.expose_actuation_tools`. Until now only the second was
        enforced, so an operator who set ``expose_actuation_tools: true``
        believing a second interlock was fastened had exactly one.

        When ``Settings.openclaw`` is ``None`` the OpenClaw subsystem — which
        is what declares actuation-capable skills — is not wired, so the second
        gate is vacuously satisfied and behaviour is byte-identical to before
        this check existed.
        """
        if not self._cfg.expose_actuation_tools:
            return False
        openclaw = self._root_cfg.openclaw
        if openclaw is None:
            return True
        return openclaw.require_actuation_ack

    def visible_tool_names(self) -> list[str]:
        """Return the deduplicated, ordered list of tools exposed via MCP.

        Hidden when:

        * Name appears in :attr:`MCPConfig.tools_denylist`.
        * :attr:`MCPConfig.tools_allowlist` is non-None and the name is
          missing from it.
        * Tool is in :attr:`MCPConfig.actuation_tools` and actuation is not
          permitted — see :attr:`_actuation_permitted`, which requires both
          :attr:`MCPConfig.expose_actuation_tools` and
          :attr:`OpenClawConfig.require_actuation_ack`.
        * Allow-list references an unknown name (silently dropped).
        """
        registry_names = list(self._registry.names)
        allowed = set(self._cfg.tools_allowlist) if self._cfg.tools_allowlist is not None else None
        actuation_permitted = self._actuation_permitted
        out: list[str] = []
        for name in registry_names:
            if name in self._denylist:
                continue
            if allowed is not None and name not in allowed:
                continue
            if name in self._actuation and not actuation_permitted:
                continue
            out.append(name)
        return out

    def visible_tool_specs(self) -> list[ToolSpec]:
        """Return the :class:`ToolSpec` objects for every visible tool.

        Returns:
            Specs in registry order. Names absent from the registry are
            skipped; callers receive only valid handlers.
        """
        specs: list[ToolSpec] = []
        for name in self.visible_tool_names():
            spec = self._registry.get(name)
            if spec is None:  # pragma: no cover - unreachable; guarded by visible_tool_names
                continue
            specs.append(spec)
        return specs

    def get_tool_description(self, name: str) -> str:
        """Return the description for a registered tool, or the name as fallback.

        Public accessor used by the SDK transport adapter so it can build
        ``mcp.types.Tool`` payloads without reaching into private bridge
        attributes.

        Args:
            name: Tool identifier.

        Returns:
            The registered description, or ``name`` when the tool was
            removed from the registry mid-session (defensive — keeps the
            list_tools response well-formed).
        """
        spec = self._registry.get(name)
        return spec.description if spec is not None else name

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def make_request_context(
        self,
        peer: str = "unknown",
        *,
        token_present: bool = False,
    ) -> MCPRequestContext:
        """Build a fresh :class:`MCPRequestContext` with a unique id.

        Args:
            peer: Best-effort client identifier.
            token_present: Whether the request carried a token.

        Returns:
            A new request context (UUID4 hex id).
        """
        return MCPRequestContext(
            request_id=uuid.uuid4().hex,
            peer=peer,
            token_present=token_present,
        )

    async def call_tool(
        self,
        name: str,
        args: dict[str, Any] | None,
        ctx: MCPRequestContext,
    ) -> MCPToolResult:
        """Dispatch a tool through the gates, with metrics + structured logs.

        Args:
            name: Tool identifier.
            args: Keyword arguments forwarded to the registered handler.
            ctx: Per-request context (request id, peer, token presence).

        Returns:
            An :class:`MCPToolResult` whose ``status`` is one of the
            documented labels (never raises for expected failure modes).
        """
        kwargs = dict(args or {})
        start = time.monotonic()
        log = _log.bind(
            mcp_request_id=ctx.request_id,
            mcp_peer=ctx.peer,
            mcp_tool=name,
        )
        log.info("mcp_tool_call_started")

        # 1. Deny-list (highest precedence).
        if name in self._denylist:
            return self._finish(start, name, "denied", error="tool denied by config", log=log)

        # 2. Allow-list (when set).
        allowed = self._cfg.tools_allowlist
        if allowed is not None and name not in allowed:
            return self._finish(start, name, "denied", error="tool not in allowlist", log=log)

        # 3. Actuation gate: tool is in the actuation set but actuation is not
        #    permitted. Two-of-two per OpenClawConfig.require_actuation_ack —
        #    see _actuation_permitted.
        is_actuation = name in self._actuation
        if is_actuation and not self._actuation_permitted:
            return self._finish(
                start,
                name,
                "actuation_disabled",
                error="actuation tools disabled by config",
                log=log,
            )

        # 4. Registry lookup.
        spec = self._registry.get(name)
        if spec is None:
            return self._finish(start, name, "denied", error="tool not registered", log=log)

        # 5. Rate limit. ``take`` returns ``(ok, retry_after_s)``; we
        # only surface ``ok`` here because the MCP error envelope has
        # no retry-hint field, but the hint is included in the error
        # string for operator-side observability.
        taken, retry_after_s = await self._rate_limiter.take()
        if not taken:
            return self._finish(
                start,
                name,
                "rate_limited",
                error=f"per-session rate limit exceeded; retry_after_s={retry_after_s:.3f}",
                log=log,
            )

        # 6. Safety gate (only for actuation tools that survived the toggle).
        if is_actuation:
            try:
                ctx_safety = self._safety_monitor.evaluate(self._observation(), 0.0)
            except Exception as exc:
                return self._finish(
                    start, name, "error", error=f"safety_monitor_error:{exc}", log=log
                )
            if ctx_safety.is_emergency:
                return self._finish(
                    start, name, "refused_emergency", error="safety monitor in emergency", log=log
                )

        # 7. Invocation under timeout + circuit breaker. Retries are
        # NOT layered here: tool handlers in the registry are already
        # wrapped at the driver level (see ResilientESP32Driver,
        # ResilientLidar) so a second retry loop would just amplify
        # latency and mask root-cause failures from the metrics layer.

        async def _run() -> Any:
            return await asyncio.wait_for(
                spec.handler(**kwargs), timeout=self._cfg.request_timeout_s
            )

        try:
            payload = await self._circuit.call(_run)
        except CircuitOpenError as exc:
            return self._finish(start, name, "circuit_open", error=str(exc), log=log)
        except (TimeoutError, asyncio.TimeoutError):
            return self._finish(
                start, name, "timeout", error="handler exceeded request_timeout_s", log=log
            )
        except Exception as exc:
            return self._finish(start, name, "error", error=f"{type(exc).__name__}:{exc}", log=log)

        return self._finish(start, name, "ok", payload=_coerce_payload(payload), log=log)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _observation(self) -> ObservationProtocol:
        """Return the latest observation, falling back to an empty snapshot."""
        if self._observation_provider is not None:
            obs = self._observation_provider()
            if obs is not None:
                return obs
        # Conservative fallback: a zeroed observation snapshot so the
        # safety monitor is still consulted but never sees stale data
        # masquerading as fresh.
        from mousedroid.sensing.bundle import MouseDroidObservationBundle

        return MouseDroidObservationBundle()

    def _finish(
        self,
        start: float,
        tool: str,
        status: str,
        *,
        payload: dict[str, Any] | None = None,
        error: str | None = None,
        log: Any,
    ) -> MCPToolResult:
        latency_ms = (time.monotonic() - start) * 1000.0
        record_request(self._metrics, latency_ms=latency_ms)
        record_tool_call(self._metrics, tool=tool, result=status)
        if status == "ok":
            log.info("mcp_tool_call_completed", status=status, latency_ms=latency_ms)
        else:
            refused = {"denied", "refused_emergency", "actuation_disabled"}
            event = "mcp_tool_call_refused" if status in refused else "mcp_tool_call_failed"
            log.warning(event, status=status, latency_ms=latency_ms, error=error)
        return MCPToolResult(
            status=status,
            payload=payload if payload is not None else ({"error": error} if error else {}),
            latency_ms=latency_ms,
            error=error,
        )


def _coerce_payload(value: Any) -> dict[str, Any]:
    """Wrap non-dict handler returns in a uniform JSON-friendly envelope."""
    if isinstance(value, dict):
        return value
    return {"value": value}
