"""MouseDroid MCP server implementation.

Owns the lifecycle (``start`` / ``stop``), wires together the bridge,
resource providers, prompts, and auth, and runs the background sampler
that pulls telemetry frames into the MCP-side ring buffer.

The actual MCP wire protocol (transports, JSON-RPC framing, capability
negotiation) is delegated to the optional ``mcp`` SDK; that import is
lazy and gated so the rest of the codebase remains importable on
machines without the package installed. When ``mcp`` is unavailable the
server logs a warning and refuses to start (the factory function returns
``None`` before ever instantiating it in production).

The server is supervised exactly like the telemetry server — it tracks
its background tasks with :func:`spawn_tracked` and drains them with
:func:`cancel_and_drain` on shutdown.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import TYPE_CHECKING, Any

from mousedroid.common.async_utils import cancel_and_drain, spawn_tracked
from mousedroid.logging.setup import get_logger
from mousedroid.mcp.auth import BearerTokenValidator
from mousedroid.mcp.prompts import default_prompts
from mousedroid.mcp.protocol import MCPServerProtocol as MCPServerProtocol  # re-export
from mousedroid.mcp.resources import (
    ConfigResourceProvider,
    LogResourceProvider,
    MemoryResourceProvider,
    TelemetryResourceProvider,
    parse_resource_uri,
)
from mousedroid.mcp.tool_bridge import MCPToolBridge

if TYPE_CHECKING:
    from mousedroid.common.tools.registry import ToolRegistry
    from mousedroid.config.schema import MCPConfig, Settings
    from mousedroid.safety.protocol import SafetyMonitorProtocol
    from mousedroid.telemetry.log_buffer import LogRingBuffer
    from mousedroid.telemetry.metrics import MetricsRegistry
    from mousedroid.telemetry.protocol import TelemetryPublisherProtocol

_log = get_logger(__name__)


class MouseDroidMCPServer:
    """Concrete :class:`MCPServerProtocol` implementation.

    Holds the full bridge, providers, and lifecycle tasks. Designed to
    be supervised by the orchestrator the same way ``TelemetryServer``
    is — ``await server.start()`` / ``await server.stop()`` with no
    background ownership outside this class.
    """

    def __init__(
        self,
        cfg: MCPConfig,
        root_cfg: Settings,
        tool_registry: ToolRegistry,
        safety_monitor: SafetyMonitorProtocol,
        telemetry_publisher: TelemetryPublisherProtocol | None = None,
        log_buffer: LogRingBuffer | None = None,
        metrics_registry: MetricsRegistry | None = None,
        memory_tier: Any | None = None,
    ) -> None:
        """Wire all collaborators.

        Args:
            cfg: MCP-specific configuration block.
            root_cfg: Root settings (used by the bridge for retry/CB
                fallbacks and by the resource provider for redaction).
            tool_registry: Shared tool registry instance.
            safety_monitor: Live safety monitor for actuation gates.
            telemetry_publisher: Optional publisher; ``None`` disables
                the telemetry resource.
            log_buffer: Optional log ring buffer; ``None`` disables the
                log resource.
            metrics_registry: Optional metrics registry; ``None``
                disables metric recording without affecting behaviour.
            memory_tier: Optional memory tier for the memory resource.
        """
        self._cfg = cfg
        self._root_cfg = root_cfg
        self._metrics = metrics_registry
        self._key_pattern = re.compile(cfg.redact_key_pattern)

        self._auth = BearerTokenValidator(
            cfg.auth_token_env_var,
            required=_auth_required(cfg),
        )
        self._bridge = MCPToolBridge(
            cfg=cfg,
            root_cfg=root_cfg,
            tool_registry=tool_registry,
            safety_monitor=safety_monitor,
            metrics=metrics_registry,
        )
        self._telemetry = TelemetryResourceProvider(cfg, telemetry_publisher)
        self._logs = LogResourceProvider(cfg, log_buffer, key_pattern=self._key_pattern)
        self._config_provider = ConfigResourceProvider(cfg, root_cfg, key_pattern=self._key_pattern)
        self._memory = MemoryResourceProvider(cfg, memory_tier, key_pattern=self._key_pattern)
        self._prompts = default_prompts()

        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._running: bool = False
        self._client_count: int = 0
        self._serve_task: asyncio.Task[Any] | None = None

    # ------------------------------------------------------------------
    # MCPServerProtocol
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Bind the configured transport and start the sampler task."""
        if self._running:
            _log.warning("mcp_server_start_idempotent_skip")
            return

        # Validate the auth secret eagerly so misconfiguration surfaces
        # at start rather than on the first request.
        if self._auth.required and not self._auth.has_secret:
            msg = (
                f"MCP auth required but {self._cfg.auth_token_env_var} is unset; refusing to start"
            )
            _log.warning("mcp_server_start_missing_token", env_var=self._cfg.auth_token_env_var)
            raise RuntimeError(msg)

        self._running = True
        # Telemetry sampler runs on a fixed cadence so the publisher
        # queue never grows unbounded waiting on MCP clients.
        sample_period = 1.0 / self._cfg.sample_telemetry_hz
        spawn_tracked(
            self._background_tasks,
            self._sampler_loop(sample_period),
            name="mcp_telemetry_sampler",
        )
        # The serve loop is spawned only when a real MCP SDK is
        # available — otherwise the server still exposes its bridge for
        # in-process tests but does not bind a transport.
        try:
            self._serve_task = spawn_tracked(
                self._background_tasks,
                self._serve_loop(),
                name="mcp_serve",
            )
        except Exception:  # pragma: no cover - defensive guard
            _log.warning("mcp_server_serve_loop_spawn_failed", exc_info=True)

        _log.info(
            "mcp_server_started",
            transport=self._cfg.transport,
            host=self._cfg.host,
            port=self._cfg.port,
            visible_tools=self._bridge.visible_tool_names(),
            resources=self.list_resource_uris(),
        )

    async def stop(self) -> None:
        """Drain the sampler / serve tasks and mark the server stopped."""
        if not self._running:
            return
        self._running = False
        drained = await cancel_and_drain(self._background_tasks)
        _log.info("mcp_server_stopped", drained_tasks=drained)

    @property
    def is_running(self) -> bool:
        """Whether the server is currently running."""
        return self._running

    @property
    def client_count(self) -> int:
        """Best-effort connected client count."""
        return self._client_count

    # ------------------------------------------------------------------
    # Surface accessors used by the SDK adapter and tests.
    # ------------------------------------------------------------------

    def list_tool_names(self) -> list[str]:
        """Tool names visible to MCP clients."""
        return self._bridge.visible_tool_names()

    def tool_description(self, name: str) -> str:
        """Return the human-readable description for a registered tool.

        Args:
            name: Tool identifier; expected to be visible per
                :meth:`list_tool_names`.

        Returns:
            The tool's description, or the name itself when no spec is
            registered (defensive — keeps the SDK list_tools response
            well-formed even if a tool is removed mid-session).
        """
        return self._bridge.get_tool_description(name)

    def list_prompt_names(self) -> list[str]:
        """Prompt identifiers exposed to MCP clients."""
        return [p.name for p in self._prompts]

    def list_prompts(self) -> list[Any]:
        """Return the full prompt registry (name + description + template).

        Returns:
            A list of :class:`~mousedroid.mcp.prompts.MCPPrompt` records
            so the transport adapter can build typed responses without
            re-deriving descriptions.
        """
        return list(self._prompts)

    def get_prompt(self, name: str) -> Any:
        """Look up a single prompt by name.

        Args:
            name: Prompt identifier (must appear in
                :meth:`list_prompt_names`).

        Returns:
            The :class:`~mousedroid.mcp.prompts.MCPPrompt` record.

        Raises:
            KeyError: When no prompt matches ``name``.
        """
        for prompt in self._prompts:
            if prompt.name == name:
                return prompt
        msg = f"unknown MCP prompt: {name!r}"
        raise KeyError(msg)

    def list_resource_uris(self) -> list[str]:
        """Resource URIs exposed to MCP clients."""
        uris: list[str] = []
        uris.extend(self._telemetry.list_uris())
        uris.extend(self._logs.list_uris())
        uris.extend(self._config_provider.list_uris())
        uris.extend(self._memory.list_uris())
        return uris

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        peer: str = "stdio",
        token: str | None = None,
    ) -> dict[str, Any]:
        """Invoke a tool by name through the bridge.

        Args:
            name: Tool identifier.
            arguments: Optional kwargs for the handler.
            peer: Best-effort client identifier for log correlation.
            token: Optional bearer token (ignored for stdio loopback).

        Returns:
            The :class:`MCPToolResult` rendered as a flat dict — keys
            ``status``, ``payload``, ``latency_ms``, ``error``.
        """
        self._auth.validate(token)
        ctx = self._bridge.make_request_context(peer=peer, token_present=token is not None)
        result = await self._bridge.call_tool(name, arguments, ctx)
        return {
            "status": result.status,
            "payload": result.payload,
            "latency_ms": result.latency_ms,
            "error": result.error,
        }

    async def read_resource(
        self,
        uri: str,
        *,
        peer: str = "stdio",
        token: str | None = None,
    ) -> dict[str, Any]:
        """Read a resource by URI.

        Args:
            uri: Full ``mousedroid://...`` URI.
            peer: Best-effort client identifier for log correlation.
            token: Optional bearer token (ignored for stdio loopback).

        Returns:
            JSON-friendly payload.

        Raises:
            KeyError: When no provider matches the URI.
            PermissionError: When the matching provider is disabled.
        """
        self._auth.validate(token)
        _, path, query = parse_resource_uri(uri)
        log = _log.bind(mcp_peer=peer, mcp_resource_uri=uri)
        if path.startswith("/telemetry"):
            log.info("mcp_resource_read", provider="telemetry")
            return self._telemetry.read(path, query)
        if path.startswith("/logs"):
            log.info("mcp_resource_read", provider="logs")
            return self._logs.read(path, query)
        if path.startswith("/config"):
            log.info("mcp_resource_read", provider="config")
            return self._config_provider.read(path, query)
        if path.startswith("/memory"):
            log.info("mcp_resource_read", provider="memory")
            t0 = time.monotonic()
            try:
                return await self._memory.read(path, query)
            finally:
                latency_ms = (time.monotonic() - t0) * 1000.0
                if self._metrics:
                    self._metrics.observe_mcp_memory_query_latency_ms(latency_ms)
                if self._root_cfg.baselines is not None:
                    limit = self._root_cfg.baselines.max_memory_query_latency_ms
                    if latency_ms > limit:
                        _log.warning(
                            "mcp_memory_query_latency_degraded",
                            latency_ms=latency_ms,
                            limit_ms=limit,
                        )
        msg = f"no MCP provider handles resource: {uri!r}"
        raise KeyError(msg)

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    async def _sampler_loop(self, period_s: float) -> None:
        """Drain the telemetry publisher queue at ``sample_telemetry_hz``."""
        try:
            while self._running:
                await self._telemetry.sample_once()
                await asyncio.sleep(period_s)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive
            _log.warning("mcp_sampler_loop_error", exc_info=True)

    async def _serve_loop(self) -> None:
        """Run the optional MCP SDK serve loop (transport-specific).

        Delegation rules (all driven by :class:`MCPConfig`, no hardcoded
        thresholds):

        * If the optional ``mcp`` SDK is missing, idle until cancelled.
          The bridge stays usable for in-process callers.
        * If the SDK is available but :attr:`MCPConfig.bind_transport`
          is ``False`` (default), idle. This keeps unit tests and
          embedded callers free of real socket binding.
        * If the SDK is available and ``bind_transport`` is ``True``,
          delegate to :meth:`MCPTransportAdapter.serve` which binds the
          configured stdio / SSE / streamable_http transport.

        The idle poll interval is derived from
        :attr:`MCPConfig.sample_telemetry_hz` so a single config field
        controls both the telemetry sampler cadence and how quickly the
        serve loop notices ``self._running`` flipping during shutdown.
        """
        from mousedroid.mcp.transport import build_transport_adapter

        idle_period_s = 1.0 / self._cfg.sample_telemetry_hz
        adapter = build_transport_adapter(self)
        if adapter is None:
            _log.info(
                "mcp_serve_loop_idle",
                reason="mcp_package_not_installed",
                hint="install mousedroid[mcp] for real transports",
            )
            await self._idle_until_stopped(idle_period_s)
            return
        if not self._cfg.bind_transport:
            _log.info(
                "mcp_serve_loop_idle",
                reason="bind_transport_disabled",
                hint="set mcp.bind_transport=true (or MOUSEDROID_MCP__BIND_TRANSPORT=true) to bind",
            )
            await self._idle_until_stopped(idle_period_s)
            return
        _log.info("mcp_serve_loop_started", transport=self._cfg.transport)
        try:
            await adapter.serve()
        except asyncio.CancelledError:
            raise
        except Exception:
            _log.warning("mcp_serve_loop_error", exc_info=True)
            raise

    async def _idle_until_stopped(self, period_s: float) -> None:
        """Sleep in ``period_s`` increments until the server stops."""
        try:
            while self._running:
                await asyncio.sleep(period_s)
        except asyncio.CancelledError:
            raise


def _auth_required(cfg: MCPConfig) -> bool:
    """Decide whether the validator must enforce a token.

    stdio + loopback transports are exempt because the parent process
    owns the connection. Every other configuration requires a token.
    """
    if cfg.transport == "stdio":
        return False
    return cfg.host not in {"127.0.0.1", "localhost"}
