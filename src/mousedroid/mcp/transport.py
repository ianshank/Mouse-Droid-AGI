"""SDK transport adapter for the MCP server.

Wraps the optional ``mcp`` package's ``Server`` class and maps its
callbacks (``list_tools``, ``call_tool``, ``list_resources``,
``read_resource``, ``list_prompts``, ``get_prompt``) onto
:class:`MouseDroidMCPServer` methods.

The import of ``mcp`` is lazy and gated; when the package is missing,
:func:`build_transport_adapter` returns ``None`` and the existing idle
loop in :meth:`MouseDroidMCPServer._serve_loop` keeps the bridge usable
for in-process callers.

Backwards compatibility: this module is only imported lazily inside
``MouseDroidMCPServer._serve_loop``; importing :mod:`mousedroid.mcp`
without the SDK installed continues to work exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.mcp.server import MouseDroidMCPServer

_log = get_logger(__name__)

# Stable name advertised to MCP clients during initialization. The MCP
# protocol requires a server name; this is purely a label and is not a
# tunable knob, so it lives here rather than in ``MCPConfig``.
MCP_SERVER_NAME = "mousedroid"


@dataclass
class MCPTransportAdapter:
    """Bind a real ``mcp.server.Server`` to a :class:`MouseDroidMCPServer`.

    Attributes:
        server: The MouseDroid MCP server instance owning the bridge,
            providers, and lifecycle.
        sdk_server: The instantiated ``mcp.server.Server`` whose
            decorators/handlers will be wired in subsequent tasks.
    """

    server: MouseDroidMCPServer
    sdk_server: Any  # ``mcp.server.Server`` when the SDK is installed.

    async def serve(self) -> None:
        """Bind the configured transport and run the SDK loop.

        Dispatches based on :attr:`MCPConfig.transport`. The
        ``MCPConfig`` validator already constrains the value to one of
        ``stdio``, ``sse``, ``streamable_http``, so the ``else`` branch
        is defensive only.

        Raises:
            ValueError: If ``self.server._cfg.transport`` is not a
                recognised value (defensive guard).
        """
        transport = self.server._cfg.transport
        self._register_handlers()
        if transport == "stdio":
            await self._serve_stdio()
        elif transport == "sse":
            await self._serve_sse()
        elif transport == "streamable_http":
            await self._serve_streamable_http()
        else:  # pragma: no cover - defensive (constrained by Pydantic Literal)
            msg = f"unsupported MCP transport: {transport!r}"
            raise ValueError(msg)

    def _register_handlers(self) -> None:
        """Attach the ``_on_*`` callbacks to ``self.sdk_server``.

        Uses the ``mcp.server.Server`` decorator pattern. The decorators
        are *factories* — calling them returns a registrar that takes
        the handler. This is why we invoke ``list_tools()`` first and
        then call the result with our coroutine.
        """
        self.sdk_server.list_tools()(self._on_list_tools)
        self.sdk_server.call_tool()(self._on_call_tool)
        self.sdk_server.list_resources()(self._on_list_resources)
        self.sdk_server.read_resource()(self._on_read_resource)
        self.sdk_server.list_prompts()(self._on_list_prompts)
        self.sdk_server.get_prompt()(self._on_get_prompt)

    async def _serve_stdio(self) -> None:  # pragma: no cover - binds real stdio
        """Run the SDK over stdio (parent process owns the connection).

        Coverage-excluded because the body opens a real stdio transport
        and blocks on JSON-RPC traffic; it is exercised end-to-end by
        ``tests/integration/test_mcp_stdio_client.py`` via the SDK's
        in-memory connector, and by manual operator-guide validation
        against Claude Desktop / Claude Code.
        """
        import mcp.server.stdio as _stdio

        async with _stdio.stdio_server() as (read, write):
            await self.sdk_server.run(read, write, self.sdk_server.create_initialization_options())

    async def _serve_sse(self) -> None:  # pragma: no cover - exercised in integration test
        """Bind the SSE transport via Starlette + uvicorn.

        Wires the SDK's :class:`mcp.server.sse.SseServerTransport` into a
        Starlette app guarded by :class:`BearerAuthMiddleware`. The auth
        validator reads the bearer secret from
        ``MCPConfig.auth_token_env_var``; the schema validator already
        refused to load a non-loopback bind without that env var set, so
        we never bind a publicly-reachable port without a secret.
        """
        await self._serve_starlette(transport_kind="sse")

    async def _serve_streamable_http(self) -> None:  # pragma: no cover - integration only
        """Bind the streamable_http transport via Starlette + uvicorn.

        Uses the SDK's
        :class:`mcp.server.streamable_http_manager.StreamableHTTPSessionManager`
        — same bearer middleware as SSE, same uvicorn lifecycle.
        """
        await self._serve_starlette(transport_kind="streamable_http")

    async def _serve_starlette(self, *, transport_kind: str) -> None:
        """Shared Starlette + uvicorn lifecycle for SSE / streamable_http.

        Logging is intentionally routed through structlog by passing
        ``log_config=None`` to uvicorn; uvicorn's default config would
        install a duplicate stdlib handler that diverges from the
        project's structured-log convention.
        """
        cfg = self.server._cfg
        token_validator = self._build_validator()
        app = self._build_starlette_app(transport_kind, token_validator)

        import uvicorn

        # ``lifespan="on"`` is required for the streamable_http transport's
        # session-manager context (see ``_build_starlette_app``); SSE has
        # no lifespan and Starlette runs an empty default one cleanly.
        config = uvicorn.Config(
            app,
            host=cfg.host,
            port=cfg.port,
            log_config=None,
            access_log=False,
            lifespan="on",
        )
        srv = uvicorn.Server(config)
        _log.info(
            "mcp_transport_started",
            transport=transport_kind,
            host=cfg.host,
            port=cfg.port,
            bind_external=cfg.bind_external,
        )
        try:
            await srv.serve()
        finally:
            _log.info(
                "mcp_transport_stopped",
                transport=transport_kind,
                host=cfg.host,
                port=cfg.port,
            )

    def _build_validator(self) -> Any:
        """Construct the bearer validator.

        Auth is required when ANY of:

        * ``bind_external=true`` (operator opted into a publicly-reachable
          listener; the schema validator already enforced env-var
          presence in that case).
        * The configured ``auth_token_env_var`` is set in the environment
          (operator deliberately provisioned a secret — even on loopback
          we honour it so local tests and dual-host development
          deployments enforce the same envelope production sees).

        Loopback binds with no token configured fall back to
        ``required=False`` so a developer running ``stdio`` or a quick
        local SSE smoke without a token still works.
        """
        import os

        from mousedroid.mcp.auth import BearerTokenValidator

        cfg = self.server._cfg
        token_present = bool(os.environ.get(cfg.auth_token_env_var))
        return BearerTokenValidator(
            cfg.auth_token_env_var,
            required=cfg.bind_external or token_present,
        )

    def _build_starlette_app(self, transport_kind: str, validator: Any) -> Any:
        """Compose the Starlette app for the given transport kind.

        Two transports map onto the same app shape so both share the
        bearer middleware composition; the only divergence is the
        routing strategy of the underlying SDK transport.

        * **SSE** uses a Starlette :class:`~starlette.routing.Route`
          pinned to ``/sse``, with a class-based ASGI endpoint
          (:class:`_SSEEndpoint`). A class instance bypasses Starlette's
          :func:`request_response` wrapper — which would otherwise emit
          a fresh ``http.response.start`` after the SDK already wrote
          one — and is treated as raw ASGI. The companion
          ``/messages/`` endpoint is mounted as a raw-ASGI handler.
        * **streamable_http** uses a single ``Mount("/", ...)`` because
          the SDK's session manager dispatches every method/path
          internally. A lifespan context wraps the session manager's
          ``run()`` so its background task group is alive for the
          server's lifetime.
        """
        from starlette.applications import Starlette
        from starlette.middleware import Middleware
        from starlette.routing import Mount

        from mousedroid.mcp.middleware import BearerAuthMiddleware

        middleware = [
            Middleware(BearerAuthMiddleware, validator=validator),
        ]

        if transport_kind == "sse":
            from mcp.server.sse import SseServerTransport
            from starlette.routing import Route

            sse = SseServerTransport("/messages/")
            sdk_server = self.sdk_server

            class _SSEEndpoint:
                """Raw-ASGI endpoint for the SSE handshake.

                Starlette's :class:`Route` only wraps async *functions*
                in :func:`starlette.routing.request_response` (which
                expects a returned :class:`~starlette.responses.Response`
                and emits a fresh ``http.response.start``). A class
                instance with ``__call__(scope, receive, send)`` is
                treated as raw ASGI instead — so the start event sent
                by ``connect_sse`` is the only one on the wire and there
                is no double-send protocol violation.
                """

                async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
                    async with sse.connect_sse(scope, receive, send) as (read, write):
                        await sdk_server.run(
                            read,
                            write,
                            sdk_server.create_initialization_options(),
                        )

            routes = [
                Route("/sse", endpoint=_SSEEndpoint(), methods=["GET"]),
                Mount("/messages/", app=sse.handle_post_message),
            ]
            return Starlette(routes=routes, middleware=middleware)

        # streamable_http
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

        manager = StreamableHTTPSessionManager(app=self.sdk_server)

        async def _handle_streamable(  # pragma: no cover - exercised by uvicorn
            scope: Any, receive: Any, send: Any
        ) -> None:
            await manager.handle_request(scope, receive, send)

        routes = [Mount("/", app=_handle_streamable)]

        # The session manager's internal task group must be alive while
        # requests are served. Wrap the Starlette app in a lifespan that
        # keeps it running for the duration of the server lifetime.
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _lifespan(_: Any) -> Any:  # pragma: no cover - exercised by uvicorn
            async with manager.run():
                yield

        return Starlette(routes=routes, middleware=middleware, lifespan=_lifespan)

    # ------------------------------------------------------------------
    # SDK callbacks — thin delegates onto :class:`MouseDroidMCPServer`.
    #
    # Returns are typed against ``mcp.types`` so the SDK decorators
    # accept them directly. ``inputSchema`` is a permissive
    # ``object`` schema; per-tool schemas can be promoted to a
    # ``MCPConfig.tool_schemas`` mapping in a later iteration without
    # changing the wire contract.
    # ------------------------------------------------------------------

    async def _on_list_tools(self) -> list[Any]:
        """Return the visible tools as ``mcp.types.Tool`` objects."""
        from mcp import types as _mt

        return [
            _mt.Tool(
                name=name,
                description=self.server.tool_description(name),
                inputSchema={"type": "object", "additionalProperties": True},
            )
            for name in self.server.list_tool_names()
        ]

    async def _on_call_tool(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        """Dispatch a tool call through the bridge.

        Returns a plain dict; the SDK wraps it as ``StructuredContent``
        and serialises it into ``CallToolResult``.
        """
        return await self.server.call_tool(name, arguments, peer="sdk")

    async def _on_list_resources(self) -> list[Any]:
        """Return the resource URIs as ``mcp.types.Resource`` objects."""
        from mcp import types as _mt
        from pydantic import AnyUrl

        return [
            _mt.Resource(name=_resource_name(uri), uri=AnyUrl(uri))
            for uri in self.server.list_resource_uris()
        ]

    async def _on_read_resource(self, uri: Any) -> str:
        """Read a resource by URI and return JSON-serialised content.

        The SDK's ``read_resource`` decorator passes an ``AnyUrl``; we
        normalise to ``str`` for the existing
        :meth:`MouseDroidMCPServer.read_resource` contract and serialise
        the dict payload to JSON text so the client receives a well-typed
        ``ReadResourceContents`` envelope.
        """
        import json

        payload = await self.server.read_resource(str(uri), peer="sdk")
        return json.dumps(payload, default=str)

    async def _on_list_prompts(self) -> list[Any]:
        """Return the available prompts as ``mcp.types.Prompt`` objects.

        Pulls description from each ``MCPPrompt`` so clients see something
        meaningful next to the name.
        """
        from mcp import types as _mt

        return [
            _mt.Prompt(name=p.name, description=p.description) for p in self.server.list_prompts()
        ]

    async def _on_get_prompt(self, name: str, arguments: dict[str, str] | None) -> Any:
        """Return the rendered prompt for ``name`` as ``mcp.types.GetPromptResult``.

        The MouseDroid prompts are static templates (no arg substitution
        in this revision). ``arguments`` is accepted to match the SDK
        signature; future revisions can interpolate.

        Raises:
            KeyError: When ``name`` is not in the prompt registry. The
                SDK surfaces this as an MCP error to the client.
        """
        from mcp import types as _mt

        prompt = self.server.get_prompt(name)
        return _mt.GetPromptResult(
            description=prompt.description,
            messages=[
                _mt.PromptMessage(
                    role="user",
                    content=_mt.TextContent(type="text", text=prompt.template),
                )
            ],
        )


def _resource_name(uri: str) -> str:
    """Derive a human-readable name from a ``mousedroid://...`` URI."""
    after_scheme = uri.split("://", 1)[-1]
    return after_scheme.strip("/").replace("/", "_") or "resource"


def build_transport_adapter(server: MouseDroidMCPServer) -> MCPTransportAdapter | None:
    """Construct the adapter when the optional SDK is importable.

    Args:
        server: The MouseDroid MCP server to wrap.

    Returns:
        ``MCPTransportAdapter`` when ``mcp.server`` imports cleanly;
        ``None`` otherwise (the caller falls back to the idle loop).
    """
    try:
        import mcp.server as _sdk
    except (ImportError, TypeError):
        # ``TypeError`` covers monkeypatched ``sys.modules['mcp']=None``
        # used in tests to simulate the SDK being absent.
        _log.info("mcp_transport_adapter_disabled", reason="mcp_package_missing")
        return None
    try:
        sdk_server = _sdk.Server(MCP_SERVER_NAME)
    except Exception:  # pragma: no cover - defensive
        _log.warning("mcp_transport_adapter_sdk_init_failed", exc_info=True)
        return None
    return MCPTransportAdapter(server=server, sdk_server=sdk_server)
