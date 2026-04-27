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

    async def _serve_sse(self) -> None:
        """SSE transport — currently disabled pending proper integration.

        The MCP SDK's :class:`mcp.server.sse.SseServerTransport.connect_sse`
        is an :func:`~contextlib.asynccontextmanager`, not an ASGI app, so
        the previous ``Mount("/", app=connect_sse)`` shim was incorrect.
        Implementing SSE properly requires:

        1. A Starlette ``Route("/sse", handle_sse)`` whose handler enters
           the ``connect_sse`` context manager and calls
           ``sdk_server.run(read, write, init_options)`` inside it.
        2. A separate ``Mount("/messages/", transport.handle_post_message)``
           for client POST traffic.
        3. Bearer-token middleware on Starlette so the token validated by
           :class:`MCPConfig` is enforced per-request — the current
           callbacks do not propagate ``token`` into ``call_tool``.

        Until those land, we refuse to bind. ``stdio`` remains fully
        functional and is the recommended transport for local clients
        (Claude Desktop, Claude Code).
        """
        msg = (
            "SSE transport not yet implemented end-to-end "
            "(see MCP_NEXT_STEPS.md P0 'transport bind-up' follow-ups). "
            "Use mcp.transport='stdio' or set mcp.bind_transport=false."
        )
        raise NotImplementedError(msg)

    async def _serve_streamable_http(self) -> None:
        """Streamable HTTP transport — currently disabled.

        Same status as :meth:`_serve_sse` — the bearer-token middleware
        and Starlette wiring need additional work before this is safe to
        expose. Tracked in MCP_NEXT_STEPS.md P0 follow-ups.
        """
        msg = (
            "streamable_http transport not yet implemented end-to-end "
            "(see MCP_NEXT_STEPS.md P0 'transport bind-up' follow-ups). "
            "Use mcp.transport='stdio' or set mcp.bind_transport=false."
        )
        raise NotImplementedError(msg)

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
