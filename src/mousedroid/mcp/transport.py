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
        """Run the SDK serve loop for the configured transport.

        Concrete transport implementations are added in subsequent
        tasks; this skeleton exists so the lifecycle wiring can land
        first.

        Raises:
            NotImplementedError: Always, until per-transport methods
                land in later tasks.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # SDK callbacks — thin delegates onto :class:`MouseDroidMCPServer`.
    #
    # Kept as plain ``async def`` methods so the unit tests exercise the
    # mapping without spinning up the SDK transport. They are wired into
    # the ``mcp.server.Server`` decorators in :meth:`_register_handlers`
    # in a later task.
    # ------------------------------------------------------------------

    async def _on_list_tools(self) -> list[dict[str, Any]]:
        """Return the visible tools as ``[{name, description}, ...]``."""
        return [
            {"name": name, "description": self.server.tool_description(name)}
            for name in self.server.list_tool_names()
        ]

    async def _on_call_tool(
        self, name: str, arguments: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Dispatch a tool call through the bridge."""
        return await self.server.call_tool(name, arguments, peer="sdk")

    async def _on_list_resources(self) -> list[dict[str, str]]:
        """Return the resource URIs as ``[{uri}, ...]``."""
        return [{"uri": uri} for uri in self.server.list_resource_uris()]

    async def _on_read_resource(self, uri: str) -> dict[str, Any]:
        """Read a resource by URI through the server."""
        return await self.server.read_resource(uri, peer="sdk")

    async def _on_list_prompts(self) -> list[dict[str, str]]:
        """Return the available prompt names as ``[{name}, ...]``."""
        return [{"name": name} for name in self.server.list_prompt_names()]


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
