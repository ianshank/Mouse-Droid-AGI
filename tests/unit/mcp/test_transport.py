"""Unit tests for the MCP SDK transport adapter."""

from __future__ import annotations

import sys

import pytest

from mousedroid.common.tools.registry import ToolRegistry, ToolSpec
from mousedroid.config.schema import MCPConfig, Settings
from mousedroid.mcp.server import MouseDroidMCPServer


def _registry() -> ToolRegistry:
    """Build a minimal registry with one health_check tool."""
    reg = ToolRegistry()

    async def _ok() -> dict[str, str]:
        return {"status": "ok"}

    reg.register(ToolSpec("health_check", "Liveness probe", _ok))
    return reg


def _make_server(
    mcp_cfg: MCPConfig, root_settings: Settings, safety_monitor: object
) -> MouseDroidMCPServer:
    return MouseDroidMCPServer(
        cfg=mcp_cfg,
        root_cfg=root_settings,
        tool_registry=_registry(),
        safety_monitor=safety_monitor,
    )


class TestBuildTransportAdapter:
    """``build_transport_adapter`` returns ``None`` without SDK, instance with SDK."""

    async def test_returns_none_when_sdk_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mcp_cfg: MCPConfig,
        root_settings: Settings,
        safe_safety_monitor: object,
    ) -> None:
        """Adapter is None when the optional `mcp` SDK is not importable."""
        monkeypatch.setitem(sys.modules, "mcp", None)
        monkeypatch.setitem(sys.modules, "mcp.server", None)
        from mousedroid.mcp.transport import build_transport_adapter

        server = _make_server(mcp_cfg, root_settings, safe_safety_monitor)
        adapter = build_transport_adapter(server)
        assert adapter is None

    async def test_returns_instance_when_sdk_present(
        self,
        mcp_cfg: MCPConfig,
        root_settings: Settings,
        safe_safety_monitor: object,
    ) -> None:
        """Adapter wraps the server when the SDK is importable."""
        from mousedroid.mcp.transport import MCPTransportAdapter, build_transport_adapter

        server = _make_server(mcp_cfg, root_settings, safe_safety_monitor)
        adapter = build_transport_adapter(server)
        assert isinstance(adapter, MCPTransportAdapter)
        assert adapter.server is server
        assert adapter.sdk_server is not None


class TestCallbacks:
    """The adapter's ``_on_*`` callbacks delegate to the server's bridge."""

    async def test_list_tools_callback_returns_visible_tools(
        self,
        mcp_cfg: MCPConfig,
        root_settings: Settings,
        safe_safety_monitor: object,
    ) -> None:
        from mousedroid.mcp.transport import build_transport_adapter

        server = _make_server(mcp_cfg, root_settings, safe_safety_monitor)
        adapter = build_transport_adapter(server)
        assert adapter is not None
        tools = await adapter._on_list_tools()
        assert [t["name"] for t in tools] == ["health_check"]
        assert all("description" in t for t in tools)

    async def test_call_tool_callback_dispatches_through_bridge(
        self,
        mcp_cfg: MCPConfig,
        root_settings: Settings,
        safe_safety_monitor: object,
    ) -> None:
        from mousedroid.mcp.transport import build_transport_adapter

        server = _make_server(mcp_cfg, root_settings, safe_safety_monitor)
        adapter = build_transport_adapter(server)
        assert adapter is not None
        result = await adapter._on_call_tool("health_check", {})
        assert result["status"] in {"ok", "error"}

    async def test_read_resource_callback_routes_by_uri(
        self,
        mcp_cfg: MCPConfig,
        root_settings: Settings,
        safe_safety_monitor: object,
    ) -> None:
        from mousedroid.mcp.transport import build_transport_adapter

        server = _make_server(mcp_cfg, root_settings, safe_safety_monitor)
        adapter = build_transport_adapter(server)
        assert adapter is not None
        payload = await adapter._on_read_resource("mousedroid://config/redacted")
        assert isinstance(payload, dict)

    async def test_list_resources_callback_returns_uris(
        self,
        mcp_cfg: MCPConfig,
        root_settings: Settings,
        safe_safety_monitor: object,
    ) -> None:
        from mousedroid.mcp.transport import build_transport_adapter

        server = _make_server(mcp_cfg, root_settings, safe_safety_monitor)
        adapter = build_transport_adapter(server)
        assert adapter is not None
        items = await adapter._on_list_resources()
        assert all("uri" in item for item in items)
        assert any("mousedroid://" in item["uri"] for item in items)

    async def test_list_prompts_callback_returns_names(
        self,
        mcp_cfg: MCPConfig,
        root_settings: Settings,
        safe_safety_monitor: object,
    ) -> None:
        from mousedroid.mcp.transport import build_transport_adapter

        server = _make_server(mcp_cfg, root_settings, safe_safety_monitor)
        adapter = build_transport_adapter(server)
        assert adapter is not None
        prompts = await adapter._on_list_prompts()
        assert all("name" in p for p in prompts)
