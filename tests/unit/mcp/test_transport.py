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
