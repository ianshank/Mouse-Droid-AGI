from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from mousedroid.common.tools.registry import ToolRegistry, ToolSpec
from mousedroid.config.schema import MCPConfig, Settings
from mousedroid.mcp.protocol import MCPServerProtocol
from mousedroid.mcp.server import MouseDroidMCPServer, _auth_required


def _registry() -> ToolRegistry:
    reg = ToolRegistry()

    async def _ok() -> dict[str, str]:
        return {"status": "ok"}

    reg.register(ToolSpec("health_check", "ok", _ok))
    return reg


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_then_stop(
        self, mcp_cfg: MCPConfig, root_settings: Settings, safe_safety_monitor
    ) -> None:
        s = MouseDroidMCPServer(
            cfg=mcp_cfg,
            root_cfg=root_settings,
            tool_registry=_registry(),
            safety_monitor=safe_safety_monitor,
        )
        assert isinstance(s, MCPServerProtocol)
        assert s.is_running is False
        await s.start()
        assert s.is_running is True
        await s.stop()
        assert s.is_running is False

    @pytest.mark.asyncio
    async def test_start_idempotent(
        self, mcp_cfg: MCPConfig, root_settings: Settings, safe_safety_monitor
    ) -> None:
        s = MouseDroidMCPServer(
            cfg=mcp_cfg,
            root_cfg=root_settings,
            tool_registry=_registry(),
            safety_monitor=safe_safety_monitor,
        )
        await s.start()
        await s.start()  # second call should no-op
        assert s.is_running is True
        await s.stop()

    @pytest.mark.asyncio
    async def test_stop_idempotent(
        self, mcp_cfg: MCPConfig, root_settings: Settings, safe_safety_monitor
    ) -> None:
        s = MouseDroidMCPServer(
            cfg=mcp_cfg,
            root_cfg=root_settings,
            tool_registry=_registry(),
            safety_monitor=safe_safety_monitor,
        )
        await s.stop()  # before start
        await s.start()
        await s.stop()
        await s.stop()  # double-stop

    @pytest.mark.asyncio
    async def test_missing_token_for_remote_raises_at_start(
        self,
        root_settings: Settings,
        safe_safety_monitor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MOUSEDROID_MCP_TOKEN", "x")
        any_iface = "0.0.0.0"  # noqa: S104
        cfg = MCPConfig.model_validate({"enabled": True, "transport": "sse", "host": any_iface})
        # Now drop the token so start() fails fast on the missing secret.
        monkeypatch.delenv("MOUSEDROID_MCP_TOKEN", raising=False)
        s = MouseDroidMCPServer(
            cfg=cfg,
            root_cfg=root_settings,
            tool_registry=_registry(),
            safety_monitor=safe_safety_monitor,
        )
        with pytest.raises(RuntimeError, match="MOUSEDROID_MCP_TOKEN"):
            await s.start()


class TestSurface:
    @pytest.mark.asyncio
    async def test_call_tool_health_check(
        self, mcp_cfg: MCPConfig, root_settings: Settings, safe_safety_monitor
    ) -> None:
        s = MouseDroidMCPServer(
            cfg=mcp_cfg,
            root_cfg=root_settings,
            tool_registry=_registry(),
            safety_monitor=safe_safety_monitor,
        )
        out = await s.call_tool("health_check")
        assert out["status"] == "ok"
        assert out["payload"] == {"status": "ok"}
        assert out["error"] is None

    @pytest.mark.asyncio
    async def test_read_resource_unknown_uri(
        self, mcp_cfg: MCPConfig, root_settings: Settings, safe_safety_monitor
    ) -> None:
        s = MouseDroidMCPServer(
            cfg=mcp_cfg,
            root_cfg=root_settings,
            tool_registry=_registry(),
            safety_monitor=safe_safety_monitor,
        )
        with pytest.raises(KeyError):
            await s.read_resource("mousedroid://nonsense/path")

    @pytest.mark.asyncio
    async def test_read_resource_config_redacted(
        self, mcp_cfg: MCPConfig, root_settings: Settings, safe_safety_monitor
    ) -> None:
        s = MouseDroidMCPServer(
            cfg=mcp_cfg,
            root_cfg=root_settings,
            tool_registry=_registry(),
            safety_monitor=safe_safety_monitor,
        )
        out = await s.read_resource("mousedroid://config/redacted")
        assert "settings" in out

    @pytest.mark.asyncio
    async def test_telemetry_resource_disabled_without_publisher(
        self, mcp_cfg: MCPConfig, root_settings: Settings, safe_safety_monitor
    ) -> None:
        s = MouseDroidMCPServer(
            cfg=mcp_cfg,
            root_cfg=root_settings,
            tool_registry=_registry(),
            safety_monitor=safe_safety_monitor,
        )
        with pytest.raises(PermissionError):
            await s.read_resource("mousedroid://telemetry/latest")

    def test_list_prompt_names(
        self, mcp_cfg: MCPConfig, root_settings: Settings, safe_safety_monitor
    ) -> None:
        s = MouseDroidMCPServer(
            cfg=mcp_cfg,
            root_cfg=root_settings,
            tool_registry=_registry(),
            safety_monitor=safe_safety_monitor,
        )
        names = s.list_prompt_names()
        assert "diagnose-last-failure" in names
        assert "summarise-recent-telemetry" in names

    def test_list_resource_uris_includes_config_only_by_default(
        self, mcp_cfg: MCPConfig, root_settings: Settings, safe_safety_monitor
    ) -> None:
        s = MouseDroidMCPServer(
            cfg=mcp_cfg,
            root_cfg=root_settings,
            tool_registry=_registry(),
            safety_monitor=safe_safety_monitor,
        )
        uris = s.list_resource_uris()
        # No telemetry / log / memory wired in this fixture
        assert "mousedroid://config/redacted" in uris


class TestAuthRequired:
    def test_stdio_does_not_require(self) -> None:
        cfg = MCPConfig(enabled=True, transport="stdio", host="127.0.0.1")
        assert _auth_required(cfg) is False

    def test_loopback_does_not_require(self) -> None:
        cfg = MCPConfig(enabled=True, transport="sse", host="127.0.0.1")
        assert _auth_required(cfg) is False

    def test_remote_requires(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MOUSEDROID_MCP_TOKEN", "x")
        cfg = MCPConfig.model_validate({"enabled": True, "transport": "sse", "host": "10.0.0.1"})
        assert _auth_required(cfg) is True


class TestSamplerLoop:
    @pytest.mark.asyncio
    async def test_sampler_drains_publisher_queue(
        self, root_settings: Settings, safe_safety_monitor
    ) -> None:
        from mousedroid.telemetry.protocol import TelemetryFrame

        cfg = MCPConfig.model_validate(
            {
                "enabled": True,
                "transport": "stdio",
                "sample_telemetry_hz": 50.0,
                "resources": {"telemetry_enabled": True},
            }
        )
        queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue()
        for i in range(3):
            queue.put_nowait(TelemetryFrame(timestamp=float(i)))
        publisher = MagicMock()
        publisher.get_queue.return_value = queue
        s = MouseDroidMCPServer(
            cfg=cfg,
            root_cfg=root_settings,
            tool_registry=_registry(),
            safety_monitor=safe_safety_monitor,
            telemetry_publisher=publisher,
        )
        await s.start()
        # Allow the sampler at least one tick.
        await asyncio.sleep(0.1)
        out = await s.read_resource("mousedroid://telemetry/latest")
        await s.stop()
        assert out["frame"] is not None
        assert out["frame"]["timestamp"] == 2.0
