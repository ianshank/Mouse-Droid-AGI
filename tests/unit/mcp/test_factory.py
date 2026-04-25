from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mousedroid.common.tools.registry import create_default_registry
from mousedroid.config.schema import Settings
from mousedroid.factory import build_mcp_server


def _safety_monitor(emergency: bool = False):
    monitor = MagicMock()
    monitor.evaluate.return_value = MagicMock(is_emergency=emergency)
    return monitor


class TestBuildMCPServer:
    def test_returns_none_when_mcp_not_configured(self) -> None:
        cfg = Settings.model_validate({"mock_hardware": True})
        out = build_mcp_server(
            cfg, tool_registry=create_default_registry(), safety_monitor=_safety_monitor()
        )
        assert out is None

    def test_returns_none_when_disabled(self) -> None:
        cfg = Settings.model_validate({"mock_hardware": True, "mcp": {"enabled": False}})
        out = build_mcp_server(
            cfg, tool_registry=create_default_registry(), safety_monitor=_safety_monitor()
        )
        assert out is None

    def test_returns_server_when_enabled(self) -> None:
        cfg = Settings.model_validate(
            {"mock_hardware": True, "mcp": {"enabled": True, "transport": "stdio"}}
        )
        out = build_mcp_server(
            cfg, tool_registry=create_default_registry(), safety_monitor=_safety_monitor()
        )
        assert out is not None
        assert out.is_running is False

    def test_port_collision_raises(self) -> None:
        cfg = Settings.model_validate(
            {
                "mock_hardware": True,
                "telemetry": {"enabled": True, "port": 8765},
                "mcp": {
                    "enabled": True,
                    "transport": "sse",
                    "host": "127.0.0.1",
                    "port": 8765,
                },
            }
        )
        with pytest.raises(ValueError, match="collides"):
            build_mcp_server(
                cfg,
                tool_registry=create_default_registry(),
                safety_monitor=_safety_monitor(),
            )

    def test_stdio_transport_skips_port_collision_check(self) -> None:
        # stdio doesn't bind a port; collision is irrelevant.
        cfg = Settings.model_validate(
            {
                "mock_hardware": True,
                "telemetry": {"enabled": True, "port": 8765},
                "mcp": {"enabled": True, "transport": "stdio", "port": 8765},
            }
        )
        out = build_mcp_server(
            cfg, tool_registry=create_default_registry(), safety_monitor=_safety_monitor()
        )
        assert out is not None
