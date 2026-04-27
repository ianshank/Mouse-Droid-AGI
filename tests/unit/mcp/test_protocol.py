from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from mousedroid.mcp.protocol import MCPRequestContext, MCPServerProtocol, MCPToolResult


class TestProtocolShape:
    def test_protocol_is_runtime_checkable(self) -> None:
        # Real implementation
        from mousedroid.mcp.server import MouseDroidMCPServer  # noqa: F401

        # A bare object missing the protocol attrs should not satisfy it.
        class _Empty:
            pass

        assert not isinstance(_Empty(), MCPServerProtocol)

    def test_concrete_server_satisfies_protocol(
        self, mcp_cfg, root_settings, safe_safety_monitor
    ) -> None:
        from mousedroid.common.tools.registry import create_default_registry
        from mousedroid.mcp.server import MouseDroidMCPServer

        s = MouseDroidMCPServer(
            cfg=mcp_cfg,
            root_cfg=root_settings,
            tool_registry=create_default_registry(),
            safety_monitor=safe_safety_monitor,
        )
        assert isinstance(s, MCPServerProtocol)


class TestRequestContext:
    def test_default_token_present_false(self) -> None:
        ctx = MCPRequestContext(request_id="abc")
        assert ctx.token_present is False
        assert ctx.peer == "unknown"

    def test_frozen(self) -> None:
        ctx = MCPRequestContext(request_id="abc", peer="stdio")
        with pytest.raises(FrozenInstanceError):
            ctx.peer = "modified"  # type: ignore[misc]


class TestToolResult:
    def test_default_payload_is_empty_dict(self) -> None:
        r = MCPToolResult(status="ok")
        assert r.payload == {}
        assert r.error is None
        assert r.latency_ms == 0.0

    def test_frozen(self) -> None:
        r = MCPToolResult(status="ok")
        with pytest.raises(FrozenInstanceError):
            r.status = "denied"  # type: ignore[misc]

    @pytest.mark.parametrize(
        "status",
        [
            "ok",
            "denied",
            "refused_emergency",
            "timeout",
            "rate_limited",
            "error",
            "client_disconnected",
            "circuit_open",
            "actuation_disabled",
        ],
    )
    def test_status_labels_documented(self, status: str) -> None:
        r = MCPToolResult(status=status)
        assert r.status == status
