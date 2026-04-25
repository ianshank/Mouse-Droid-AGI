"""Integration test: MCP transport adapter end-to-end against the SDK.

Uses the SDK's in-memory transport (``create_connected_server_and_client_session``)
so the round-trip exercises the same handler-registration code path the
real stdio transport uses, without the stdout/stderr routing complexity
of spawning a subprocess.

Marked ``slow`` because instantiating an SDK session is heavier than a
unit-level test; CI runs it under the dedicated slow-tests stage.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ``importorskip`` only catches ``ImportError``; the SDK's chained
# imports can surface other exceptions (e.g. ``KeyError`` from Pydantic
# generics) under unusual sys.modules states such as the pre-commit
# coverage harness. Wrap the whole probe so any failure skips the file.
try:
    import mcp  # noqa: F401
    from mcp.shared.memory import (
        create_connected_server_and_client_session,
    )
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"mcp SDK unavailable: {exc!r}",
        allow_module_level=True,
    )

from mousedroid.common.tools.registry import ToolRegistry, ToolSpec
from mousedroid.config.schema import MCPConfig, Settings
from mousedroid.mcp.server import MouseDroidMCPServer
from mousedroid.mcp.transport import MCPTransportAdapter, build_transport_adapter


def _registry() -> ToolRegistry:
    """Minimal registry exposing health_check for the round-trip."""
    reg = ToolRegistry()

    async def _ok() -> dict[str, str]:
        return {"status": "ok"}

    reg.register(ToolSpec("health_check", "Liveness probe", _ok))
    return reg


def _safe_safety_monitor() -> MagicMock:
    """Replicates the unit-suite fixture without depending on its conftest."""
    monitor = MagicMock()
    monitor.evaluate.return_value = MagicMock(is_emergency=False, violations=[])
    return monitor


@pytest.mark.slow
@pytest.mark.asyncio
async def test_in_memory_round_trip_lists_and_calls_tools() -> None:
    """Verify the adapter's handlers are reachable via the SDK client API.

    The handlers go through the real ``mcp.server.Server`` decorator
    machinery, so this catches signature mismatches that unit tests
    can miss.
    """
    cfg = MCPConfig.model_validate({"enabled": True, "transport": "stdio"})
    root = Settings.model_validate({"mock_hardware": True})
    server = MouseDroidMCPServer(
        cfg=cfg,
        root_cfg=root,
        tool_registry=_registry(),
        safety_monitor=_safe_safety_monitor(),
    )
    adapter = build_transport_adapter(server)
    assert isinstance(adapter, MCPTransportAdapter)
    adapter._register_handlers()

    async with create_connected_server_and_client_session(adapter.sdk_server) as session:
        listed = await session.list_tools()
        names = [t.name for t in listed.tools]
        assert "health_check" in names

        result = await session.call_tool("health_check", arguments={})
        assert result.isError is False
