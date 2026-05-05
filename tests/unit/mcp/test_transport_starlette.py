"""Unit tests for ``MCPTransportAdapter._build_starlette_app``.

The integration test in ``tests/integration/test_mcp_sse_e2e.py`` exercises
the full uvicorn lifecycle, but its coverage is invisible to ``pytest-cov``
because the server runs inside a separate task group. These unit tests
construct the Starlette app directly so the routing and middleware
composition are verified — and counted toward coverage — without binding
a port.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

mcp_pkg = pytest.importorskip("mcp.server")  # SDK extra
starlette_pkg = pytest.importorskip("starlette")

from mousedroid.config.schema import MCPConfig, Settings


@pytest.fixture
def safe_safety_monitor() -> Any:
    monitor = MagicMock()
    monitor.evaluate.return_value = MagicMock(is_emergency=False, violations=[])
    return monitor


def _build_adapter(cfg: MCPConfig, root: Settings, monitor: Any) -> Any:
    from mousedroid.common.tools.registry import ToolRegistry, ToolSpec
    from mousedroid.mcp.server import MouseDroidMCPServer
    from mousedroid.mcp.transport import build_transport_adapter

    registry = ToolRegistry()

    async def _ok() -> dict[str, str]:
        return {"status": "ok"}

    registry.register(ToolSpec("health_check", "probe", _ok))
    server = MouseDroidMCPServer(
        cfg=cfg, root_cfg=root, tool_registry=registry, safety_monitor=monitor
    )
    return build_transport_adapter(server)


def test_build_starlette_app_for_sse_has_sse_route_and_messages_mount(
    safe_safety_monitor: Any,
) -> None:
    cfg = MCPConfig.model_validate({"enabled": True, "transport": "sse", "host": "127.0.0.1"})
    root = Settings.model_validate({"mock_hardware": True})
    adapter = _build_adapter(cfg, root, safe_safety_monitor)
    assert adapter is not None

    validator = adapter._build_validator()
    app = adapter._build_starlette_app("sse", validator)

    # Starlette stores Mount paths without their trailing slash, so
    # ``Mount("/messages/", ...)`` shows up as ``"/messages"``.
    paths = [getattr(r, "path", None) for r in app.routes]
    assert "/sse" in paths
    assert "/messages" in paths


def test_build_starlette_app_for_streamable_http_has_root_mount_and_lifespan(
    safe_safety_monitor: Any,
) -> None:
    cfg = MCPConfig.model_validate(
        {"enabled": True, "transport": "streamable_http", "host": "127.0.0.1"}
    )
    root = Settings.model_validate({"mock_hardware": True})
    adapter = _build_adapter(cfg, root, safe_safety_monitor)
    assert adapter is not None

    validator = adapter._build_validator()
    app = adapter._build_starlette_app("streamable_http", validator)

    # Starlette normalises the root Mount path to ``""`` (trailing
    # slashes are stripped). Either form satisfies the wiring contract.
    paths = [getattr(r, "path", None) for r in app.routes]
    assert "" in paths or "/" in paths
    # Lifespan is wired (Starlette stores it as the app's router lifespan).
    assert app.router.lifespan_context is not None


def test_build_validator_required_when_token_set(
    monkeypatch: pytest.MonkeyPatch, safe_safety_monitor: Any
) -> None:
    """A configured token forces ``required=True`` even on loopback."""
    monkeypatch.setenv("MOUSEDROID_MCP_TOKEN", "secret")
    cfg = MCPConfig.model_validate({"enabled": True, "transport": "sse", "host": "127.0.0.1"})
    root = Settings.model_validate({"mock_hardware": True})
    adapter = _build_adapter(cfg, root, safe_safety_monitor)
    assert adapter is not None
    validator = adapter._build_validator()
    assert validator.required is True
    assert validator.has_secret is True


def test_build_validator_optional_when_token_absent(
    monkeypatch: pytest.MonkeyPatch, safe_safety_monitor: Any
) -> None:
    """No token + loopback bind → validator is non-required (dev mode)."""
    monkeypatch.delenv("MOUSEDROID_MCP_TOKEN", raising=False)
    cfg = MCPConfig.model_validate({"enabled": True, "transport": "sse", "host": "127.0.0.1"})
    root = Settings.model_validate({"mock_hardware": True})
    adapter = _build_adapter(cfg, root, safe_safety_monitor)
    assert adapter is not None
    validator = adapter._build_validator()
    assert validator.required is False
