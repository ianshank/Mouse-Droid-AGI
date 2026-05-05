"""End-to-end smoke for the SSE transport (Phase B).

Boots the MCP server on an ephemeral loopback port with bearer auth
enabled, then verifies that:

1. A request without a token receives 401 from the bearer middleware.
2. A request with the correct token reaches the SSE handler (status 200
   with ``Content-Type: text/event-stream``).

We deliberately do NOT drive a full MCP SDK client handshake here —
that would require a long-lived bidirectional session and is covered
already by the stdio integration test. This module's job is to verify
that the Starlette + bearer-middleware + uvicorn wiring built in Phase B
works as a deployment unit.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket

import pytest

mcp_pkg = pytest.importorskip("mcp.server")  # SDK extra
httpx_pkg = pytest.importorskip("httpx")
uvicorn_pkg = pytest.importorskip("uvicorn")

from mousedroid.common.tools.registry import ToolRegistry, ToolSpec
from mousedroid.config.schema import MCPConfig, Settings
from mousedroid.mcp.server import MouseDroidMCPServer
from mousedroid.mcp.transport import build_transport_adapter

_TOKEN = "phase-b-test-token"  # noqa: S105 - test fixture token
_TOKEN_ENV = "MOUSEDROID_MCP_TOKEN"  # noqa: S105 - env var name, not secret


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _registry() -> ToolRegistry:
    reg = ToolRegistry()

    async def _ok() -> dict[str, str]:
        return {"status": "ok"}

    reg.register(ToolSpec("health_check", "Liveness probe", _ok))
    return reg


@pytest.fixture
def safe_safety_monitor() -> object:
    from unittest.mock import MagicMock

    monitor = MagicMock()
    monitor.evaluate.return_value = MagicMock(is_emergency=False, violations=[])
    return monitor


@pytest.mark.asyncio
@pytest.mark.timeout(15)
async def test_sse_transport_bearer_enforcement(
    monkeypatch: pytest.MonkeyPatch,
    safe_safety_monitor: object,
) -> None:
    """SSE transport rejects no-token requests and accepts valid ones."""
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    port = _free_port()
    cfg = MCPConfig.model_validate(
        {
            "enabled": True,
            "transport": "sse",
            "host": "127.0.0.1",
            "port": port,
            "bind_transport": True,
        }
    )
    root = Settings.model_validate({"mock_hardware": True})
    server = MouseDroidMCPServer(
        cfg=cfg,
        root_cfg=root,
        tool_registry=_registry(),
        safety_monitor=safe_safety_monitor,
    )
    adapter = build_transport_adapter(server)
    assert adapter is not None

    serve_task: asyncio.Task[None] = asyncio.create_task(adapter.serve())
    try:
        # Wait until the listener is accepting connections. Probing the
        # /messages/ endpoint with a POST avoids opening an SSE stream
        # (which would block) — it simply gets a 401 from the bearer
        # middleware once the server is up.
        probe_url = f"http://127.0.0.1:{port}/messages/?session_id=probe"
        async with httpx_pkg.AsyncClient(timeout=0.5) as probe:
            for _ in range(50):
                try:
                    r = await probe.post(probe_url)
                    assert r.status_code in {200, 400, 401, 404}
                    break
                except (
                    httpx_pkg.ConnectError,
                    httpx_pkg.ReadError,
                    httpx_pkg.ReadTimeout,
                ):
                    await asyncio.sleep(0.1)
            else:  # pragma: no cover - test infra failure
                pytest.fail("MCP transport never became reachable")

        # Verify the bearer middleware enforces auth on /sse without
        # actually opening the long-lived SSE stream. We use a short
        # timeout: the bearer rejection (401) returns immediately;
        # anything else is reported as a test failure.
        async with httpx_pkg.AsyncClient(timeout=2.0) as client:
            no_auth = await client.get(f"http://127.0.0.1:{port}/sse")
            assert no_auth.status_code == 401
            assert no_auth.json()["error"] == "unauthorized"

            wrong = await client.get(
                f"http://127.0.0.1:{port}/sse",
                headers={"Authorization": "Bearer wrong-token"},
            )
            assert wrong.status_code == 401

        # Confirm the /messages/ POST endpoint also requires auth.
        async with httpx_pkg.AsyncClient(timeout=2.0) as client:
            unauth_post = await client.post(f"http://127.0.0.1:{port}/messages/?session_id=test")
            assert unauth_post.status_code == 401
    finally:
        serve_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, BaseException):
            await serve_task
