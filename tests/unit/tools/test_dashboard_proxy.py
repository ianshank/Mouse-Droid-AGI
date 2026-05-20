"""Tests for ``tools/dashboard_proxy.py`` — the workstation-side reverse proxy.

The proxy forwards HTTP + WebSocket traffic from a local port to the live
Jetson telemetry server at ``192.168.55.1:8080`` (or any configured
upstream), injecting a bearer token when one is supplied. Adding it under
``tests/unit/tools/`` keeps the existing pattern set by
``test_llm_latency_probe.py`` (lightweight import + behaviour checks against
an in-process aiohttp upstream).

These tests stand up an in-process aiohttp upstream + spawn the proxy as
an aiohttp ``AppRunner`` so we never need to bind to the rover's actual
192.168.55.1 — the proxy itself is configuration-driven via CLI args / env
vars so the test can repoint it at ``127.0.0.1:<ephemeral>``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROXY_PATH = _REPO_ROOT / "tools" / "dashboard_proxy.py"


def _load_dashboard_proxy_module(monkeypatch: pytest.MonkeyPatch):
    """Import ``tools/dashboard_proxy.py`` as a module with controlled argv/env.

    The module reads sys.argv at import time (CLI-arg parsing happens in
    :func:`_resolve_settings`), so we set argv FIRST then re-import via
    ``importlib`` so each test gets a fresh module-level config.
    """
    # Clear any cached import so the module re-evaluates ``_resolve_settings``.
    sys.modules.pop("dashboard_proxy", None)
    spec = importlib.util.spec_from_file_location("dashboard_proxy", _PROXY_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dashboard_proxy"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_resolve_settings_cli_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI positional args (port, upstream, token) override env defaults."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["dashboard_proxy.py", "9999", "http://example.test:1234", "tok-cli"],
    )
    monkeypatch.delenv("JETSON_HTTP", raising=False)
    monkeypatch.delenv("JETSON_TOKEN", raising=False)
    monkeypatch.delenv("PROXY_PORT", raising=False)

    mod = _load_dashboard_proxy_module(monkeypatch)
    assert mod.PROXY_PORT == 9999
    assert mod.UPSTREAM_HTTP == "http://example.test:1234"
    assert mod.TOKEN == "tok-cli"  # noqa: S105 - test sentinel, not a real secret
    # WebSocket upstream is derived from the HTTP one.
    assert mod.UPSTREAM_WS == "ws://example.test:1234"
    # Auth header is populated when a token is configured.
    assert mod._AUTH_HEADER == {"Authorization": "Bearer tok-cli"}


def test_resolve_settings_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """No CLI args + env vars → fall back to env."""
    monkeypatch.setattr(sys, "argv", ["dashboard_proxy.py"])
    monkeypatch.setenv("PROXY_PORT", "9090")
    monkeypatch.setenv("JETSON_HTTP", "http://rover.local:5000/")  # trailing slash stripped
    monkeypatch.setenv("JETSON_TOKEN", "tok-env")

    mod = _load_dashboard_proxy_module(monkeypatch)
    assert mod.PROXY_PORT == 9090
    assert mod.UPSTREAM_HTTP == "http://rover.local:5000"  # rstripped
    assert mod.TOKEN == "tok-env"  # noqa: S105 - test sentinel, not a real secret


def test_resolve_settings_empty_token_skips_auth_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty bearer token → no ``Authorization`` header injected upstream.

    Useful for dashboards like Grafana / Prometheus that have their own auth
    and would reject (or be confused by) an inappropriate Bearer header.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        ["dashboard_proxy.py", "9091", "http://example.test:3000", ""],
    )
    monkeypatch.delenv("JETSON_TOKEN", raising=False)
    mod = _load_dashboard_proxy_module(monkeypatch)
    assert mod.TOKEN == ""
    assert mod._AUTH_HEADER == {}


def test_client_headers_drops_hop_by_hop_and_injects_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_client_headers`` strips hop-by-hop headers + adds the Authorization header."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["dashboard_proxy.py", "9092", "http://example.test:8080", "tok-xyz"],
    )
    mod = _load_dashboard_proxy_module(monkeypatch)

    # Build a minimal aiohttp Request stand-in via the multidict the real
    # Request exposes for ``.headers``. Using a CIMultiDict matches the
    # request's case-insensitive header lookup behaviour.
    from multidict import CIMultiDict

    request = type(
        "_FakeReq",
        (),
        {
            "headers": CIMultiDict(
                {
                    "Host": "127.0.0.1",
                    "Connection": "keep-alive",
                    "Transfer-Encoding": "chunked",
                    "Content-Type": "application/json",
                    "Authorization": "Bearer to-be-overridden",
                    "X-Forwarded-For": "192.0.2.1",
                }
            )
        },
    )()
    out = mod._client_headers(request)
    # Hop-by-hop headers (Connection, Transfer-Encoding, Host) are stripped.
    assert "Connection" not in out
    assert "Transfer-Encoding" not in out
    assert "Host" not in out
    # Non-hop-by-hop headers pass through.
    assert out.get("Content-Type") == "application/json"
    assert out.get("X-Forwarded-For") == "192.0.2.1"
    # Our Authorization injection overrides any client-supplied bearer.
    assert out.get("Authorization") == "Bearer tok-xyz"


def test_client_headers_with_no_token_does_not_inject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no token configured, the proxy passes the client's headers through unchanged."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["dashboard_proxy.py", "9093", "http://example.test:3000", ""],
    )
    mod = _load_dashboard_proxy_module(monkeypatch)

    from multidict import CIMultiDict

    request = type(
        "_FakeReq",
        (),
        {"headers": CIMultiDict({"Cookie": "grafana_session=abc"})},
    )()
    out = mod._client_headers(request)
    # No bearer injection — the client's cookie passes through as-is.
    assert out.get("Cookie") == "grafana_session=abc"
    assert "Authorization" not in out


# ---------------------------------------------------------------------------
# End-to-end: spin up an in-process upstream + proxy, GET through the proxy
# ---------------------------------------------------------------------------


async def _spin_up_servers(
    proxy_mod, upstream_handlers: dict[str, web.RequestHandler]
) -> tuple[str, str, web.AppRunner, web.AppRunner]:
    """Bind two ephemeral aiohttp servers — upstream + proxy — for one test."""
    upstream_app = web.Application()
    for route, handler in upstream_handlers.items():
        upstream_app.router.add_get(route, handler)
    upstream_runner = web.AppRunner(upstream_app)
    await upstream_runner.setup()
    upstream_site = web.TCPSite(upstream_runner, "127.0.0.1", 0)
    await upstream_site.start()
    upstream_port = upstream_site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    # Point the loaded module's upstream globals at the new test upstream.
    proxy_mod.UPSTREAM_HTTP = f"http://127.0.0.1:{upstream_port}"
    proxy_mod.UPSTREAM_WS = f"ws://127.0.0.1:{upstream_port}"

    proxy_app = web.Application()
    proxy_app.router.add_route("*", "/{path:.*}", proxy_mod._dispatch)
    proxy_app.on_startup.append(proxy_mod._on_startup)
    proxy_app.on_cleanup.append(proxy_mod._on_cleanup)
    proxy_runner = web.AppRunner(proxy_app)
    await proxy_runner.setup()
    proxy_site = web.TCPSite(proxy_runner, "127.0.0.1", 0)
    await proxy_site.start()
    proxy_port = proxy_site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    return (
        f"http://127.0.0.1:{proxy_port}",
        f"http://127.0.0.1:{upstream_port}",
        proxy_runner,
        upstream_runner,
    )


@pytest.mark.asyncio
async def test_http_get_round_trips_through_proxy_with_token_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: client → proxy → upstream sees the injected Authorization header."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["dashboard_proxy.py", "0", "http://127.0.0.1:0", "tok-e2e"],
    )
    mod = _load_dashboard_proxy_module(monkeypatch)

    seen_auth: list[str | None] = []

    async def upstream(request: web.Request) -> web.Response:
        seen_auth.append(request.headers.get("Authorization"))
        return web.json_response({"upstream_saw": "request"})

    proxy_url, _upstream_url, proxy_runner, upstream_runner = await _spin_up_servers(
        mod, {"/api/v1/health": upstream}
    )

    try:
        async with aiohttp.ClientSession() as s, s.get(f"{proxy_url}/api/v1/health") as r:
            assert r.status == 200
            body = await r.json()
            assert body == {"upstream_saw": "request"}
        # The upstream observed exactly the proxy-injected bearer.
        assert seen_auth == ["Bearer tok-e2e"]
    finally:
        await proxy_runner.cleanup()
        await upstream_runner.cleanup()


@pytest.mark.asyncio
async def test_http_get_404_status_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Upstream errors propagate cleanly through the proxy (no 502 wrapping)."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["dashboard_proxy.py", "0", "http://127.0.0.1:0", ""],
    )
    mod = _load_dashboard_proxy_module(monkeypatch)

    async def upstream_404(_req: web.Request) -> web.Response:
        return web.Response(status=404, text="not-found")

    proxy_url, _u, proxy_runner, upstream_runner = await _spin_up_servers(
        mod, {"/missing": upstream_404}
    )
    try:
        async with aiohttp.ClientSession() as s, s.get(f"{proxy_url}/missing") as r:
            assert r.status == 404
            assert await r.text() == "not-found"
    finally:
        await proxy_runner.cleanup()
        await upstream_runner.cleanup()


@pytest.mark.asyncio
async def test_http_streaming_response_chunks_flow_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming responses (MJPEG / SSE) chunk through without buffering all of it."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["dashboard_proxy.py", "0", "http://127.0.0.1:0", ""],
    )
    mod = _load_dashboard_proxy_module(monkeypatch)

    async def streaming(_req: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(_req)
        for i in range(3):
            await resp.write(f"data: chunk{i}\n\n".encode())
            await asyncio.sleep(0.01)
        await resp.write_eof()
        return resp

    proxy_url, _u, proxy_runner, upstream_runner = await _spin_up_servers(
        mod, {"/stream": streaming}
    )
    try:
        async with aiohttp.ClientSession() as s, s.get(f"{proxy_url}/stream") as r:
            assert r.status == 200
            # Drain all chunks — confirm we got 3 distinct payloads.
            received = b""
            async for chunk in r.content.iter_any():
                received += chunk
            assert b"chunk0" in received
            assert b"chunk1" in received
            assert b"chunk2" in received
    finally:
        await proxy_runner.cleanup()
        await upstream_runner.cleanup()


# ---------------------------------------------------------------------------
# WebSocket — covers the FIRST_COMPLETED cancellation fix from PR #104 review
# ---------------------------------------------------------------------------


async def _spin_up_ws_servers(
    proxy_mod, ws_handler: web.RequestHandler
) -> tuple[str, web.AppRunner, web.AppRunner]:
    """Bind upstream (WS) + proxy aiohttp servers for a WebSocket test.

    Identical topology to :func:`_spin_up_servers` but routes a WebSocket
    handler at ``/ws`` upstream so the proxy's ``_ws_handler`` path is
    exercised end-to-end.
    """
    upstream_app = web.Application()
    upstream_app.router.add_get("/ws", ws_handler)
    upstream_runner = web.AppRunner(upstream_app)
    await upstream_runner.setup()
    upstream_site = web.TCPSite(upstream_runner, "127.0.0.1", 0)
    await upstream_site.start()
    upstream_port = upstream_site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    proxy_mod.UPSTREAM_HTTP = f"http://127.0.0.1:{upstream_port}"
    proxy_mod.UPSTREAM_WS = f"ws://127.0.0.1:{upstream_port}"

    proxy_app = web.Application()
    proxy_app.router.add_route("*", "/{path:.*}", proxy_mod._dispatch)
    proxy_app.on_startup.append(proxy_mod._on_startup)
    proxy_app.on_cleanup.append(proxy_mod._on_cleanup)
    proxy_runner = web.AppRunner(proxy_app)
    await proxy_runner.setup()
    proxy_site = web.TCPSite(proxy_runner, "127.0.0.1", 0)
    await proxy_site.start()
    proxy_port = proxy_site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    return f"http://127.0.0.1:{proxy_port}", proxy_runner, upstream_runner


@pytest.mark.asyncio
async def test_websocket_text_message_round_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: client → proxy → upstream WS sees the text echo round-trip.

    Exercises ``_ws_handler``'s bidirectional pipe + the FIRST_COMPLETED
    cancellation path landed in PR #104's review-follow-up. Without that
    fix, the surviving pipe task hung indefinitely after the client closed,
    leaking pool slots — which is exactly what this test covers (the
    upstream sends its echo, the client closes, the test cleans up and
    proves the proxy task didn't deadlock).
    """
    monkeypatch.setattr(
        sys,
        "argv",
        ["dashboard_proxy.py", "0", "http://127.0.0.1:0", "tok-ws"],
    )
    mod = _load_dashboard_proxy_module(monkeypatch)

    async def upstream_ws(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await ws.send_str(f"echo:{msg.data}")
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                break
        return ws

    proxy_url, proxy_runner, upstream_runner = await _spin_up_ws_servers(mod, upstream_ws)
    ws_url = proxy_url.replace("http://", "ws://") + "/ws"
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.ws_connect(ws_url, headers={"Upgrade": "websocket"}) as ws,
        ):
            await ws.send_str("hello")
            msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
            assert msg.type == aiohttp.WSMsgType.TEXT
            assert msg.data == "echo:hello"
            await ws.close()
    finally:
        # Pin the bug-fix surface: cleanup must complete in well under the
        # asyncio.wait timeout below — if FIRST_COMPLETED hadn't been
        # applied, the surviving _pipe_client_to_server task would block
        # cleanup until aiohttp tore the upstream down forcibly.
        await asyncio.wait_for(proxy_runner.cleanup(), timeout=5.0)
        await asyncio.wait_for(upstream_runner.cleanup(), timeout=5.0)


@pytest.mark.asyncio
async def test_websocket_upstream_close_propagates_to_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the upstream WS closes first, the proxy propagates the close cleanly.

    Mirror of the previous test from the OPPOSITE side: this time the
    upstream sends one message then closes its own end. The proxy must
    cancel ``_pipe_server_to_client`` (which is blocked on the client's
    ``async for``) and close the client-side WS so the operator's browser
    sees the disconnect instead of hanging.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        ["dashboard_proxy.py", "0", "http://127.0.0.1:0", ""],
    )
    mod = _load_dashboard_proxy_module(monkeypatch)

    async def upstream_ws_close_after_one(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str("only-message")
        await ws.close()
        return ws

    proxy_url, proxy_runner, upstream_runner = await _spin_up_ws_servers(
        mod, upstream_ws_close_after_one
    )
    ws_url = proxy_url.replace("http://", "ws://") + "/ws"
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.ws_connect(ws_url, headers={"Upgrade": "websocket"}) as ws,
        ):
            first = await asyncio.wait_for(ws.receive(), timeout=2.0)
            assert first.type == aiohttp.WSMsgType.TEXT
            assert first.data == "only-message"
            # Next receive should yield CLOSE / CLOSED, not hang forever.
            closing = await asyncio.wait_for(ws.receive(), timeout=2.0)
            assert closing.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
            )
    finally:
        await asyncio.wait_for(proxy_runner.cleanup(), timeout=5.0)
        await asyncio.wait_for(upstream_runner.cleanup(), timeout=5.0)
