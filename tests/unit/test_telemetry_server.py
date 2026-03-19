"""Tests for TelemetryServer — REST endpoints, WebSocket, auth, CORS.

Uses aiohttp.test_utils for in-process testing (no real port binding).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, patch

import pytest

from mousedroid.config.schema import TelemetryConfig
from mousedroid.telemetry.protocol import TelemetryFrame

# Guard: skip all tests if aiohttp is not installed
aiohttp = pytest.importorskip("aiohttp")

from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from mousedroid.telemetry.log_buffer import LogRingBuffer  # noqa: E402
from mousedroid.telemetry.server import TelemetryServer  # noqa: E402


def _make_health_monitor():
    monitor = AsyncMock()
    monitor.check_health = AsyncMock(
        return_value={
            "status": "ok",
            "gpu_temp_c": 45.0,
            "gpu_load_pct": 30.0,
        }
    )
    return monitor


def _make_server(
    cfg: TelemetryConfig | None = None,
    log_buffer: LogRingBuffer | None = None,
) -> tuple[TelemetryServer, asyncio.Queue]:
    if cfg is None:
        cfg = TelemetryConfig(enabled=True)
    queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=64)
    health = _make_health_monitor()
    server = TelemetryServer(
        cfg=cfg,
        telemetry_queue=queue,
        health_monitor=health,
        log_buffer=log_buffer,
    )
    return server, queue


def _build_app(server: TelemetryServer) -> web.Application:
    """Build aiohttp app for testing without binding a port."""
    app = web.Application(middlewares=server._build_middlewares())
    server._register_routes(app)
    return app


# --- REST endpoint tests ---


async def test_status_endpoint():
    server, _queue = _make_server()
    app = _build_app(server)
    server._running = True

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/status")
        assert resp.status == 200
        data = await resp.json()
        assert "status" in data
        assert "uptime_s" in data
        assert "ws_clients" in data


async def test_sensors_endpoint_no_data():
    server, _queue = _make_server()
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/sensors")
        assert resp.status == 503


async def test_sensors_endpoint_with_data():
    server, _queue = _make_server()
    app = _build_app(server)
    server._latest_frame = TelemetryFrame(
        timestamp=1.0, distance_m=2.5, tick_count=42
    )

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/sensors")
        assert resp.status == 200
        data = await resp.json()
        assert data["timestamp"] == 1.0
        assert data["distance_m"] == 2.5
        assert data["tick_count"] == 42


async def test_health_endpoint():
    server, _queue = _make_server()
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"
        assert "gpu_temp_c" in data


async def test_health_endpoint_with_battery():
    server, _queue = _make_server()
    app = _build_app(server)
    server._latest_frame = TelemetryFrame(
        battery_voltage=11.8,
        safety={"is_emergency": False},
    )

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/health")
        data = await resp.json()
        assert data["battery_voltage"] == 11.8


async def test_logs_endpoint_disabled():
    server, _queue = _make_server(log_buffer=None)
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/logs")
        assert resp.status == 503


async def test_logs_endpoint_with_buffer():
    buf = LogRingBuffer(maxlen=100)
    buf(None, "info", {"event": "test_event", "level": "info"})
    server, _queue = _make_server(log_buffer=buf)
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/logs?n=10")
        assert resp.status == 200
        data = await resp.json()
        assert data["count"] >= 1
        assert len(data["entries"]) >= 1


async def test_logs_endpoint_invalid_n():
    """Query parameter 'n' must be an integer — invalid values return 400."""
    buf = LogRingBuffer(maxlen=100)
    server, _queue = _make_server(log_buffer=buf)
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/logs?n=abc")
        assert resp.status == 400
        data = await resp.json()
        assert data["error"] == "invalid_n"


async def test_logs_endpoint_negative_n():
    """Negative n is clamped to 0 — returns empty list."""
    buf = LogRingBuffer(maxlen=100)
    server, _queue = _make_server(log_buffer=buf)
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/logs?n=-5")
        assert resp.status == 200
        data = await resp.json()
        # n clamped to 0, so get_recent(0) returns []
        assert isinstance(data["entries"], list)


async def test_logs_endpoint_non_serializable_values():
    """Non-JSON-serializable values in log entries are stringified."""
    buf = LogRingBuffer(maxlen=100)
    buf(None, "info", {"event": "test", "obj": object()})
    server, _queue = _make_server(log_buffer=buf)
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/logs?n=10")
        assert resp.status == 200
        data = await resp.json()
        assert data["count"] >= 1
        # The non-serializable object should be stringified
        assert isinstance(data["entries"][0]["obj"], str)


async def test_network_endpoint():
    server, _queue = _make_server()
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/network")
        assert resp.status == 200
        data = await resp.json()
        assert "interfaces" in data
        assert "server_url" in data
        assert "server_port" in data


async def test_network_endpoint_mdns_name():
    cfg = TelemetryConfig(enabled=True, mdns_enabled=True)
    server, _queue = _make_server(cfg=cfg)
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/network")
        data = await resp.json()
        assert "mdns_name" in data


# --- CORS tests ---


async def test_cors_headers_present():
    server, _queue = _make_server()
    app = _build_app(server)
    server._running = True

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/status")
        assert "Access-Control-Allow-Origin" in resp.headers


async def test_cors_specific_origin_allowed():
    """Only matching origins get reflected when specific origins are set."""
    cfg = TelemetryConfig(
        enabled=True,
        cors_origins=["http://localhost:3000"],
    )
    server, _queue = _make_server(cfg=cfg)
    app = _build_app(server)
    server._running = True

    async with TestClient(TestServer(app)) as client:
        resp = await client.get(
            "/api/v1/status",
            headers={"Origin": "http://localhost:3000"},
        )
        assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
        assert "Vary" in resp.headers


async def test_cors_specific_origin_denied():
    """Non-matching origins get no ACAO header when specific origins are set."""
    cfg = TelemetryConfig(
        enabled=True,
        cors_origins=["http://localhost:3000"],
    )
    server, _queue = _make_server(cfg=cfg)
    app = _build_app(server)
    server._running = True

    async with TestClient(TestServer(app)) as client:
        resp = await client.get(
            "/api/v1/status",
            headers={"Origin": "http://evil.com"},
        )
        assert "Access-Control-Allow-Origin" not in resp.headers


async def test_cors_options_preflight():
    """OPTIONS requests should get CORS headers."""
    server, _queue = _make_server()
    app = _build_app(server)
    server._running = True

    async with TestClient(TestServer(app)) as client:
        resp = await client.options("/api/v1/status")
        assert "Access-Control-Allow-Origin" in resp.headers
        assert "Access-Control-Allow-Methods" in resp.headers


# --- Auth middleware tests ---


async def test_auth_middleware_rejects_without_key():
    cfg = TelemetryConfig(enabled=True, api_key="secret")
    server, _queue = _make_server(cfg=cfg)
    app = _build_app(server)
    server._running = True

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/status")
        assert resp.status == 401


async def test_auth_middleware_accepts_with_key():
    cfg = TelemetryConfig(enabled=True, api_key="secret")
    server, _queue = _make_server(cfg=cfg)
    app = _build_app(server)
    server._running = True

    async with TestClient(TestServer(app)) as client:
        resp = await client.get(
            "/api/v1/status",
            headers={"X-API-Key": "secret"},
        )
        assert resp.status == 200


async def test_auth_disabled_by_default():
    server, _queue = _make_server()
    app = _build_app(server)
    server._running = True

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/status")
        assert resp.status == 200


# --- WebSocket tests ---


async def test_websocket_connection():
    server, _queue = _make_server()
    app = _build_app(server)
    server._running = True

    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws"):
            assert len(server._ws_clients) == 1
        # After disconnect
        await asyncio.sleep(0.1)
        assert len(server._ws_clients) == 0


async def test_websocket_max_clients():
    cfg = TelemetryConfig(enabled=True, max_clients=1)
    server, _queue = _make_server(cfg=cfg)
    app = _build_app(server)
    server._running = True

    async with (  # noqa: SIM117
        TestClient(TestServer(app)) as client,
        client.ws_connect("/ws"),
    ):
        # Second connection should be rejected
        async with client.ws_connect("/ws") as ws2:
            msg = await ws2.receive()
            assert msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
            )


async def test_websocket_receives_broadcast():
    server, queue = _make_server()
    app = _build_app(server)
    server._running = True

    # Start broadcast loop
    broadcast_task = asyncio.create_task(server._broadcast_loop())

    try:
        async with (
            TestClient(TestServer(app)) as client,
            client.ws_connect("/ws") as ws,
        ):
            # Push a frame to the queue
            frame = TelemetryFrame(timestamp=42.0, tick_count=7)
            await queue.put(frame)

            # Client should receive it
            msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
            assert msg.type == aiohttp.WSMsgType.TEXT
            data = json.loads(msg.data)
            assert data["timestamp"] == 42.0
            assert data["tick_count"] == 7
    finally:
        server._running = False
        broadcast_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await broadcast_task


async def test_websocket_auth_rejects_without_key():
    """WS connections are rejected when API key is configured but not provided."""
    cfg = TelemetryConfig(enabled=True, api_key="ws-secret")
    server, _queue = _make_server(cfg=cfg)
    app = _build_app(server)
    server._running = True

    async with (
        TestClient(TestServer(app)) as client,
        client.ws_connect("/ws") as ws,
    ):
        msg = await ws.receive()
        assert msg.type in (
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
        )


async def test_log_stream_ws_disabled():
    """Log stream WS returns close when log_buffer is None."""
    server, _queue = _make_server(log_buffer=None)
    app = _build_app(server)
    server._running = True

    async with (
        TestClient(TestServer(app)) as client,
        client.ws_connect("/api/v1/logs/stream") as ws,
    ):
        msg = await ws.receive()
        assert msg.type in (
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
        )


async def test_log_stream_ws_receives_entries():
    """Log stream WS receives live log entries."""
    buf = LogRingBuffer(maxlen=100)
    server, _queue = _make_server(log_buffer=buf)
    app = _build_app(server)
    server._running = True

    async with (
        TestClient(TestServer(app)) as client,
        client.ws_connect("/api/v1/logs/stream") as ws,
    ):
        # Push a log entry after client connects
        await asyncio.sleep(0.1)
        buf(None, "info", {"event": "live_entry", "level": "info"})
        await asyncio.sleep(0.1)

        msg = await asyncio.wait_for(ws.receive(), timeout=3.0)
        assert msg.type == aiohttp.WSMsgType.TEXT
        data = json.loads(msg.data)
        assert data["event"] == "live_entry"

        # Stop the server so WS handler exits
        server._running = False


# --- Server lifecycle tests ---


async def test_server_start_stop():
    """Test start() and stop() lifecycle."""
    cfg = TelemetryConfig(enabled=True, mdns_enabled=False)
    server, _queue = _make_server(cfg=cfg)

    await server.start()
    assert server.is_running is True

    await server.stop()
    assert server.is_running is False
    assert server.client_count == 0


async def test_server_stop_cancels_broadcast():
    """Stop cancels the broadcast task cleanly."""
    cfg = TelemetryConfig(enabled=True, mdns_enabled=False)
    server, _queue = _make_server(cfg=cfg)

    await server.start()
    assert server._broadcast_task is not None

    await server.stop()
    assert server._broadcast_task.cancelled() or server._broadcast_task.done()


def test_server_initial_state():
    server, _queue = _make_server()
    assert server.is_running is False
    assert server.client_count == 0


# --- mDNS tests ---


async def test_mdns_register_without_zeroconf():
    """mDNS registration logs warning when zeroconf not installed."""
    server, _queue = _make_server()

    with patch.dict("sys.modules", {"zeroconf": None}):
        # Should not raise, just log warning
        await server._register_mdns()


async def test_mdns_unregister_noop_when_not_registered():
    """Unregister is safe when mDNS was never registered."""
    server, _queue = _make_server()
    server._zeroconf = None
    server._service_info = None
    await server._unregister_mdns()


# --- Broadcast dead client cleanup ---


async def test_broadcast_cleans_dead_clients():
    """Dead (closed) WS clients are removed during broadcast."""
    server, queue = _make_server()
    app = _build_app(server)
    server._running = True

    broadcast_task = asyncio.create_task(server._broadcast_loop())

    try:
        async with TestClient(TestServer(app)) as client:
            async with client.ws_connect("/ws"):
                assert len(server._ws_clients) == 1
            # Client disconnected — ws is now closed
            await asyncio.sleep(0.1)

            # Push a frame to trigger dead-client cleanup
            frame = TelemetryFrame(timestamp=99.0)
            await queue.put(frame)
            await asyncio.sleep(0.2)

            assert len(server._ws_clients) == 0
    finally:
        server._running = False
        broadcast_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await broadcast_task
