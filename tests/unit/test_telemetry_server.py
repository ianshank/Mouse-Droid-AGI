"""Tests for TelemetryServer — REST endpoints, WebSocket, auth, CORS.

Uses aiohttp.test_utils for in-process testing (no real port binding).
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mousedroid.config.schema import MetricsConfig, TelemetryConfig
from mousedroid.telemetry.metrics import MetricsRegistry
from mousedroid.telemetry.protocol import TelemetryFrame

# Guard: skip all tests if aiohttp is not installed
aiohttp = pytest.importorskip("aiohttp")
import contextlib  # noqa: E402

from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from mousedroid.constants import MAX_LOG_ENTRIES  # noqa: E402
from mousedroid.telemetry.log_buffer import LogRingBuffer  # noqa: E402
from mousedroid.telemetry.network import NetworkInterface  # noqa: E402
from mousedroid.telemetry.server import TelemetryServer  # noqa: E402


class _StubPublisher:
    def __init__(self, queue: asyncio.Queue[TelemetryFrame], frames_dropped: int = 0) -> None:
        self._queue = queue
        self._frames_dropped = frames_dropped

    async def publish(self, frame: TelemetryFrame) -> None:
        self._queue.put_nowait(frame)

    def get_queue(self) -> asyncio.Queue[TelemetryFrame]:
        return self._queue

    @property
    def stats(self) -> dict[str, int]:
        return {
            "frames_published": 0,
            "frames_dropped": self._frames_dropped,
        }


_STUB_IFACES = [NetworkInterface(name="eth0", ip="10.0.0.1", interface_type="ethernet", up=True)]
_STUB_IP = "10.0.0.1"


def _make_health_monitor() -> AsyncMock:
    monitor = AsyncMock()
    monitor.check_health = AsyncMock(
        return_value={"status": "ok", "gpu_temp_c": 45.0, "gpu_load_pct": 30.0}
    )
    return monitor


def _make_server(
    cfg: TelemetryConfig | None = None,
    log_buffer: LogRingBuffer | None = None,
    metrics_registry: MetricsRegistry | None = None,
    publisher: _StubPublisher | None = None,
) -> tuple[TelemetryServer, asyncio.Queue[TelemetryFrame]]:
    if cfg is None:
        cfg = TelemetryConfig(enabled=True)
    queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=64)
    health = _make_health_monitor()
    server = TelemetryServer(
        cfg=cfg,
        telemetry_queue=queue,
        health_monitor=health,
        log_buffer=log_buffer,
        metrics_registry=metrics_registry,
        publisher=publisher,
    )
    return server, queue


def _build_app(server: TelemetryServer) -> web.Application:
    """Build aiohttp app for testing without binding a port."""
    app = web.Application(middlewares=server._build_middlewares())
    server._register_routes(app)
    return app


# --- REST endpoint tests ---


async def test_status_endpoint() -> None:
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
        assert "tick_count" in data


async def test_status_endpoint_with_frame() -> None:
    """Status includes tick_count from the latest frame."""
    server, _queue = _make_server()
    app = _build_app(server)
    server._running = True
    server._latest_frame = TelemetryFrame(tick_count=99)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/status")
        data = await resp.json()
        assert data["tick_count"] == 99
        assert data["status"] == "running"


async def test_status_endpoint_stopped() -> None:
    """Status reports stopped when server is not running."""
    server, _queue = _make_server()
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/status")
        data = await resp.json()
        assert data["status"] == "stopped"


async def test_sensors_endpoint_no_data() -> None:
    server, _queue = _make_server()
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/sensors")
        assert resp.status == 503


async def test_sensors_endpoint_with_data() -> None:
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


async def test_health_endpoint() -> None:
    server, _queue = _make_server()
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"
        assert "gpu_temp_c" in data


async def test_health_endpoint_with_battery() -> None:
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
        assert data["safety"] == {"is_emergency": False}


async def test_logs_endpoint_disabled() -> None:
    server, _queue = _make_server(log_buffer=None)
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/logs")
        assert resp.status == 503


async def test_logs_endpoint_with_buffer() -> None:
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


async def test_logs_endpoint_invalid_n() -> None:
    """Invalid n parameter returns 400."""
    buf = LogRingBuffer(maxlen=100)
    server, _queue = _make_server(log_buffer=buf)
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/logs?n=abc")
        assert resp.status == 400
        data = await resp.json()
        assert data["error"] == "invalid_n"


async def test_logs_endpoint_negative_n() -> None:
    """Negative n is clamped to 0."""
    buf = LogRingBuffer(maxlen=100)
    buf(None, "info", {"event": "entry1"})
    server, _queue = _make_server(log_buffer=buf)
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/logs?n=-5")
        assert resp.status == 200
        data = await resp.json()
        assert data["count"] == 0


async def test_logs_endpoint_large_n_clamped() -> None:
    """n > MAX_LOG_ENTRIES is clamped to MAX_LOG_ENTRIES."""
    buf = LogRingBuffer(maxlen=100)
    buf(None, "info", {"event": "entry1"})
    server, _queue = _make_server(log_buffer=buf)
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get(f"/api/v1/logs?n={MAX_LOG_ENTRIES + 500}")
        assert resp.status == 200
        data = await resp.json()
        # Should return entries (clamped, not rejected)
        assert data["count"] >= 1


async def test_logs_endpoint_non_serializable_values() -> None:
    """Non-JSON-serializable values are converted to strings."""
    buf = LogRingBuffer(maxlen=100)
    buf(None, "info", {"event": "test", "obj": object()})
    server, _queue = _make_server(log_buffer=buf)
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/logs?n=10")
        assert resp.status == 200
        data = await resp.json()
        # The non-serializable object should be converted to str
        assert isinstance(data["entries"][0]["obj"], str)


async def test_logs_endpoint_default_n() -> None:
    """Default n is 50 when not specified."""
    buf = LogRingBuffer(maxlen=100)
    for i in range(60):
        buf(None, "info", {"n": i})
    server, _queue = _make_server(log_buffer=buf)
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/logs")
        assert resp.status == 200
        data = await resp.json()
        assert data["count"] == 50


async def test_network_endpoint() -> None:
    server, _queue = _make_server()
    app = _build_app(server)

    with (
        patch(
            "mousedroid.telemetry.server.get_network_interfaces",
            new=AsyncMock(return_value=_STUB_IFACES),
        ),
        patch("mousedroid.telemetry.server.get_default_ip", return_value=_STUB_IP),
    ):
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/v1/network")
            assert resp.status == 200
            data = await resp.json()
            assert "interfaces" in data
            assert "server_url" in data
            assert "server_port" in data


async def test_network_endpoint_mdns_name() -> None:
    cfg = TelemetryConfig(enabled=True, mdns_enabled=True)
    server, _queue = _make_server(cfg=cfg)
    app = _build_app(server)

    with (
        patch(
            "mousedroid.telemetry.server.get_network_interfaces",
            new=AsyncMock(return_value=_STUB_IFACES),
        ),
        patch("mousedroid.telemetry.server.get_default_ip", return_value=_STUB_IP),
    ):
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/v1/network")
            data = await resp.json()
            assert "mdns_name" in data


# --- Metrics endpoint tests ---


async def test_metrics_endpoint_exposed_when_registry_present() -> None:
    cfg = TelemetryConfig(enabled=True, metrics_path="/metrics")
    registry = MetricsRegistry(MetricsConfig(enabled=True))
    server, _queue = _make_server(cfg=cfg, metrics_registry=registry)
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/metrics")
        assert resp.status == 200
        text = await resp.text()
        assert "# HELP" in text
        content_type = resp.headers.get("Content-Type", "")
        assert "text/plain" in content_type


async def test_metrics_endpoint_not_registered_without_registry() -> None:
    """When no metrics registry is provided, /metrics should not be registered."""
    server, _queue = _make_server(metrics_registry=None)
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/metrics")
        assert resp.status == 404


# --- CORS tests ---


async def test_cors_headers_present() -> None:
    server, _queue = _make_server()
    app = _build_app(server)
    server._running = True

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/status")
        assert "Access-Control-Allow-Origin" in resp.headers


async def test_cors_unrestricted_default() -> None:
    """Default CORS allows all origins with '*'."""
    server, _queue = _make_server()
    app = _build_app(server)
    server._running = True

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/status")
        assert resp.headers["Access-Control-Allow-Origin"] == "*"


async def test_cors_specific_origin_allowed() -> None:
    """When specific origins are configured, only matching origins get the header."""
    cfg = TelemetryConfig(enabled=True, cors_origins=["http://example.com"])
    server, _queue = _make_server(cfg=cfg)
    app = _build_app(server)
    server._running = True

    async with TestClient(TestServer(app)) as client:
        resp = await client.get(
            "/api/v1/status",
            headers={"Origin": "http://example.com"},
        )
        assert resp.headers.get("Access-Control-Allow-Origin") == "http://example.com"
        assert "Vary" in resp.headers


async def test_cors_specific_origin_rejected() -> None:
    """Disallowed origin does not get Access-Control-Allow-Origin."""
    cfg = TelemetryConfig(enabled=True, cors_origins=["http://example.com"])
    server, _queue = _make_server(cfg=cfg)
    app = _build_app(server)
    server._running = True

    async with TestClient(TestServer(app)) as client:
        resp = await client.get(
            "/api/v1/status",
            headers={"Origin": "http://evil.com"},
        )
        assert "Access-Control-Allow-Origin" not in resp.headers


async def test_cors_options_request() -> None:
    """OPTIONS preflight returns correct CORS headers."""
    server, _queue = _make_server()
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.options("/api/v1/status")
        assert resp.status == 200
        assert "Access-Control-Allow-Methods" in resp.headers


# --- Auth middleware tests ---


async def test_auth_middleware_rejects_without_key() -> None:
    cfg = TelemetryConfig(enabled=True, api_key="secret")
    server, _queue = _make_server(cfg=cfg)
    app = _build_app(server)
    server._running = True

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/status")
        assert resp.status == 401


async def test_auth_middleware_accepts_with_key() -> None:
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


async def test_auth_disabled_by_default() -> None:
    server, _queue = _make_server()
    app = _build_app(server)
    server._running = True

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/v1/status")
        assert resp.status == 200


async def test_auth_ws_accepts_query_param() -> None:
    """WebSocket auth accepts api_key as query parameter."""
    cfg = TelemetryConfig(enabled=True, api_key="ws-secret")
    server, _queue = _make_server(cfg=cfg)
    app = _build_app(server)
    server._running = True

    async with TestClient(TestServer(app)) as client, client.ws_connect(
        "/ws?api_key=ws-secret"
    ):
        assert server.client_count == 1


# --- WebSocket tests ---


async def test_websocket_connection() -> None:
    server, _queue = _make_server()
    app = _build_app(server)
    server._running = True

    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws"):
            assert len(server._ws_clients) == 1
        # After disconnect
        await asyncio.sleep(0.1)
        assert len(server._ws_clients) == 0


async def test_websocket_max_clients() -> None:
    cfg = TelemetryConfig(enabled=True, max_clients=1)
    server, _queue = _make_server(cfg=cfg)
    app = _build_app(server)
    server._running = True

    async with (
        TestClient(TestServer(app)) as client,
        client.ws_connect("/ws"),
        client.ws_connect("/ws") as ws2,
    ):
        # Second connection should be rejected
        msg = await ws2.receive()
        assert msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED)


async def test_websocket_receives_broadcast() -> None:
    server, queue = _make_server()
    app = _build_app(server)
    server._running = True

    # Start broadcast loop
    broadcast_task = asyncio.create_task(server._broadcast_loop())

    try:
        async with TestClient(TestServer(app)) as client, client.ws_connect("/ws") as ws:
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


async def test_broadcast_loop_updates_metrics_registry() -> None:
    cfg = TelemetryConfig(enabled=True, metrics_path="/metrics")
    queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=64)
    health = _make_health_monitor()
    registry = MetricsRegistry(MetricsConfig(enabled=True))
    publisher = _StubPublisher(queue, frames_dropped=3)
    server = TelemetryServer(
        cfg=cfg,
        telemetry_queue=queue,
        health_monitor=health,
        metrics_registry=registry,
        publisher=publisher,
    )
    server._running = True

    broadcast_task = asyncio.create_task(server._broadcast_loop())
    try:
        await queue.put(
            TelemetryFrame(
                loop_time_ms=17.5,
                battery_voltage=12.3,
                health={"gpu_temp_c": 57.0},
                safety={"violations": ["law1", "law2"]},
            )
        )
        await asyncio.sleep(0.05)
    finally:
        server._running = False
        broadcast_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await broadcast_task

    text = registry.render_prometheus()
    assert "loop_time" in text
    assert "battery_voltage" in text
    assert "gpu_temp" in text
    assert 'law="law1"' in text
    assert 'law="law2"' in text
    assert "publish_hz" in text
    assert " 10" in text or " 10.0" in text
    assert "frame_drops_total 3" in text


async def test_broadcast_loop_removes_dead_clients() -> None:
    """Dead WebSocket clients are cleaned up during broadcast."""
    server, queue = _make_server()
    server._running = True

    # Create a mock dead WS client
    mock_ws = MagicMock()
    mock_ws.closed = True
    server._ws_clients.append(mock_ws)

    broadcast_task = asyncio.create_task(server._broadcast_loop())
    try:
        await queue.put(TelemetryFrame(timestamp=1.0))
        await asyncio.sleep(0.05)
        # Dead client should be removed
        assert mock_ws not in server._ws_clients
    finally:
        server._running = False
        broadcast_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await broadcast_task


async def test_broadcast_loop_msgpack_serialization() -> None:
    """Broadcast uses msgpack when configured."""
    msgpack = pytest.importorskip("msgpack")
    cfg = TelemetryConfig(enabled=True, serialization="msgpack")
    server, queue = _make_server(cfg=cfg)
    app = _build_app(server)
    server._running = True

    broadcast_task = asyncio.create_task(server._broadcast_loop())
    try:
        async with TestClient(TestServer(app)) as client, client.ws_connect("/ws") as ws:
            await queue.put(TelemetryFrame(timestamp=99.0, tick_count=5))
            msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
            assert msg.type == aiohttp.WSMsgType.BINARY
            data = msgpack.unpackb(msg.data, raw=False)
            assert data["timestamp"] == 99.0
    finally:
        server._running = False
        broadcast_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await broadcast_task


async def test_broadcast_loop_timeout_continues() -> None:
    """Broadcast loop continues when queue is empty (timeout)."""
    server, _queue = _make_server()
    server._running = True

    broadcast_task = asyncio.create_task(server._broadcast_loop())
    # Let it loop once with empty queue
    await asyncio.sleep(0.05)
    # Should still be running
    assert not broadcast_task.done()

    server._running = False
    broadcast_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await broadcast_task


async def test_broadcast_loop_metrics_without_gpu_temp() -> None:
    """Broadcast handles health dict without gpu_temp_c gracefully."""
    cfg = TelemetryConfig(enabled=True)
    registry = MetricsRegistry(MetricsConfig(enabled=True))
    server, queue = _make_server(cfg=cfg, metrics_registry=registry)
    server._running = True

    broadcast_task = asyncio.create_task(server._broadcast_loop())
    try:
        await queue.put(TelemetryFrame(health={"cpu_temp_c": 50.0}))  # no gpu_temp_c
        await asyncio.sleep(0.05)
    finally:
        server._running = False
        broadcast_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await broadcast_task


async def test_broadcast_loop_non_dict_health() -> None:
    """Broadcast handles non-dict health gracefully."""
    cfg = TelemetryConfig(enabled=True)
    registry = MetricsRegistry(MetricsConfig(enabled=True))
    server, queue = _make_server(cfg=cfg, metrics_registry=registry)
    server._running = True

    broadcast_task = asyncio.create_task(server._broadcast_loop())
    try:
        await queue.put(TelemetryFrame(health="unavailable"))
        await asyncio.sleep(0.05)
    finally:
        server._running = False
        broadcast_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await broadcast_task


async def test_sync_publisher_metrics_no_publisher() -> None:
    """_sync_publisher_metrics is a no-op when publisher is None."""
    registry = MetricsRegistry(MetricsConfig(enabled=True))
    server, _queue = _make_server(metrics_registry=registry)
    # Should not raise
    server._sync_publisher_metrics()


async def test_sync_publisher_metrics_no_registry() -> None:
    """_sync_publisher_metrics is a no-op when metrics is None."""
    server, _queue = _make_server()
    server._sync_publisher_metrics()


# --- Log stream WebSocket tests ---


async def test_log_stream_disabled() -> None:
    """Log stream WebSocket closes when log_buffer is None."""
    server, _queue = _make_server(log_buffer=None)
    app = _build_app(server)
    server._running = True

    async with (
        TestClient(TestServer(app)) as client,
        client.ws_connect("/api/v1/logs/stream") as ws,
    ):
        msg = await ws.receive()
        assert msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED)


async def test_log_stream_receives_entries() -> None:
    """Log stream WebSocket receives new log entries."""
    buf = LogRingBuffer(maxlen=100)
    server, _queue = _make_server(log_buffer=buf)
    app = _build_app(server)
    server._running = True

    async with (
        TestClient(TestServer(app)) as client,
        client.ws_connect("/api/v1/logs/stream") as ws,
    ):
        # Wait for subscription to set up
        await asyncio.sleep(0.05)
        # Push a log entry
        buf(None, "info", {"event": "live_entry", "level": "info"})
        # Receive it
        msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
        assert msg.type == aiohttp.WSMsgType.TEXT
        data = json.loads(msg.data)
        assert data["event"] == "live_entry"

    server._running = False


# --- mDNS tests ---


async def test_register_mdns_import_error() -> None:
    """mDNS registration handles missing zeroconf gracefully."""
    cfg = TelemetryConfig(enabled=True, mdns_enabled=True)
    server, _queue = _make_server(cfg=cfg)

    with patch.dict("sys.modules", {"zeroconf": None}):
        # Should not raise — logs warning
        await server._register_mdns()


async def test_register_mdns_exception() -> None:
    """mDNS registration handles general exceptions gracefully."""
    cfg = TelemetryConfig(enabled=True, mdns_enabled=True)
    server, _queue = _make_server(cfg=cfg)

    mock_zeroconf_mod = MagicMock()
    mock_zeroconf_mod.Zeroconf.side_effect = RuntimeError("fail")

    with patch.dict("sys.modules", {"zeroconf": mock_zeroconf_mod}), patch(
        "mousedroid.telemetry.server.get_default_ip",
        return_value="127.0.0.1",
    ):
        await server._register_mdns()


async def test_unregister_mdns_noop_when_none() -> None:
    """Unregistering mDNS when not registered is safe."""
    server, _queue = _make_server()
    server._zeroconf = None
    server._service_info = None
    await server._unregister_mdns()


async def test_unregister_mdns_with_exception() -> None:
    """Unregistering mDNS handles exceptions gracefully."""
    server, _queue = _make_server()
    mock_zc = MagicMock()
    mock_zc.unregister_service = MagicMock(side_effect=RuntimeError("fail"))
    server._zeroconf = mock_zc
    server._service_info = MagicMock()

    with patch("asyncio.to_thread", side_effect=RuntimeError("fail")):
        await server._unregister_mdns()
    # Should reset to None even after error
    assert server._zeroconf is None
    assert server._service_info is None


# --- Server lifecycle tests ---


def test_server_initial_state() -> None:
    server, _queue = _make_server()
    assert server.is_running is False
    assert server.client_count == 0


async def test_server_start_and_stop() -> None:
    """Full start/stop lifecycle on localhost."""
    cfg = TelemetryConfig(
        enabled=True,
        host="127.0.0.1",
        port=19876,
        mdns_enabled=False,
    )
    server, _queue = _make_server(cfg=cfg)

    # Patch TCPSite to avoid actual port binding
    with patch("aiohttp.web.TCPSite") as mock_site_cls:
        mock_site = AsyncMock()
        mock_site_cls.return_value = mock_site

        await server.start()
        assert server.is_running is True

        await server.stop()
        assert server.is_running is False


async def test_server_start_with_mdns() -> None:
    """Start with mDNS enabled calls register."""
    cfg = TelemetryConfig(
        enabled=True,
        host="127.0.0.1",
        port=19877,
        mdns_enabled=True,
    )
    server, _queue = _make_server(cfg=cfg)

    with (
        patch("aiohttp.web.TCPSite") as mock_site_cls,
        patch.object(server, "_register_mdns", new_callable=AsyncMock) as mock_mdns,
    ):
        mock_site_cls.return_value = AsyncMock()
        await server.start()
        mock_mdns.assert_awaited_once()

    with patch.object(server, "_unregister_mdns", new_callable=AsyncMock) as mock_un:
        await server.stop()
        mock_un.assert_awaited_once()


async def test_server_stop_closes_ws_clients() -> None:
    """Stop closes all connected WebSocket clients."""
    server, _queue = _make_server()
    server._running = True

    mock_ws = AsyncMock()
    server._ws_clients.append(mock_ws)

    await server.stop()
    mock_ws.close.assert_awaited_once()
    assert len(server._ws_clients) == 0


async def test_server_metrics_path_from_config() -> None:
    """Server uses metrics_path from config when not overridden."""
    cfg = TelemetryConfig(enabled=True, metrics_path="/custom/metrics")
    registry = MetricsRegistry(MetricsConfig(enabled=True))
    server, _queue = _make_server(cfg=cfg, metrics_registry=registry)
    assert server._metrics_path == "/custom/metrics"


async def test_server_metrics_path_override() -> None:
    """Explicit metrics_path overrides config."""
    cfg = TelemetryConfig(enabled=True, metrics_path="/custom/metrics")
    queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=64)
    health = _make_health_monitor()
    registry = MetricsRegistry(MetricsConfig(enabled=True))
    server = TelemetryServer(
        cfg=cfg,
        telemetry_queue=queue,
        health_monitor=health,
        metrics_registry=registry,
        metrics_path="/override/metrics",
    )
    assert server._metrics_path == "/override/metrics"


async def test_server_sets_publish_hz_on_registry() -> None:
    """Constructor sets publish_hz on metrics registry."""
    cfg = TelemetryConfig(enabled=True, publish_hz=20.0)
    registry = MetricsRegistry(MetricsConfig(enabled=True))
    _server, _queue = _make_server(cfg=cfg, metrics_registry=registry)
    text = registry.render_prometheus()
    assert "20" in text
