"""Smoke tests for the telemetry subsystem.

Exercises the full telemetry stack end-to-end:
- TelemetryFrame construction and serialisation
- LogRingBuffer capture, ring-over, and live-streaming
- TelemetryPublisher rate-limiting and drop behaviour
- MockTelemetryServer protocol compliance and lifecycle
- FrameBuilder observation→frame conversion
- TelemetryServer full lifecycle with real port binding
- Publisher → broadcast loop → WebSocket frame delivery chain
- REST endpoints (status, sensors, health, logs, network)
- CORS headers and API-key authentication
- Multi-client WebSocket fan-out

These tests require aiohttp; the entire module is skipped if it is absent.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mousedroid.config.schema import TelemetryConfig
from mousedroid.telemetry.log_buffer import LogRingBuffer
from mousedroid.telemetry.mock_server import MockTelemetryServer
from mousedroid.telemetry.network import NetworkInterface
from mousedroid.telemetry.protocol import TelemetryFrame, TelemetryServerProtocol
from mousedroid.telemetry.publisher import TelemetryPublisher

# Stub network responses to avoid real DNS/socket I/O on Windows
_STUB_INTERFACES = [
    NetworkInterface(name="eth0", ip="192.168.1.100", interface_type="ethernet", up=True)
]
_STUB_DEFAULT_IP = "192.168.1.100"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.smoke


def _free_port() -> int:
    """Return an OS-assigned free TCP port on loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _cfg(**kwargs: object) -> TelemetryConfig:
    defaults: dict[str, object] = {
        "enabled": True,
        "queue_size": 8,
        "publish_hz": 30.0,
        "mdns_enabled": False,
    }
    defaults.update(kwargs)
    return TelemetryConfig(**defaults)  # type: ignore[arg-type]


def _frame(**kwargs: object) -> TelemetryFrame:
    defaults: dict[str, object] = {
        "timestamp": 1.0,
        "distance_m": 0.5,
        "tick_count": 1,
        "battery_voltage": 11.8,
        "motor_state": [0.1, 0.0, 0.0, 11.8],
    }
    defaults.update(kwargs)
    return TelemetryFrame(**defaults)  # type: ignore[arg-type]


def _health_monitor() -> AsyncMock:
    monitor = AsyncMock()
    monitor.check_health = AsyncMock(
        return_value={"status": "ok", "gpu_temp_c": 45.0, "gpu_load_pct": 30.0}
    )
    return monitor


# ---------------------------------------------------------------------------
# TelemetryFrame
# ---------------------------------------------------------------------------


def test_frame_to_dict_contains_all_keys() -> None:
    frame = _frame(timestamp=99.9, tick_count=42)
    d = frame.to_dict()
    assert d["timestamp"] == 99.9
    assert d["tick_count"] == 42
    for key in ("distance_m", "motor_state", "battery_voltage", "vision_norm", "audio_rms"):
        assert key in d, f"Missing key: {key}"


def test_frame_round_trips_through_json() -> None:
    frame = _frame(
        safety={"is_emergency": False, "violations": []},
        health={"gpu_temp_c": 50.0},
    )
    recovered = json.loads(json.dumps(frame.to_dict()))
    assert recovered["timestamp"] == frame.timestamp
    assert recovered["safety"]["is_emergency"] is False


def test_frame_is_frozen_dataclass() -> None:
    frame = _frame()
    with pytest.raises(AttributeError, match=r"cannot assign to field|FrozenInstance"):
        frame.timestamp = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LogRingBuffer
# ---------------------------------------------------------------------------


def test_log_buffer_captures_entries() -> None:
    buf = LogRingBuffer(maxlen=10)
    buf(None, "info", {"event": "alpha"})
    buf(None, "info", {"event": "beta"})
    recent = buf.get_recent(10)
    assert len(recent) == 2
    assert recent[0]["event"] == "alpha"
    assert recent[1]["event"] == "beta"


def test_log_buffer_respects_maxlen() -> None:
    buf = LogRingBuffer(maxlen=3)
    for i in range(5):
        buf(None, "info", {"event": f"e{i}"})
    assert buf.size == 3
    assert buf.get_recent(10)[-1]["event"] == "e4"


def test_log_buffer_get_recent_n_truncates() -> None:
    buf = LogRingBuffer(maxlen=20)
    for i in range(10):
        buf(None, "info", {"event": f"e{i}"})
    assert len(buf.get_recent(3)) == 3


def test_log_buffer_is_transparent_processor() -> None:
    buf = LogRingBuffer()
    event: dict[str, object] = {"event": "test", "level": "warning", "custom": 42}
    result = buf(None, "warning", event)
    assert result is event


async def test_log_buffer_subscribe_receives_live_entries() -> None:
    buf = LogRingBuffer(maxlen=100)
    q = buf.subscribe()
    buf(None, "info", {"event": "live_event"})
    item = await asyncio.wait_for(q.get(), timeout=1.0)
    assert item["event"] == "live_event"
    buf.unsubscribe(q)


async def test_log_buffer_double_unsubscribe_is_safe() -> None:
    buf = LogRingBuffer()
    q = buf.subscribe()
    buf.unsubscribe(q)
    buf.unsubscribe(q)  # should not raise


# ---------------------------------------------------------------------------
# TelemetryPublisher
# ---------------------------------------------------------------------------


async def test_publisher_initial_stats_are_zero() -> None:
    """Initial stats are all zero (forward-compatible with new counter additions).

    Inclusive check rather than ``== {expected_dict}`` so adding a new
    counter to ``TelemetryPublisher.stats`` (e.g. ``lidar_raw_published`` /
    ``lidar_raw_dropped`` added by Tier C1) doesn't silently break this
    smoke gate. We pin the two baseline counter names that have always
    been part of the contract, then assert every counter on a fresh
    publisher starts at 0 so new counters auto-inherit the invariant.
    """
    pub = TelemetryPublisher(_cfg())
    stats = pub.stats
    # Baseline counters that have always been part of the contract.
    assert stats["frames_published"] == 0
    assert stats["frames_dropped"] == 0
    # Generic invariant: any counter on a fresh publisher must start at 0.
    assert all(v == 0 for v in stats.values()), f"non-zero counter on fresh publisher: {stats}"


async def test_publisher_enqueues_frame_and_updates_stats() -> None:
    pub = TelemetryPublisher(_cfg(publish_hz=60.0))
    frame = _frame(tick_count=1)
    await pub.publish(frame)
    assert pub.stats["frames_published"] == 1
    q = pub.get_queue()
    assert q.qsize() == 1
    assert (await q.get()) is frame


async def test_publisher_rate_limiting_drops_rapid_calls() -> None:
    pub = TelemetryPublisher(_cfg(publish_hz=1.0))
    await pub.publish(_frame(tick_count=1))  # first passes
    await pub.publish(_frame(tick_count=2))  # too fast - skipped
    await pub.publish(_frame(tick_count=3))  # too fast - skipped
    assert pub.stats["frames_published"] == 1


async def test_publisher_drops_when_queue_full() -> None:
    pub = TelemetryPublisher(_cfg(queue_size=2, publish_hz=60.0))
    await pub.publish(_frame(tick_count=1))
    pub._last_publish = 0.0
    await pub.publish(_frame(tick_count=2))
    pub._last_publish = 0.0  # bypass rate limit
    await pub.publish(_frame(tick_count=3))  # queue full → drop
    assert pub.stats["frames_dropped"] == 1
    assert pub.stats["frames_published"] == 2


async def test_publisher_get_queue_returns_asyncio_queue() -> None:
    pub = TelemetryPublisher(_cfg())
    assert isinstance(pub.get_queue(), asyncio.Queue)


# ---------------------------------------------------------------------------
# MockTelemetryServer
# ---------------------------------------------------------------------------


async def test_mock_server_lifecycle() -> None:
    mock = MockTelemetryServer()
    assert mock.is_running is False
    await mock.start()
    assert mock.is_running is True
    await mock.stop()
    assert mock.is_running is False


async def test_mock_server_client_count_always_zero() -> None:
    mock = MockTelemetryServer()
    await mock.start()
    assert mock.client_count == 0


def test_mock_server_satisfies_protocol() -> None:
    assert isinstance(MockTelemetryServer(), TelemetryServerProtocol)


# ---------------------------------------------------------------------------
# FrameBuilder
# ---------------------------------------------------------------------------


def _make_obs(
    distance: float = 1.5,
    vision_dim: int = 4,
    audio_len: int = 8,
) -> MagicMock:
    obs = MagicMock()
    obs.timestamp = time.monotonic()
    obs.distance_m = distance
    obs.motor_state = np.array([0.1, 0.0, 0.05, 11.7])
    obs.vision_features = np.random.rand(vision_dim).astype(np.float32)
    obs.audio_chunk = np.random.rand(audio_len).astype(np.float32) * 0.1
    obs.valid_mask = np.ones(3, dtype=np.float32)
    return obs


def _make_safety(*, is_emergency: bool = False) -> MagicMock:
    ctx = MagicMock()
    ctx.is_emergency = is_emergency
    ctx.law_violations = []
    ctx.forward_clearance_ok = not is_emergency
    return ctx


def test_frame_builder_populates_fields() -> None:
    from mousedroid.telemetry.frame_builder import build_telemetry_frame

    frame = build_telemetry_frame(_make_obs(distance=2.1), _make_safety(), 5.0, 5)
    assert isinstance(frame, TelemetryFrame)
    assert frame.distance_m == pytest.approx(2.1, rel=1e-5)
    assert frame.loop_time_ms == pytest.approx(5.0)
    assert isinstance(frame.motor_state, list)
    assert frame.vision_norm >= 0.0
    assert frame.audio_rms >= 0.0


def test_frame_builder_emergency_flag_propagates() -> None:
    from mousedroid.telemetry.frame_builder import build_telemetry_frame

    frame = build_telemetry_frame(_make_obs(), _make_safety(is_emergency=True), 3.0, 1)
    assert frame.safety["is_emergency"] is True
    assert frame.safety["forward_clearance_ok"] is False


def test_frame_builder_battery_voltage_extracted() -> None:
    from mousedroid.constants import MOTOR_STATE_BATTERY_INDEX
    from mousedroid.telemetry.frame_builder import build_telemetry_frame

    obs = _make_obs()
    obs.motor_state = np.zeros(MOTOR_STATE_BATTERY_INDEX + 1, dtype=np.float32)
    obs.motor_state[MOTOR_STATE_BATTERY_INDEX] = 12.3
    frame = build_telemetry_frame(obs, _make_safety(), 1.0, 1)
    assert frame.battery_voltage == pytest.approx(12.3, rel=1e-4)


def test_frame_builder_vision_norm_is_scalar() -> None:
    from mousedroid.telemetry.frame_builder import build_telemetry_frame

    obs = _make_obs(vision_dim=512)
    frame = build_telemetry_frame(obs, _make_safety(), 1.0, 1)
    assert isinstance(frame.vision_norm, float)
    assert frame.vision_norm >= 0.0


# ---------------------------------------------------------------------------
# TelemetryServer — HTTP + WebSocket (aiohttp required)
# ---------------------------------------------------------------------------

aiohttp = pytest.importorskip("aiohttp")

from aiohttp.test_utils import TestClient, TestServer

from mousedroid.telemetry.server import TelemetryServer


def _make_server(
    cfg: TelemetryConfig | None = None,
    log_buffer: LogRingBuffer | None = None,
) -> tuple[TelemetryServer, asyncio.Queue[TelemetryFrame]]:
    if cfg is None:
        cfg = _cfg()
    q: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=64)
    server = TelemetryServer(
        cfg=cfg,
        telemetry_queue=q,
        health_monitor=_health_monitor(),
        log_buffer=log_buffer,
    )
    return server, q


def _build_app(server: TelemetryServer):  # type: ignore[return]
    from aiohttp import web

    app = web.Application(middlewares=server._build_middlewares())
    server._register_routes(app)
    return app


# --- Lifecycle ---


def test_server_initial_state() -> None:
    server, _ = _make_server()
    assert server.is_running is False
    assert server.client_count == 0


async def test_server_real_port_start_stop() -> None:
    port = _free_port()
    server, _ = _make_server(cfg=_cfg(host="127.0.0.1", port=port))
    await server.start()
    assert server.is_running is True
    await server.stop()
    assert server.is_running is False


# --- REST: /api/v1/status ---


async def test_status_returns_200_with_expected_keys() -> None:
    server, _ = _make_server()
    server._running = True
    async with TestClient(TestServer(_build_app(server))) as client:
        resp = await client.get("/api/v1/status")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "running"
        assert "uptime_s" in data
        assert "ws_clients" in data
        assert "tick_count" in data


# --- REST: /api/v1/sensors ---


async def test_sensors_returns_503_when_no_frame() -> None:
    server, _ = _make_server()
    async with TestClient(TestServer(_build_app(server))) as client:
        resp = await client.get("/api/v1/sensors")
        assert resp.status == 503


async def test_sensors_returns_latest_frame() -> None:
    server, _ = _make_server()
    server._latest_frame = _frame(timestamp=77.7, tick_count=99)
    async with TestClient(TestServer(_build_app(server))) as client:
        resp = await client.get("/api/v1/sensors")
        assert resp.status == 200
        data = await resp.json()
        assert data["timestamp"] == pytest.approx(77.7)
        assert data["tick_count"] == 99


# --- REST: /api/v1/health ---


async def test_health_merges_health_monitor_data() -> None:
    server, _ = _make_server()
    async with TestClient(TestServer(_build_app(server))) as client:
        resp = await client.get("/api/v1/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"
        assert data["gpu_temp_c"] == pytest.approx(45.0)


async def test_health_includes_battery_from_latest_frame() -> None:
    server, _ = _make_server()
    server._latest_frame = _frame(battery_voltage=12.0)
    async with TestClient(TestServer(_build_app(server))) as client:
        resp = await client.get("/api/v1/health")
        data = await resp.json()
        assert data["battery_voltage"] == pytest.approx(12.0)


# --- REST: /api/v1/logs ---


async def test_logs_returns_503_without_log_buffer() -> None:
    server, _ = _make_server(log_buffer=None)
    async with TestClient(TestServer(_build_app(server))) as client:
        resp = await client.get("/api/v1/logs")
        assert resp.status == 503


async def test_logs_returns_entries_with_buffer() -> None:
    buf = LogRingBuffer()
    buf(None, "info", {"event": "smoke_event", "tick": 1})
    server, _ = _make_server(log_buffer=buf)
    async with TestClient(TestServer(_build_app(server))) as client:
        resp = await client.get("/api/v1/logs?n=5")
        assert resp.status == 200
        data = await resp.json()
        assert data["count"] >= 1
        assert len(data["entries"]) >= 1


# --- REST: /api/v1/network ---


async def test_network_endpoint_returns_interfaces_and_url() -> None:
    server, _ = _make_server()
    with (
        patch(
            "mousedroid.telemetry.server.get_network_interfaces",
            new=AsyncMock(return_value=_STUB_INTERFACES),
        ),
        patch("mousedroid.telemetry.server.get_default_ip", return_value=_STUB_DEFAULT_IP),
    ):
        async with TestClient(TestServer(_build_app(server))) as client:
            resp = await client.get("/api/v1/network")
            assert resp.status == 200
            data = await resp.json()
            assert "interfaces" in data
            assert isinstance(data["interfaces"], list)
            assert "server_url" in data
            assert "server_port" in data


# --- CORS ---


async def test_cors_headers_present_on_rest_responses() -> None:
    server, _ = _make_server()
    server._running = True
    with (
        patch(
            "mousedroid.telemetry.server.get_network_interfaces",
            new=AsyncMock(return_value=_STUB_INTERFACES),
        ),
        patch("mousedroid.telemetry.server.get_default_ip", return_value=_STUB_DEFAULT_IP),
    ):
        async with TestClient(TestServer(_build_app(server))) as client:
            for path in ("/api/v1/status", "/api/v1/health", "/api/v1/network"):
                resp = await client.get(path)
                assert "Access-Control-Allow-Origin" in resp.headers, f"Missing CORS on {path}"


async def test_cors_blocks_unlisted_origin() -> None:
    server, _ = _make_server(cfg=_cfg(cors_origins=["http://allowed.local"]))
    server._running = True
    async with TestClient(TestServer(_build_app(server))) as client:
        resp = await client.get("/api/v1/status", headers={"Origin": "http://evil.example"})
        origin = resp.headers.get("Access-Control-Allow-Origin", "")
        assert "evil.example" not in origin


# --- API key auth ---


async def test_auth_required_when_api_key_configured() -> None:
    server, _ = _make_server(cfg=_cfg(api_key="tok-smoke"))
    server._running = True
    async with TestClient(TestServer(_build_app(server))) as client:
        resp = await client.get("/api/v1/status")
        assert resp.status == 401


async def test_auth_accepted_with_correct_api_key() -> None:
    server, _ = _make_server(cfg=_cfg(api_key="tok-smoke"))
    server._running = True
    async with TestClient(TestServer(_build_app(server))) as client:
        resp = await client.get("/api/v1/status", headers={"X-API-Key": "tok-smoke"})
        assert resp.status == 200


async def test_no_auth_required_by_default() -> None:
    server, _ = _make_server()
    server._running = True
    async with TestClient(TestServer(_build_app(server))) as client:
        resp = await client.get("/api/v1/status")
        assert resp.status == 200


async def test_wrong_api_key_returns_401() -> None:
    server, _ = _make_server(cfg=_cfg(api_key="correct"))
    server._running = True
    async with TestClient(TestServer(_build_app(server))) as client:
        resp = await client.get("/api/v1/status", headers={"X-API-Key": "wrong"})
        assert resp.status == 401


# --- WebSocket ---


async def test_ws_connect_increments_client_count() -> None:
    server, _ = _make_server()
    server._running = True
    async with TestClient(TestServer(_build_app(server))) as client:
        async with client.ws_connect("/ws"):
            assert server.client_count == 1
        await asyncio.sleep(0.05)
        assert server.client_count == 0


async def test_ws_max_clients_rejects_excess_connection() -> None:
    server, _ = _make_server(cfg=_cfg(max_clients=1))
    server._running = True
    async with (
        TestClient(TestServer(_build_app(server))) as client,
        client.ws_connect("/ws"),
        client.ws_connect("/ws") as ws2,
    ):
        msg = await ws2.receive()
        assert msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED)


# --- End-to-end: publisher → broadcast → WebSocket ---


async def test_publisher_to_ws_frame_delivery() -> None:
    """Full chain: publish frame → broadcast loop fans out → WS client receives it."""
    server, queue = _make_server()
    server._running = True
    broadcast = asyncio.create_task(server._broadcast_loop())
    try:
        async with (
            TestClient(TestServer(_build_app(server))) as client,
            client.ws_connect("/ws") as ws,
        ):
            frame = _frame(timestamp=55.5, tick_count=77)
            await queue.put(frame)
            msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
            assert msg.type == aiohttp.WSMsgType.TEXT
            payload = json.loads(msg.data)
            assert payload["timestamp"] == pytest.approx(55.5)
            assert payload["tick_count"] == 77
    finally:
        server._running = False
        broadcast.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await broadcast


async def test_multiple_ws_clients_all_receive_same_frame() -> None:
    """Two connected clients both receive the broadcast frame."""
    server, queue = _make_server(cfg=_cfg(max_clients=5))
    server._running = True
    broadcast = asyncio.create_task(server._broadcast_loop())
    try:
        async with (
            TestClient(TestServer(_build_app(server))) as client,
            client.ws_connect("/ws") as ws1,
            client.ws_connect("/ws") as ws2,
        ):
            frame = _frame(timestamp=10.0, tick_count=5)
            await queue.put(frame)
            msg1 = await asyncio.wait_for(ws1.receive(), timeout=2.0)
            msg2 = await asyncio.wait_for(ws2.receive(), timeout=2.0)
            assert json.loads(msg1.data)["tick_count"] == 5
            assert json.loads(msg2.data)["tick_count"] == 5
    finally:
        server._running = False
        broadcast.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await broadcast


async def test_ws_receives_updated_sensor_data_on_reconnect() -> None:
    """After disconnect + reconnect, client receives fresh frames, not stale ones."""
    server, queue = _make_server()
    server._running = True
    broadcast = asyncio.create_task(server._broadcast_loop())
    try:
        async with TestClient(TestServer(_build_app(server))) as client:
            # First connection
            async with client.ws_connect("/ws") as ws:
                await queue.put(_frame(tick_count=1))
                msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                assert json.loads(msg.data)["tick_count"] == 1

            # Second connection — should get new frame
            async with client.ws_connect("/ws") as ws:
                await queue.put(_frame(tick_count=2))
                msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                assert json.loads(msg.data)["tick_count"] == 2
    finally:
        server._running = False
        broadcast.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await broadcast


# --- Integration: TelemetryPublisher feeds TelemetryServer REST ---


async def test_publisher_feeds_sensors_endpoint() -> None:
    """Frame published to TelemetryPublisher is visible on /api/v1/sensors."""
    pub = TelemetryPublisher(_cfg(publish_hz=60.0))
    server = TelemetryServer(
        cfg=_cfg(),
        telemetry_queue=pub.get_queue(),
        health_monitor=_health_monitor(),
    )
    server._running = True
    broadcast = asyncio.create_task(server._broadcast_loop())
    try:
        # Push a frame via the publisher
        await pub.publish(_frame(timestamp=42.0, tick_count=99))
        # Give broadcast loop a cycle to pull from queue and set _latest_frame
        await asyncio.sleep(0.05)
        async with TestClient(TestServer(_build_app(server))) as client:
            resp = await client.get("/api/v1/sensors")
            assert resp.status == 200
            data = await resp.json()
            assert data["timestamp"] == pytest.approx(42.0)
            assert data["tick_count"] == 99
    finally:
        server._running = False
        broadcast.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await broadcast
