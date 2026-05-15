"""Unit tests for ``_handle_lidar_raw_ws`` via aiohttp TestClient.

Mirrors the pattern in ``tests/unit/test_telemetry_server.py`` so the
branch-coverage gate (which collects coverage from ``tests/unit/telemetry``)
sees the handler exercised over real aiohttp WebSocket plumbing without
needing a port bind.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from mousedroid.config.schema import (
    HealthConfig,
    JetsonConfig,
    MetricsConfig,
    TelemetryConfig,
)
from mousedroid.health.monitor import HealthMonitor
from mousedroid.telemetry.metrics import MetricsRegistry
from mousedroid.telemetry.protocol import LidarRawScan
from mousedroid.telemetry.publisher import TelemetryPublisher
from mousedroid.telemetry.server import TelemetryServer


def _build_server(*, with_queue: bool = True, max_clients: int = 4) -> TelemetryServer:
    cfg = TelemetryConfig(
        enabled=True,
        mdns_enabled=False,
        publish_hz=30.0,
        lidar_raw_publish_hz=30.0,
        max_clients=max_clients,
        ws_handshake_timeout_s=0.2,
    )
    publisher = TelemetryPublisher(cfg)
    return TelemetryServer(
        cfg=cfg,
        telemetry_queue=publisher.get_queue(),
        health_monitor=HealthMonitor(HealthConfig(), JetsonConfig()),
        metrics_registry=MetricsRegistry(MetricsConfig()),
        publisher=publisher,
        lidar_raw_queue=publisher.get_lidar_raw_queue() if with_queue else None,
    )


def _build_app(server: TelemetryServer) -> web.Application:
    app = web.Application(middlewares=server._build_middlewares())
    server._register_routes(app)
    return app


@pytest.mark.asyncio
async def test_handler_closes_when_no_queue_wired() -> None:
    """``_handle_lidar_raw_ws`` returns close-code 4404 without a queue."""
    server = _build_server(with_queue=False)
    server._running = True

    async with (
        TestClient(TestServer(_build_app(server))) as client,
        client.ws_connect(server._cfg.lidar_raw_ws_path) as ws,
    ):
        msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
        assert msg.type == aiohttp.WSMsgType.CLOSE


@pytest.mark.asyncio
async def test_handler_rejects_when_max_clients_reached() -> None:
    """A second connection above ``max_clients`` is closed with 4029."""
    server = _build_server(max_clients=1)
    server._running = True
    path = server._cfg.lidar_raw_ws_path

    async with (
        TestClient(TestServer(_build_app(server))) as client,
        client.ws_connect(path),  # first occupies the slot
        client.ws_connect(path) as ws2,
    ):
        msg = await asyncio.wait_for(ws2.receive(), timeout=1.0)
        assert msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED)


@pytest.mark.asyncio
async def test_max_clients_check_is_race_safe() -> None:
    """Concurrent connects past ``max_clients`` cannot all slip past the guard.

    Addresses Gemini medium review (PR #79 comment_id=3238374802): the
    previous code awaited ``ws.prepare(request)`` between the count
    check and the append, which yielded the event loop and let two
    racing connects both pass the limit. With the reservation moved
    into a synchronous critical section, exactly ``max_clients``
    connections succeed and the rest are closed with 4029.
    """
    server = _build_server(max_clients=2)
    server._running = True
    path = server._cfg.lidar_raw_ws_path

    async with TestClient(TestServer(_build_app(server))) as client:
        # Launch four concurrent connects; only two should make it
        # into ``_lidar_ws_clients``. Without the fix, the race would
        # occasionally let three or four through.
        async def _connect() -> aiohttp.ClientWebSocketResponse:
            return await client.ws_connect(path)

        sockets = await asyncio.gather(
            _connect(), _connect(), _connect(), _connect(), return_exceptions=True
        )
        try:
            # Give the server time to settle accept/reject of each.
            await asyncio.sleep(0.1)
            assert len(server._lidar_ws_clients) <= server._cfg.max_clients
        finally:
            for ws in sockets:
                if isinstance(ws, aiohttp.ClientWebSocketResponse):
                    await ws.close()


@pytest.mark.asyncio
async def test_handler_accepts_negotiated_client_and_receives_scan() -> None:
    """Full lifecycle: connect, negotiate, deliver a scan, disconnect."""
    server = _build_server()
    publisher = server._publisher
    assert publisher is not None
    server._running = True
    # Drive the broadcast loop alongside the handler.
    bg = asyncio.create_task(server._lidar_raw_broadcast_loop())

    try:
        async with (
            TestClient(TestServer(_build_app(server))) as client,
            client.ws_connect(server._cfg.lidar_raw_ws_path) as ws,
        ):
            await ws.send_json(
                {
                    "hello": {
                        "protocol_version": 1,
                        "supported_serializations": ["json"],
                    }
                }
            )
            ack_msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
            ack = json.loads(ack_msg.data)
            assert ack["hello_ack"]["ok"] is True

            publisher._lidar_raw_last_publish = 0.0  # type: ignore[attr-defined]
            await publisher.publish_lidar_raw(
                LidarRawScan(
                    timestamp=0.0,
                    angles_rad=[0.0, 1.0],
                    distances_m=[1.0, 2.0],
                    n_points=2,
                    scan_duration_s=0.1,
                )
            )

            scan: dict[str, Any] | None = None
            deadline = asyncio.get_event_loop().time() + 1.5
            while asyncio.get_event_loop().time() < deadline and scan is None:
                msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                if msg.type != aiohttp.WSMsgType.TEXT:
                    break
                payload = json.loads(msg.data)
                if "angles_rad" in payload:
                    scan = payload
        assert scan is not None
        assert scan["n_points"] == 2
    finally:
        server._running = False
        bg.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await bg


@pytest.mark.asyncio
async def test_main_ws_hello_negotiation_via_aiohttp() -> None:
    """The main ``/ws`` endpoint honours the optional hello message."""
    server = _build_server()
    server._running = True

    async with (
        TestClient(TestServer(_build_app(server))) as client,
        client.ws_connect(server._cfg.ws_path) as ws,
    ):
        await ws.send_json(
            {
                "hello": {
                    "protocol_version": 1,
                    "supported_serializations": ["json"],
                    "preferred_serialization": "json",
                }
            }
        )
        ack_msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
        ack = json.loads(ack_msg.data)
        assert ack["hello_ack"]["negotiated"] is True
        assert ack["hello_ack"]["serialization"] == "json"
