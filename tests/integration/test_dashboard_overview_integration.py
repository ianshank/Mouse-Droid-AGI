"""Integration: the fused summary survives the publish path + route wiring.

Confirms a frame built from a real observation bundle carries a populated
``fused`` summary through the ``TelemetryPublisher`` queue (what the ``/ws``
broadcast serialises via ``to_dict()``), and that the server registers the new
``/`` and ``/dashboard`` routes alongside the existing ``/camera`` + ``/lidar``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pytest

from mousedroid.config.schema import TelemetryConfig
from mousedroid.safety.context import SafetyContext
from mousedroid.sensing.bundle import MouseDroidObservationBundle
from mousedroid.telemetry.frame_builder import build_telemetry_frame
from mousedroid.telemetry.publisher import TelemetryPublisher

aiohttp = pytest.importorskip("aiohttp")

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from mousedroid.telemetry.protocol import TelemetryFrame
from mousedroid.telemetry.server import TelemetryServer

pytestmark = pytest.mark.asyncio


def _app() -> web.Application:
    queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=8)
    health = AsyncMock()
    server = TelemetryServer(cfg=TelemetryConfig(), telemetry_queue=queue, health_monitor=health)
    app = web.Application(middlewares=server._build_middlewares())
    server._register_routes(app)
    return app


def _bundle() -> MouseDroidObservationBundle:
    return MouseDroidObservationBundle(
        _timestamp=0.0,
        _vision_features=np.ones(8, dtype=np.float32),
        _distance_m=1.0,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _audio_chunk=np.zeros(16, dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32),
        _lidar_features=np.ones(36, dtype=np.float32),
        _lidar_n_points=200,
    )


async def test_fused_survives_publish_path() -> None:
    """A built frame's ``fused`` summary round-trips through the publisher."""
    cfg = TelemetryConfig(enabled=True, publish_hz=60.0, queue_size=32)
    publisher = TelemetryPublisher(cfg)
    queue = publisher.get_queue()

    frame = build_telemetry_frame(
        _bundle(), SafetyContext(is_emergency=False), loop_time_ms=5.0, tick_count=7
    )
    publisher._last_publish = 0.0
    await publisher.publish(frame)

    received = await asyncio.wait_for(queue.get(), timeout=0.5)
    payload = received.to_dict()  # what the /ws broadcast serialises
    assert "fused" in payload
    assert payload["fused"]["lidar_present"] is True
    assert payload["fused"]["n_modalities"] == 5
    assert payload["fused"]["modalities"]["lidar"] is True


async def test_server_registers_dashboard_routes() -> None:
    queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=8)
    health = AsyncMock()
    server = TelemetryServer(cfg=TelemetryConfig(), telemetry_queue=queue, health_monitor=health)
    app = web.Application(middlewares=server._build_middlewares())
    server._register_routes(app)

    paths = {route.resource.canonical for route in app.router.routes()}
    assert "/" in paths
    assert "/dashboard" in paths
    assert "/camera" in paths  # existing pages still registered
    assert "/lidar" in paths


async def test_root_handler_redirects_to_dashboard() -> None:
    """Exercise _handle_root (covers the redirect handler in the integration tier)."""
    async with TestClient(TestServer(_app())) as client:
        resp = await client.get("/?token=t", allow_redirects=False)
        assert resp.status == 302
        assert resp.headers["Location"] == "/dashboard?token=t"


async def test_dashboard_handler_serves_page() -> None:
    """Exercise _handle_dashboard_page (covers the page handler in the integration tier)."""
    async with TestClient(TestServer(_app())) as client:
        resp = await client.get("/dashboard")
        assert resp.status == 200
        body = await resp.text()
    assert "MouseDroid — Dashboard" in body
