"""Tests for the camera routes: /camera, /camera/frame.jpg, /camera/stream.

Uses aiohttp.test_utils for in-process testing (no port binding).
"""

from __future__ import annotations

import asyncio

import pytest

from mousedroid.config.schema import TelemetryConfig

aiohttp = pytest.importorskip("aiohttp")
from unittest.mock import AsyncMock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from mousedroid.telemetry.server import TelemetryServer


def _make_health_monitor() -> AsyncMock:
    monitor = AsyncMock()
    monitor.check_health = AsyncMock(return_value={"status": "ok"})
    return monitor


def _make_raw_frame_source(jpeg: bytes | None = b"\xff\xd8\xff\xd9") -> AsyncMock:
    source = AsyncMock()
    source.capture_raw_jpeg = AsyncMock(return_value=jpeg)
    return source


def _make_server(
    *,
    raw_frame_source: AsyncMock | None = None,
) -> tuple[TelemetryServer, asyncio.Queue]:
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    server = TelemetryServer(
        cfg=TelemetryConfig(enabled=True),
        telemetry_queue=queue,
        health_monitor=_make_health_monitor(),
        raw_frame_source=raw_frame_source,
    )
    return server, queue


def _build_app(server: TelemetryServer) -> web.Application:
    app = web.Application(middlewares=server._build_middlewares())
    server._register_routes(app)
    return app


async def test_camera_frame_returns_jpeg() -> None:
    """GET /camera/frame.jpg returns 200 with image/jpeg content type."""
    source = _make_raw_frame_source(b"\xff\xd8\xff\xd9")
    server, _ = _make_server(raw_frame_source=source)
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/camera/frame.jpg")
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "image/jpeg"
        body = await resp.read()
        assert body == b"\xff\xd8\xff\xd9"


async def test_camera_frame_returns_503_when_capture_returns_none() -> None:
    """GET /camera/frame.jpg returns 503 when capture_raw_jpeg returns None."""
    source = _make_raw_frame_source(None)
    server, _ = _make_server(raw_frame_source=source)
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/camera/frame.jpg")
        assert resp.status == 503


async def test_camera_frame_returns_404_when_no_source() -> None:
    """GET /camera/frame.jpg returns 404 when raw_frame_source is not configured."""
    server, _ = _make_server(raw_frame_source=None)
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/camera/frame.jpg")
        assert resp.status == 404


async def test_camera_stream_returns_404_when_no_source() -> None:
    """GET /camera/stream returns 404 when raw_frame_source is not configured."""
    server, _ = _make_server(raw_frame_source=None)
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/camera/stream")
        assert resp.status == 404


async def test_camera_stream_returns_mjpeg_headers() -> None:
    """GET /camera/stream returns multipart/x-mixed-replace content type."""
    jpeg = b"\xff\xd8\xff\xd9"

    call_count = 0

    async def _capture() -> bytes | None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return jpeg
        # Stop the stream after first frame by raising CancelledError
        raise asyncio.CancelledError

    source = AsyncMock()
    source.capture_raw_jpeg = _capture

    server, _ = _make_server(raw_frame_source=source)
    server._running = True  # ensure stream loop can start
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client, client.session.get(
        client.make_url("/camera/stream"), timeout=aiohttp.ClientTimeout(total=1.0)
    ) as resp:
        assert resp.status == 200
        ct = resp.headers["Content-Type"]
        assert "multipart/x-mixed-replace" in ct
        assert "mousedroidframe" in ct
