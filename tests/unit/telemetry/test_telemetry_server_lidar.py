"""Tests for the ``/lidar`` HTML page and LiDAR Prometheus metrics path.

Uses aiohttp.test_utils for in-process testing (no port binding).
"""

from __future__ import annotations

import asyncio

import pytest

from mousedroid.config.schema import MetricsConfig, TelemetryConfig
from mousedroid.telemetry.metrics import MetricsRegistry
from mousedroid.telemetry.protocol import TelemetryFrame

aiohttp = pytest.importorskip("aiohttp")
from unittest.mock import AsyncMock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from mousedroid.telemetry.server import TelemetryServer


def _make_health_monitor() -> AsyncMock:
    monitor = AsyncMock()
    monitor.check_health = AsyncMock(return_value={"status": "ok"})
    return monitor


def _make_server(
    *,
    metrics_registry: MetricsRegistry | None = None,
    lidar_max_range_m: float | None = 12.0,
) -> tuple[TelemetryServer, asyncio.Queue[TelemetryFrame]]:
    queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=64)
    server = TelemetryServer(
        cfg=TelemetryConfig(enabled=True),
        telemetry_queue=queue,
        health_monitor=_make_health_monitor(),
        metrics_registry=metrics_registry,
        lidar_max_range_m=lidar_max_range_m,
    )
    return server, queue


def _build_app(server: TelemetryServer) -> web.Application:
    app = web.Application(middlewares=server._build_middlewares())
    server._register_routes(app)
    return app


async def test_lidar_page_served() -> None:
    """GET /lidar returns the HTML visualisation with a WebSocket client."""
    server, _ = _make_server()
    app = _build_app(server)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/lidar")
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/html")
        body = await resp.text()
        assert "<canvas" in body
        assert "new WebSocket(" in body
        assert "lidar_sectors" in body


async def test_metrics_endpoint_includes_lidar_families() -> None:
    """Broadcast-loop metric writers publish LiDAR gauges via /metrics."""
    registry = MetricsRegistry(MetricsConfig())
    server, _ = _make_server(metrics_registry=registry, lidar_max_range_m=10.0)
    app = _build_app(server)

    # Simulate what _broadcast_loop does for a single frame.
    registry.set_lidar_sectors([0.9, 0.2, 0.5, 1.0], max_range_m=10.0)
    registry.set_lidar_min_distance_m(2.0)
    registry.set_lidar_scan_points(400)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/metrics")
        assert resp.status == 200
        text = await resp.text()

    assert 'mousedroid_lidar_sector_distance_m{sector="0"} 9' in text
    assert 'mousedroid_lidar_sector_distance_m{sector="1"} 2' in text
    assert "mousedroid_lidar_min_distance_m 2" in text
    assert "mousedroid_lidar_scan_points 400" in text


def test_metrics_registry_respects_track_lidar_toggle() -> None:
    """``MetricsConfig.track_lidar=False`` suppresses all LiDAR families."""
    registry = MetricsRegistry(MetricsConfig(track_lidar=False))
    registry.set_lidar_sectors([0.5, 0.5], max_range_m=5.0)
    registry.set_lidar_min_distance_m(1.0)
    registry.set_lidar_scan_points(10)

    text = registry.render_prometheus()
    assert "mousedroid_lidar_sector_distance_m" not in text
    assert "mousedroid_lidar_min_distance_m" not in text
    assert "mousedroid_lidar_scan_points" not in text
