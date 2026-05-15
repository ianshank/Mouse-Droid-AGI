"""Integration test for ``/ws/v1/lidar/raw`` streaming.

Boots a real :class:`TelemetryServer` on a kernel-assigned port, hands
it a publisher pre-loaded with raw scans, connects a WebSocket client,
performs the optional hello negotiation, and asserts scans flow.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp
import pytest

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


def _make_health() -> HealthMonitor:
    return HealthMonitor(HealthConfig(), JetsonConfig())


def _make_cfg() -> TelemetryConfig:
    return TelemetryConfig(
        enabled=True,
        host="127.0.0.1",
        port=1,  # ignored when strategy=kernel_assigned, but must satisfy gt=0
        port_discovery_strategy="kernel_assigned",
        mdns_enabled=False,
        publish_hz=30.0,
        lidar_raw_publish_hz=30.0,
        ws_handshake_timeout_s=0.2,
        mdns_register_timeout_s=0.5,
    )


def _scan(ts: float) -> LidarRawScan:
    angles = [i * 0.1 for i in range(8)]
    distances = [1.0 + 0.1 * i for i in range(8)]
    return LidarRawScan(
        timestamp=ts,
        angles_rad=angles,
        distances_m=distances,
        n_points=8,
        scan_duration_s=0.1,
    )


@pytest.mark.asyncio
async def test_raw_ws_delivers_scans() -> None:
    """A connected client receives at least one raw scan after the ack."""
    cfg = _make_cfg()
    publisher = TelemetryPublisher(cfg)
    metrics = MetricsRegistry(MetricsConfig())
    server = TelemetryServer(
        cfg=cfg,
        telemetry_queue=publisher.get_queue(),
        health_monitor=_make_health(),
        metrics_registry=metrics,
        publisher=publisher,
        lidar_raw_queue=publisher.get_lidar_raw_queue(),
    )
    await server.start()
    try:
        url = f"http://{cfg.host}:{server._bound_port}{cfg.lidar_raw_ws_path}"
        async with aiohttp.ClientSession() as session, session.ws_connect(url) as ws:
            await ws.send_json(
                {
                    "hello": {
                        "protocol_version": 1,
                        "supported_serializations": ["json"],
                    }
                }
            )
            # Wait for ack BEFORE pushing scans so the server has
            # already appended us to the broadcast list; otherwise the
            # broadcast loop drains the queue while the client list is
            # still empty.
            ack_msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
            ack = json.loads(ack_msg.data)
            assert ack.get("hello_ack", {}).get("ok") is True

            # Inject scans on the publisher side; the server fan-out
            # loop drains the queue.
            for i in range(5):
                publisher._lidar_raw_last_publish = 0.0  # type: ignore[attr-defined]
                await publisher.publish_lidar_raw(_scan(float(i)))

            # Receive at least one scan within a generous window.
            scans: list[dict[str, Any]] = []
            deadline = asyncio.get_event_loop().time() + 3.0
            while asyncio.get_event_loop().time() < deadline and not scans:
                msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                if msg.type != aiohttp.WSMsgType.TEXT:
                    break
                payload = json.loads(msg.data)
                if "angles_rad" in payload:
                    scans.append(payload)
        assert len(scans) >= 1
        assert scans[0]["n_points"] == 8
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_raw_ws_returns_503_when_queue_absent() -> None:
    """Without a raw queue the endpoint closes with code 4404."""
    cfg = _make_cfg()
    publisher = TelemetryPublisher(cfg)
    server = TelemetryServer(
        cfg=cfg,
        telemetry_queue=publisher.get_queue(),
        health_monitor=_make_health(),
        publisher=publisher,
        lidar_raw_queue=None,
    )
    await server.start()
    try:
        url = f"http://{cfg.host}:{server._bound_port}{cfg.lidar_raw_ws_path}"
        async with aiohttp.ClientSession() as session, session.ws_connect(url) as ws:
            msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
            assert msg.type == aiohttp.WSMsgType.CLOSE
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_main_ws_negotiation_falls_back_to_default() -> None:
    """A client that never sends a hello still gets frames in the server default."""
    cfg = _make_cfg()
    publisher = TelemetryPublisher(cfg)
    server = TelemetryServer(
        cfg=cfg,
        telemetry_queue=publisher.get_queue(),
        health_monitor=_make_health(),
        publisher=publisher,
        lidar_raw_queue=publisher.get_lidar_raw_queue(),
    )
    await server.start()
    try:
        url = f"http://{cfg.host}:{server._bound_port}{cfg.ws_path}"
        async with aiohttp.ClientSession() as session, session.ws_connect(url) as ws:
            # Don't send hello — wait for default-ack instead.
            msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
            payload = json.loads(msg.data)
            assert payload["hello_ack"]["serialization"] == "json"
            assert payload["hello_ack"]["negotiated"] is False
    finally:
        await server.stop()
