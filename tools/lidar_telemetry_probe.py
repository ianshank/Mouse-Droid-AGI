"""Stand-alone LiDAR -> telemetry-server probe.

Bypasses the orchestrator (which requires ESP32 + camera attached) and
exercises just the path the dashboard actually consumes:

  real LiDAR  ->  TelemetryPublisher.publish_lidar_raw()
              ->  TelemetryServer aiohttp app (port 8080)
              ->  /ws/v1/lidar/raw  WebSocket

Run on the Jetson with the real LiDAR connected. Spins up the real
telemetry server for ~30s on a non-default port (so it doesn't fight
the running orchestrator) and verifies frames arrive.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import time
from pathlib import Path

import aiohttp

from mousedroid.config.loader import load_settings
from mousedroid.factory import (
    build_health_monitor,
    build_lidar,
    build_telemetry_publisher,
    build_telemetry_server,
)
from mousedroid.logging.setup import get_logger
from mousedroid.validation.latency_stats import intervals_ms, summarize

_log = get_logger("lidar_telemetry_probe")


async def _drive_lidar_to_publisher(lidar, publisher, stop_event: asyncio.Event) -> int:
    """Drive the LiDAR at its natural rate; publish each scan."""
    frames_published = 0
    await lidar.start()
    try:
        while not stop_event.is_set():
            scan = await lidar.read_scan()
            if scan is not None and getattr(scan, "points", None):
                await publisher.publish_lidar_raw(scan)
                frames_published += 1
            else:
                await asyncio.sleep(0.05)
    finally:
        await lidar.stop()
    return frames_published


async def _consume_lidar_ws(
    url: str, max_frames: int, timeout_s: float
) -> tuple[list[dict], list[float]]:
    """Connect to the WS endpoint; collect up to ``max_frames`` JSON messages.

    Returns the parsed frames and a parallel list of monotonic arrival times
    (seconds) — the caller derives inter-arrival jitter from the latter without
    re-reading the clock here.
    """
    frames: list[dict] = []
    arrivals_s: list[float] = []
    ws_close_timeout = aiohttp.ClientWSTimeout(ws_close=10.0)
    terminal_msg_types = (
        aiohttp.WSMsgType.CLOSE,
        aiohttp.WSMsgType.CLOSED,
        aiohttp.WSMsgType.ERROR,
    )
    async with (
        aiohttp.ClientSession() as session,
        session.ws_connect(url, timeout=ws_close_timeout) as ws,
    ):
        deadline = time.monotonic() + timeout_s
        while len(frames) < max_frames and time.monotonic() < deadline:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            if msg.type == aiohttp.WSMsgType.TEXT:
                with contextlib.suppress(json.JSONDecodeError):
                    frames.append(json.loads(msg.data))
                    arrivals_s.append(time.monotonic())
            elif msg.type in terminal_msg_types:
                break
    return frames, arrivals_s


async def _main(args: argparse.Namespace) -> int:
    cfg = load_settings(Path(args.config))
    cfg.mock_hardware = False
    cfg.telemetry.host = "127.0.0.1"
    cfg.telemetry.port = args.port
    if cfg.telemetry.auth is not None:
        cfg.telemetry.auth.auth_enabled = False  # verification-only

    lidar = build_lidar(cfg)
    if lidar is None:
        _log.error("lidar_build_returned_none — check cfg.lidar.enabled")
        return 2

    publisher = build_telemetry_publisher(cfg)
    if publisher is None:
        _log.error("publisher_none")
        return 3

    health_monitor = build_health_monitor(cfg)
    server = build_telemetry_server(cfg, publisher=publisher, health_monitor=health_monitor)
    if server is None:
        _log.error("telemetry_server_returned_none — check cfg.telemetry.enabled + mock_force_real")
        return 4
    _log.info("telemetry_server_type", type_=type(server).__name__)

    # ``TelemetryPublisher`` is a pull-style publisher — it has no start()/stop().
    # The server pumps it via its background tasks once start() is called.
    await server.start()

    stop_event = asyncio.Event()
    producer = asyncio.create_task(_drive_lidar_to_publisher(lidar, publisher, stop_event))

    # Give the LiDAR ~3 seconds to start producing scans before the WS client connects.
    await asyncio.sleep(3.0)

    ws_url = f"ws://127.0.0.1:{args.port}{cfg.telemetry.lidar_raw_ws_path}"
    _log.info("connecting_ws", url=ws_url)
    frames, arrivals_s = await _consume_lidar_ws(
        ws_url, max_frames=args.max_frames, timeout_s=args.duration
    )

    stop_event.set()
    published = await producer
    await server.stop()

    sample_n_points = [f.get("n_points") or len(f.get("points", [])) for f in frames[:5]]
    _log.info(
        "probe_complete",
        frames_received_via_ws=len(frames),
        frames_published_by_publisher=published,
        first_5_n_points=sample_n_points,
    )

    # Inter-frame arrival jitter — needs >=2 arrivals to form one interval.
    # Surfaces dashboard stutter (a high p95/p99 vs p50 means dropped/bunched
    # frames) that a raw frame count never reveals.
    gaps_ms = intervals_ms(arrivals_s)
    if gaps_ms:
        summary = summarize(gaps_ms)
        _log.info(
            "lidar_frame_interval_summary",
            intervals=len(gaps_ms),
            min_ms=summary.min_ms,
            mean_ms=summary.mean_ms,
            p50_ms=summary.p50_ms,
            p95_ms=summary.p95_ms,
            p99_ms=summary.p99_ms,
            max_ms=summary.max_ms,
        )

    if not frames:
        return 5
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/etc/mousedroid/jetson_production.yaml")
    parser.add_argument(
        "--port",
        type=int,
        default=8090,
        help="non-default to avoid orchestrator collision",
    )
    parser.add_argument("--duration", type=float, default=15.0, help="seconds to listen on the WS")
    parser.add_argument("--max-frames", type=int, default=20)
    args = parser.parse_args()
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
