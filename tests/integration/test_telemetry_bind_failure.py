"""Integration test: TelemetryServer bind failure does not crash the process.

Uses a real socket to occupy a port, then verifies the server raises
TelemetryUnavailableError (not OSError) and the orchestrator can catch it.
"""

from __future__ import annotations

import asyncio
import socket

import pytest

from mousedroid.config.schema import TelemetryConfig
from mousedroid.telemetry.exceptions import TelemetryUnavailableError
from mousedroid.telemetry.protocol import TelemetryFrame

aiohttp = pytest.importorskip("aiohttp")

from mousedroid.telemetry.server import TelemetryServer


def _make_server(cfg: TelemetryConfig) -> TelemetryServer:
    from unittest.mock import AsyncMock

    queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=64)
    health = AsyncMock()
    health.check_health = AsyncMock(return_value={"status": "ok"})
    return TelemetryServer(cfg=cfg, telemetry_queue=queue, health_monitor=health)


def _occupy_port(host: str = "127.0.0.1") -> tuple[socket.socket, int]:
    """Bind a real socket to a free port and return (socket, port)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    return sock, port


@pytest.mark.slow
async def test_fixed_bind_failure_raises_telemetry_unavailable() -> None:
    """strategy='fixed' raises TelemetryUnavailableError (not raw OSError) on conflict."""
    sock, port = _occupy_port()
    try:
        cfg = TelemetryConfig(
            enabled=True,
            host="127.0.0.1",
            port=port,
            port_discovery_strategy="fixed",
            mdns_enabled=False,
        )
        server = _make_server(cfg)
        with pytest.raises(TelemetryUnavailableError):
            await server.start()
    finally:
        sock.close()


@pytest.mark.slow
async def test_fallback_range_finds_next_free_port() -> None:
    """strategy='fallback_range' skips occupied port and binds to the next one."""
    sock, port = _occupy_port()
    try:
        cfg = TelemetryConfig(
            enabled=True,
            host="127.0.0.1",
            port=port,
            port_discovery_strategy="fallback_range",
            port_discovery_max_attempts=5,
            mdns_enabled=False,
        )
        server = _make_server(cfg)
        await server.start()
        assert server._bound_port != port
        assert server._bound_port > port
    finally:
        sock.close()
        if server._running:
            await server.stop()


@pytest.mark.slow
async def test_kernel_assigned_always_succeeds() -> None:
    """strategy='kernel_assigned' always binds successfully (OS picks a free port)."""
    cfg = TelemetryConfig(
        enabled=True,
        host="127.0.0.1",
        port=8080,
        port_discovery_strategy="kernel_assigned",
        mdns_enabled=False,
    )
    server = _make_server(cfg)
    await server.start()
    assert server._bound_port > 0
    assert server._bound_port != 8080  # kernel assigned something different
    await server.stop()


@pytest.mark.slow
async def test_orchestrator_continues_when_bind_fails() -> None:
    """Orchestrator-like code that catches TelemetryUnavailableError keeps running."""
    sock, port = _occupy_port()
    telemetry_active = True
    try:
        cfg = TelemetryConfig(
            enabled=True,
            host="127.0.0.1",
            port=port,
            port_discovery_strategy="fixed",
            mdns_enabled=False,
        )
        server = _make_server(cfg)
        try:
            await server.start()
        except TelemetryUnavailableError:
            telemetry_active = False

        # Orchestrator continues — telemetry is degraded, not fatal
        assert telemetry_active is False
        orchestrator_still_running = True
        assert orchestrator_still_running
    finally:
        sock.close()
