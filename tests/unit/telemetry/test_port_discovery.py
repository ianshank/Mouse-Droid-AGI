"""Unit tests for TelemetryServer port discovery strategies."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mousedroid.config.schema import TelemetryConfig
from mousedroid.telemetry.exceptions import TelemetryUnavailableError
from mousedroid.telemetry.protocol import TelemetryFrame

aiohttp = pytest.importorskip("aiohttp")

from mousedroid.telemetry.server import TelemetryServer


def _make_server(cfg: TelemetryConfig) -> TelemetryServer:
    queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=64)
    health = AsyncMock()
    health.check_health = AsyncMock(return_value={"status": "ok"})
    return TelemetryServer(cfg=cfg, telemetry_queue=queue, health_monitor=health)


def _mock_tcp_site_factory(
    *,
    fail_ports: set[int] | None = None,
    assigned_port: int = 9001,
) -> Any:
    """Return a mock TCPSite class.

    Args:
        fail_ports: Port numbers that raise OSError on start().
        assigned_port: Port reported by getsockname() for kernel_assigned strategy.
    """
    fail_ports = fail_ports or set()

    class FakeSite:
        def __init__(self, runner: Any, host: str, port: int) -> None:
            self._port = port
            mock_sock = MagicMock()
            mock_sock.getsockname.return_value = (host, assigned_port)
            self._server = MagicMock()
            self._server.sockets = [mock_sock]

        async def start(self) -> None:
            if self._port in fail_ports:
                raise OSError(f"Address already in use: {self._port}")

    return FakeSite


# ---------------------------------------------------------------------------
# "fixed" strategy
# ---------------------------------------------------------------------------


class TestFixedStrategy:
    """port_discovery_strategy='fixed' binds exactly once."""

    async def test_fixed_success(self) -> None:
        """Successful bind sets _bound_port to the configured port."""
        cfg = TelemetryConfig(enabled=True, port=8080, port_discovery_strategy="fixed")
        server = _make_server(cfg)
        fake_site_cls = _mock_tcp_site_factory()

        with (
            patch("aiohttp.web.TCPSite", fake_site_cls),
            patch.object(server, "_runner", MagicMock()),
            patch.object(server, "_broadcast_loop", AsyncMock()),
            patch.object(server, "_register_mdns", AsyncMock()),
        ):
            server._runner.setup = AsyncMock()
            server._cfg = TelemetryConfig(
                enabled=True,
                port=8080,
                port_discovery_strategy="fixed",
                mdns_enabled=False,
            )
            await server.start()

        assert server._bound_port == 8080

    async def test_fixed_raises_on_oserror(self) -> None:
        """OSError during fixed bind is re-raised as TelemetryUnavailableError."""
        cfg = TelemetryConfig(
            enabled=True, port=8080, port_discovery_strategy="fixed", mdns_enabled=False
        )
        server = _make_server(cfg)
        fake_site_cls = _mock_tcp_site_factory(fail_ports={8080})

        with (
            patch("aiohttp.web.TCPSite", fake_site_cls),
            patch.object(server, "_runner", MagicMock()),
        ):
            server._runner.setup = AsyncMock()
            with pytest.raises(TelemetryUnavailableError, match="8080"):
                await server.start()


# ---------------------------------------------------------------------------
# "fallback_range" strategy
# ---------------------------------------------------------------------------


class TestFallbackRangeStrategy:
    """port_discovery_strategy='fallback_range' steps through port offsets."""

    async def test_fallback_succeeds_on_second_port(self) -> None:
        """When port N fails, port N+1 is tried and succeeds."""
        cfg = TelemetryConfig(
            enabled=True,
            port=8080,
            port_discovery_strategy="fallback_range",
            port_discovery_max_attempts=5,
            mdns_enabled=False,
        )
        server = _make_server(cfg)
        fake_site_cls = _mock_tcp_site_factory(fail_ports={8080})

        with (
            patch("aiohttp.web.TCPSite", fake_site_cls),
            patch.object(server, "_runner", MagicMock()),
            patch.object(server, "_broadcast_loop", AsyncMock()),
        ):
            server._runner.setup = AsyncMock()
            await server.start()

        assert server._bound_port == 8081

    async def test_fallback_exhausted_raises(self) -> None:
        """TelemetryUnavailableError when all ports in the range are occupied."""
        cfg = TelemetryConfig(
            enabled=True,
            port=8080,
            port_discovery_strategy="fallback_range",
            port_discovery_max_attempts=3,
            mdns_enabled=False,
        )
        server = _make_server(cfg)
        fake_site_cls = _mock_tcp_site_factory(fail_ports={8080, 8081, 8082})

        with (
            patch("aiohttp.web.TCPSite", fake_site_cls),
            patch.object(server, "_runner", MagicMock()),
        ):
            server._runner.setup = AsyncMock()
            with pytest.raises(TelemetryUnavailableError, match="3"):
                await server.start()

    async def test_fallback_first_port_available(self) -> None:
        """When the base port is free, it is used without retries."""
        cfg = TelemetryConfig(
            enabled=True,
            port=9000,
            port_discovery_strategy="fallback_range",
            port_discovery_max_attempts=10,
            mdns_enabled=False,
        )
        server = _make_server(cfg)
        fake_site_cls = _mock_tcp_site_factory()

        with (
            patch("aiohttp.web.TCPSite", fake_site_cls),
            patch.object(server, "_runner", MagicMock()),
            patch.object(server, "_broadcast_loop", AsyncMock()),
        ):
            server._runner.setup = AsyncMock()
            await server.start()

        assert server._bound_port == 9000


# ---------------------------------------------------------------------------
# "kernel_assigned" strategy
# ---------------------------------------------------------------------------


class TestKernelAssignedStrategy:
    """port_discovery_strategy='kernel_assigned' binds to OS-assigned port."""

    async def test_kernel_assigned_reads_actual_port(self) -> None:
        """_bound_port is set to the OS-assigned port reported by getsockname()."""
        cfg = TelemetryConfig(
            enabled=True,
            port=8080,
            port_discovery_strategy="kernel_assigned",
            mdns_enabled=False,
        )
        server = _make_server(cfg)
        fake_site_cls = _mock_tcp_site_factory(assigned_port=54321)

        with (
            patch("aiohttp.web.TCPSite", fake_site_cls),
            patch.object(server, "_runner", MagicMock()),
            patch.object(server, "_broadcast_loop", AsyncMock()),
        ):
            server._runner.setup = AsyncMock()
            await server.start()

        assert server._bound_port == 54321
