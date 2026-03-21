"""Telemetry server — aiohttp-based REST + WebSocket for remote monitoring.

Provides real-time sensor data streaming, health metrics, log retrieval,
and network interface information over WiFi and Ethernet connections.

All endpoints are async and run on the same event loop as the main
orchestrator. The server consumes ``TelemetryFrame`` objects from a
publisher queue and fans them out to connected WebSocket clients.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import TYPE_CHECKING, Any

from mousedroid.constants import MAX_LOG_ENTRIES, MDNS_SERVICE_TYPE, TELEMETRY_QUEUE_TIMEOUT_S
from mousedroid.logging.setup import get_logger
from mousedroid.telemetry.network import (
    get_default_ip,
    get_interface_ip,
    get_network_interfaces,
)
from mousedroid.telemetry.protocol import TelemetryFrame

if TYPE_CHECKING:
    from aiohttp import web

    from mousedroid.config.schema import TelemetryConfig
    from mousedroid.health.monitor import HealthMonitor
    from mousedroid.telemetry.log_buffer import LogRingBuffer
    from mousedroid.telemetry.metrics import MetricsRegistry
    from mousedroid.telemetry.protocol import TelemetryPublisherProtocol

_log = get_logger(__name__)

_STARTUP_TIME: float = time.monotonic()


class TelemetryServer:
    """aiohttp-based telemetry server for remote monitoring.

    Provides REST endpoints for snapshots and WebSocket endpoints for
    real-time streaming. Supports optional API key authentication,
    CORS, mDNS/Zeroconf service registration, and configurable
    serialisation (JSON or msgpack).

    Implements ``TelemetryServerProtocol``.
    """

    def __init__(
        self,
        cfg: TelemetryConfig,
        telemetry_queue: asyncio.Queue[TelemetryFrame],
        health_monitor: HealthMonitor,
        log_buffer: LogRingBuffer | None = None,
        metrics_registry: MetricsRegistry | None = None,
        metrics_path: str | None = None,
        publisher: TelemetryPublisherProtocol | None = None,
    ) -> None:
        """Initialise the telemetry server.

        Args:
            cfg: Telemetry configuration.
            telemetry_queue: Queue to consume ``TelemetryFrame`` objects from.
            health_monitor: Health monitor for ``/health`` endpoint.
            log_buffer: Optional log ring buffer for ``/logs`` endpoint.
            metrics_registry: Optional Prometheus metrics registry.  When
                provided, exposes a ``/metrics`` scrape endpoint and updates
                counters/gauges from every broadcast frame.
            metrics_path: HTTP path for the metrics endpoint.  When omitted,
                falls back to ``TelemetryConfig.metrics_path`` for backwards
                compatibility with direct ``TelemetryServer`` construction.
            publisher: Optional telemetry publisher used to synchronise
                publisher-level stats such as dropped-frame counters.
        """
        self._cfg = cfg
        self._queue = telemetry_queue
        self._health_monitor = health_monitor
        self._log_buffer = log_buffer
        self._metrics: MetricsRegistry | None = metrics_registry
        self._metrics_path = metrics_path or self._cfg.metrics_path
        self._publisher = publisher
        self._reported_frame_drops = 0

        self._ws_clients: list[web.WebSocketResponse] = []
        self._latest_frame: TelemetryFrame | None = None
        self._running = False
        self._broadcast_task: asyncio.Task[None] | None = None
        self._runner: web.AppRunner | None = None
        self._zeroconf: Any = None
        self._service_info: Any = None

        if self._metrics is not None:
            self._metrics.set_publish_hz(self._cfg.publish_hz)

    async def start(self) -> None:
        """Start the aiohttp server and background broadcast loop."""
        from aiohttp import web

        app = web.Application(middlewares=self._build_middlewares())
        self._register_routes(app)

        self._runner = web.AppRunner(app)
        await self._runner.setup()

        site = web.TCPSite(self._runner, self._cfg.host, self._cfg.port)
        await site.start()

        self._running = True
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

        if self._cfg.mdns_enabled:
            await self._register_mdns()

        _log.info(
            "telemetry_server_started",
            host=self._cfg.host,
            port=self._cfg.port,
        )

    async def stop(self) -> None:
        """Gracefully shut down server, close all WebSocket connections."""
        _log.info("telemetry_server_stopping")
        self._running = False

        if self._broadcast_task is not None:
            self._broadcast_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._broadcast_task

        for ws in list(self._ws_clients):
            await ws.close()
        self._ws_clients.clear()

        if self._cfg.mdns_enabled:
            await self._unregister_mdns()

        if self._runner is not None:
            await self._runner.cleanup()

        _log.info("telemetry_server_stopped")

    @property
    def client_count(self) -> int:
        """Number of currently connected WebSocket clients."""
        return len(self._ws_clients)

    @property
    def is_running(self) -> bool:
        """Whether the server is currently running."""
        return self._running

    # ------------------------------------------------------------------
    # Route registration
    # ------------------------------------------------------------------

    def _register_routes(self, app: web.Application) -> None:
        """Register all REST and WebSocket routes.

        Args:
            app: The aiohttp application.
        """
        prefix = self._cfg.api_prefix

        app.router.add_get(f"{prefix}/status", self._handle_status)
        app.router.add_get(f"{prefix}/sensors", self._handle_sensors)
        app.router.add_get(f"{prefix}/health", self._handle_health)
        app.router.add_get(f"{prefix}/logs", self._handle_logs)
        app.router.add_get(f"{prefix}/network", self._handle_network)
        app.router.add_get(self._cfg.ws_path, self._handle_ws)
        app.router.add_get(f"{prefix}/logs/stream", self._handle_log_stream)
        app.router.add_get(f"{prefix}/openclaw", self._handle_openclaw_status)
        if self._metrics is not None:
            app.router.add_get(self._metrics_path, self._handle_metrics)

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    def _build_middlewares(self) -> list[Any]:
        """Build middleware list based on config.

        Returns:
            List of aiohttp middleware functions.
        """
        from aiohttp import web

        cors_origins = self._cfg.cors_origins
        api_key = self._cfg.api_key

        @web.middleware
        async def cors_middleware(
            request: web.Request,
            handler: Any,
        ) -> web.StreamResponse:
            """Add CORS headers to responses."""
            if request.method == "OPTIONS":
                resp = web.Response()
            else:
                resp = await handler(request)

            # Determine which, if any, Access-Control-Allow-Origin header to send.
            # - If CORS is unrestricted (no origins configured or ["*"]), always use "*".
            # - If specific origins are configured, only echo back an allowed Origin
            #   from the request and add Vary: Origin. For disallowed/missing origins,
            #   omit Access-Control-Allow-Origin entirely.
            origin_header_value: str | None = None
            if not cors_origins or cors_origins == ["*"]:
                origin_header_value = "*"
            else:
                req_origin = request.headers.get("Origin")
                if req_origin and req_origin in cors_origins:
                    origin_header_value = req_origin
                    resp.headers.add("Vary", "Origin")

            if origin_header_value is not None:
                resp.headers["Access-Control-Allow-Origin"] = origin_header_value
            resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "X-API-Key, Content-Type"
            return resp

        @web.middleware
        async def auth_middleware(
            request: web.Request,
            handler: Any,
        ) -> web.StreamResponse:
            """Validate API key for both REST and WebSocket requests.

            For WebSocket upgrade requests, accept the API key from either
            ``X-API-Key`` or ``?api_key=…`` so auth decisions stay centralized
            in middleware and share a uniform rejection path.
            """
            is_ws_upgrade = request.headers.get("Upgrade", "").lower() == "websocket"
            if is_ws_upgrade:
                # For WebSocket, accept key from query param OR header
                key = request.query.get("api_key", request.headers.get("X-API-Key", ""))
            else:
                key = request.headers.get("X-API-Key", "")

            if key != api_key:
                raise web.HTTPUnauthorized(text="Invalid or missing API key")

            resp: web.StreamResponse = await handler(request)
            return resp

        middlewares: list[Any] = [cors_middleware]
        if api_key is not None:
            middlewares.append(auth_middleware)
        return middlewares

    # ------------------------------------------------------------------
    # REST handlers
    # ------------------------------------------------------------------

    async def _handle_status(self, request: web.Request) -> web.Response:
        """GET /api/v1/status — system status overview.

        Args:
            request: The incoming request.

        Returns:
            JSON response with system status.
        """
        from aiohttp import web

        uptime_s = time.monotonic() - _STARTUP_TIME
        tick_count = self._latest_frame.tick_count if self._latest_frame else 0

        data = {
            "status": "running" if self._running else "stopped",
            "uptime_s": round(uptime_s, 2),
            "tick_count": tick_count,
            "ws_clients": self.client_count,
        }
        return web.json_response(data)

    async def _handle_sensors(self, request: web.Request) -> web.Response:
        """GET /api/v1/sensors — latest sensor snapshot.

        Args:
            request: The incoming request.

        Returns:
            JSON response with latest TelemetryFrame.
        """
        from aiohttp import web

        if self._latest_frame is None:
            return web.json_response({"error": "no_data"}, status=503)
        return web.json_response(self._latest_frame.to_dict())

    async def _handle_health(self, request: web.Request) -> web.Response:
        """GET /api/v1/health — health metrics from HealthMonitor.

        Args:
            request: The incoming request.

        Returns:
            JSON response with health metrics.
        """
        from aiohttp import web

        health = await self._health_monitor.check_health()

        if self._latest_frame is not None:
            health["battery_voltage"] = self._latest_frame.battery_voltage
            health["safety"] = self._latest_frame.safety

        return web.json_response(health)

    async def _handle_logs(self, request: web.Request) -> web.Response:
        """GET /api/v1/logs?n=50 — recent log entries from ring buffer.

        Args:
            request: The incoming request.

        Returns:
            JSON response with recent log entries.
        """
        from aiohttp import web

        if self._log_buffer is None:
            return web.json_response({"error": "log_buffer_disabled"}, status=503)

        raw_n = request.query.get("n", "50")
        try:
            n = int(raw_n)
        except (TypeError, ValueError):
            return web.json_response(
                {"error": "invalid_n", "message": "Query parameter 'n' must be an integer."},
                status=400,
            )

        # Clamp to a non-negative, sensible range.
        if n < 0:
            n = 0
        if n > MAX_LOG_ENTRIES:
            n = MAX_LOG_ENTRIES

        entries = self._log_buffer.get_recent(n)

        serialisable = []
        for entry in entries:
            row: dict[str, Any] = {}
            for k, v in entry.items():
                try:
                    json.dumps(v)
                    row[k] = v
                except (TypeError, ValueError):
                    row[k] = str(v)
            serialisable.append(row)

        return web.json_response({"entries": serialisable, "count": len(serialisable)})

    async def _handle_network(self, request: web.Request) -> web.Response:
        """GET /api/v1/network — network interface information.

        Args:
            request: The incoming request.

        Returns:
            JSON response with interface details.
        """
        from aiohttp import web

        interfaces = await get_network_interfaces()
        default_ip = get_default_ip()
        preferred_iface = self._cfg.preferred_interface
        if preferred_iface:
            preferred_ip = await get_interface_ip(preferred_iface)
            if preferred_ip:
                default_ip = preferred_ip

        data = {
            "interfaces": [iface.to_dict() for iface in interfaces],
            "server_url": f"http://{default_ip}:{self._cfg.port}",
            "server_port": self._cfg.port,
        }

        if self._cfg.mdns_enabled:
            data["mdns_name"] = f"{self._cfg.mdns_service_name.lower().replace(' ', '-')}.local"

        return web.json_response(data)

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        """GET /metrics — Prometheus text-format metrics scrape endpoint.

        Args:
            request: The incoming request.

        Returns:
            Plain-text Prometheus exposition with ``Content-Type:
            text/plain; version=0.0.4; charset=utf-8``.
        """
        from aiohttp import web

        if self._metrics is None:
            return web.Response(status=404, text="metrics_disabled")

        text = self._metrics.render_prometheus()
        return web.Response(
            text=text,
            headers={
                "Content-Type": "text/plain; version=0.0.4; charset=utf-8",
                "X-Content-Type-Options": "nosniff",
            },
        )

    async def _handle_openclaw_status(self, request: web.Request) -> web.Response:
        """Return current OpenClaw gateway state from latest telemetry frame.

        Args:
            request: The incoming HTTP request.

        Returns:
            JSON response with OpenClaw state.
        """
        frame = self._latest_frame
        openclaw_data: dict[str, object] = {}
        if frame is not None:
            openclaw_data = frame.openclaw
        return web.json_response({"openclaw": openclaw_data})

    # ------------------------------------------------------------------
    # WebSocket handlers
    # ------------------------------------------------------------------

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket handler — stream TelemetryFrames to connected clients.

        Args:
            request: The incoming WebSocket upgrade request.

        Returns:
            The WebSocket response (kept alive until client disconnects).
        """
        from aiohttp import WSMsgType, web

        if len(self._ws_clients) >= self._cfg.max_clients:
            resp = web.WebSocketResponse()
            await resp.prepare(request)
            await resp.close(code=4029, message=b"max_clients_reached")
            return resp
        # Note: API key auth is already enforced by auth_middleware when
        # cfg.api_key is set.  No secondary check needed here.

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self._ws_clients.append(ws)
        peer = request.remote or "unknown"
        _log.info("telemetry_ws_client_connected", peer=peer, total=len(self._ws_clients))

        try:
            async for msg in ws:
                if msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            if ws in self._ws_clients:
                self._ws_clients.remove(ws)
            _log.info("telemetry_ws_client_disconnected", peer=peer, total=len(self._ws_clients))

        return ws

    async def _handle_log_stream(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket handler — stream live log entries.

        Args:
            request: The incoming WebSocket upgrade request.

        Returns:
            The WebSocket response.
        """
        from aiohttp import web

        if self._log_buffer is None:
            resp = web.WebSocketResponse()
            await resp.prepare(request)
            await resp.close(code=4030, message=b"log_buffer_disabled")
            return resp

        # Enforce API key for WebSocket log streaming if configured.
        config = getattr(self, "_config", None)
        api_key = getattr(config, "api_key", None) if config is not None else None
        if api_key:
            supplied_key = request.headers.get("X-Telemetry-Api-Key") or request.query.get(
                "api_key"
            )
            if supplied_key != api_key:
                raise web.HTTPUnauthorized(
                    text="Invalid or missing API key for log stream WebSocket"
                )

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        sub_queue = self._log_buffer.subscribe()
        try:
            while not ws.closed and self._running:
                try:
                    entry = await asyncio.wait_for(
                        sub_queue.get(), timeout=TELEMETRY_QUEUE_TIMEOUT_S,
                    )
                    serialisable: dict[str, Any] = {}
                    for k, v in entry.items():
                        try:
                            json.dumps(v)
                            serialisable[k] = v
                        except (TypeError, ValueError):
                            serialisable[k] = str(v)
                    await ws.send_json(serialisable)
                except asyncio.TimeoutError:
                    continue
                except (ConnectionResetError, RuntimeError):
                    break
        finally:
            self._log_buffer.unsubscribe(sub_queue)

        return ws

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------

    async def _broadcast_loop(self) -> None:
        """Consume from telemetry queue and fan-out to all WS clients."""
        while self._running:
            try:
                frame = await asyncio.wait_for(
                    self._queue.get(), timeout=TELEMETRY_QUEUE_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                continue

            self._latest_frame = frame
            data = frame.to_dict()

            # Push live telemetry into metrics registry (non-blocking)
            if self._metrics is not None:
                self._metrics.set_loop_time_ms(frame.loop_time_ms)
                self._metrics.set_battery_voltage(frame.battery_voltage)
                self._metrics.set_ws_client_count(len(self._ws_clients))
                self._sync_publisher_metrics()
                health = frame.health
                if isinstance(health, dict):
                    gpu_temp = health.get("gpu_temp_c")
                    if isinstance(gpu_temp, (int, float)):
                        self._metrics.set_gpu_temp_celsius(float(gpu_temp))
                safety = frame.safety
                if isinstance(safety, dict):
                    for law in safety.get("violations", []):
                        self._metrics.inc_safety_violation(str(law))

            dead_clients: list[web.WebSocketResponse] = []
            send_tasks = []

            for ws in self._ws_clients:
                if ws.closed:
                    dead_clients.append(ws)
                    continue
                if self._cfg.serialization == "msgpack":
                    send_tasks.append(self._send_msgpack(ws, data))
                else:
                    send_tasks.append(self._send_json(ws, data))

            if send_tasks:
                await asyncio.gather(*send_tasks, return_exceptions=True)

            for ws in dead_clients:
                if ws in self._ws_clients:
                    self._ws_clients.remove(ws)

    def _sync_publisher_metrics(self) -> None:
        """Synchronise publisher-owned stats into the metrics registry."""
        if self._metrics is None or self._publisher is None:
            return

        dropped_total = self._publisher.stats.get("frames_dropped", 0)
        if dropped_total > self._reported_frame_drops:
            self._metrics.inc_frame_drops(dropped_total - self._reported_frame_drops)
        self._reported_frame_drops = dropped_total

    @staticmethod
    async def _send_json(ws: web.WebSocketResponse, data: dict[str, Any]) -> None:
        """Send JSON data to a WebSocket client.

        Args:
            ws: The WebSocket response.
            data: Dictionary to serialise as JSON.
        """
        with contextlib.suppress(ConnectionResetError, RuntimeError):
            await ws.send_json(data)

    @staticmethod
    async def _send_msgpack(ws: web.WebSocketResponse, data: dict[str, Any]) -> None:
        """Send msgpack data to a WebSocket client.

        Args:
            ws: The WebSocket response.
            data: Dictionary to serialise as msgpack.
        """
        import msgpack

        try:
            packed = msgpack.packb(data, use_bin_type=True)
            await ws.send_bytes(packed)
        except (ConnectionResetError, RuntimeError):
            pass

    # ------------------------------------------------------------------
    # mDNS / Zeroconf
    # ------------------------------------------------------------------

    async def _register_mdns(self) -> None:
        """Register telemetry service via Zeroconf for LAN discovery."""
        try:
            from zeroconf import ServiceInfo, Zeroconf

            ip = get_default_ip()
            preferred_iface = self._cfg.preferred_interface
            if preferred_iface:
                preferred_ip = await get_interface_ip(preferred_iface)
                if preferred_ip:
                    ip = preferred_ip
            import socket as sock

            packed_ip = sock.inet_aton(ip)

            self._service_info = ServiceInfo(
                type_=MDNS_SERVICE_TYPE,
                name=f"{self._cfg.mdns_service_name}.{MDNS_SERVICE_TYPE}",
                addresses=[packed_ip],
                port=self._cfg.port,
                properties={
                    "path": self._cfg.ws_path,
                    "api": self._cfg.api_prefix,
                },
            )
            self._zeroconf = Zeroconf()
            await asyncio.to_thread(
                self._zeroconf.register_service,
                self._service_info,
            )
            _log.info(
                "telemetry_mdns_registered",
                service_name=self._cfg.mdns_service_name,
                ip=ip,
                port=self._cfg.port,
            )
        except ImportError:
            _log.warning("telemetry_mdns_zeroconf_not_installed")
        except Exception as exc:
            _log.warning("telemetry_mdns_registration_failed", error=str(exc))

    async def _unregister_mdns(self) -> None:
        """Unregister mDNS service."""
        try:
            if self._zeroconf is not None and self._service_info is not None:
                await asyncio.to_thread(
                    self._zeroconf.unregister_service,
                    self._service_info,
                )
                await asyncio.to_thread(self._zeroconf.close)
                _log.info("telemetry_mdns_unregistered")
        except Exception as exc:
            _log.warning("telemetry_mdns_unregistration_failed", error=str(exc))
        finally:
            self._zeroconf = None
            self._service_info = None
