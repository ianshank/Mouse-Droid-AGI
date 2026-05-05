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

from pydantic import BaseModel, Field, ValidationError

from mousedroid.common.async_utils import cancel_and_drain, spawn_tracked
from mousedroid.common.rate_limit import TokenBucket
from mousedroid.constants import MAX_LOG_ENTRIES, MDNS_SERVICE_TYPE, TELEMETRY_QUEUE_TIMEOUT_S
from mousedroid.logging.setup import get_logger
from mousedroid.security.injection_filter import InjectionRejected
from mousedroid.telemetry.network import (
    get_default_ip,
    get_interface_ip,
    get_network_interfaces,
)
from mousedroid.telemetry.protocol import TelemetryFrame

if TYPE_CHECKING:
    from aiohttp import web

    from mousedroid.config.schema import OpenClawConfig, TelemetryConfig
    from mousedroid.hardware.protocols import RawFrameSourceProtocol
    from mousedroid.health.monitor import HealthMonitor
    from mousedroid.orchestrator.mission_dispatcher import MissionDispatcherProtocol
    from mousedroid.telemetry.log_buffer import LogRingBuffer
    from mousedroid.telemetry.metrics import MetricsRegistry
    from mousedroid.telemetry.protocol import TelemetryPublisherProtocol


class MissionRequest(BaseModel):
    """Body schema for ``POST /api/v1/mission``.

    Defined at module scope so OpenClaw clients (and the test suite) can
    import it without instantiating the whole server. ``channel`` is fixed
    to ``"rest"`` so a future MCP-routed call cannot impersonate a REST
    submission on this endpoint.
    """

    nl_command: str = Field(..., description="Natural language mission command")
    idempotency_key: str | None = Field(
        None,
        description="Optional dedup token; replays within the window return 202 with cached body.",
    )
    channel: str = Field("rest", description="Channel marker; pinned to 'rest' for this endpoint.")


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
        lidar_max_range_m: float | None = None,
        raw_frame_source: RawFrameSourceProtocol | None = None,
        raw_frame_hz: float = 10.0,
        cloud_enabled: bool = False,
        mission_dispatcher: MissionDispatcherProtocol | None = None,
        openclaw_cfg: OpenClawConfig | None = None,
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
            lidar_max_range_m: LiDAR maximum detection range in metres,
                used to convert normalised sector values into metres for
                Prometheus scrape. ``None`` disables per-sector metrics
                (keeps backwards compatibility when no LiDAR configured).
            raw_frame_source: Optional camera driver exposing
                :meth:`RawFrameSourceProtocol.capture_raw_jpeg`. When
                provided, the ``/camera/stream`` MJPEG endpoint is
                registered.
            raw_frame_hz: Target frame rate for ``/camera/stream``.
            cloud_enabled: When ``True``, register cloud-health routes when
                a metrics registry is also present.
            mission_dispatcher: Optional :class:`MissionDispatcherProtocol`.
                When provided together with an enabled
                :class:`OpenClawConfig`, the server registers
                ``POST /api/v1/mission`` for the OpenClaw control plane.
                Without it the route is not registered, which preserves
                existing deployments byte-identically.
            openclaw_cfg: Optional :class:`OpenClawConfig`. When supplied
                and ``enabled=True``, gates registration of the mission
                endpoint and supplies its rate-limit / dedup parameters.
        """
        self._cfg = cfg
        self._queue = telemetry_queue
        self._health_monitor = health_monitor
        self._log_buffer = log_buffer
        self._metrics: MetricsRegistry | None = metrics_registry
        self._metrics_path = metrics_path or self._cfg.metrics_path
        self._publisher = publisher
        self._lidar_max_range_m = lidar_max_range_m
        self._raw_frame_source = raw_frame_source
        self._raw_frame_interval_s = 1.0 / max(0.1, raw_frame_hz)
        self._cloud_enabled = cloud_enabled
        self._reported_frame_drops = 0

        self._mission_dispatcher = mission_dispatcher
        self._openclaw_cfg = openclaw_cfg
        self._mission_route_enabled = (
            mission_dispatcher is not None and openclaw_cfg is not None and openclaw_cfg.enabled
        )
        if self._mission_route_enabled and openclaw_cfg is not None:
            self._mission_rate_limiter: TokenBucket | None = TokenBucket(
                openclaw_cfg.rest_rate_limit_rps,
                capacity=float(openclaw_cfg.rest_rate_limit_burst),
            )
            self._mission_dedup_window_s: float = openclaw_cfg.command_dedup_window_s
        else:
            self._mission_rate_limiter = None
            self._mission_dedup_window_s = 0.0
        # idempotency_key -> (expires_monotonic, cached_response_dict)
        self._mission_dedup: dict[str, tuple[float, dict[str, Any]]] = {}

        self._ws_clients: list[web.WebSocketResponse] = []
        self._latest_frame: TelemetryFrame | None = None
        self._running = False
        self._broadcast_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()
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
        self._broadcast_task = spawn_tracked(
            self._background_tasks,
            self._broadcast_loop(),
            name=self._broadcast_loop.__name__,
        )

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
            if self._broadcast_task in self._background_tasks:
                await cancel_and_drain(self._background_tasks)
            elif not self._broadcast_task.done():
                self._broadcast_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._broadcast_task
            self._background_tasks.discard(self._broadcast_task)
            self._broadcast_task = None

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
        if self._mission_route_enabled:
            app.router.add_post(f"{prefix}/mission", self._handle_mission_post)
        app.router.add_get(self._cfg.ws_path, self._handle_ws)
        app.router.add_get(f"{prefix}/logs/stream", self._handle_log_stream)
        app.router.add_get("/lidar", self._handle_lidar_page)
        app.router.add_get("/camera", self._handle_camera_page)
        if self._raw_frame_source is not None:
            app.router.add_get("/camera/stream", self._handle_camera_stream)
            app.router.add_get("/camera/frame.jpg", self._handle_camera_frame)
        if self._metrics is not None:
            app.router.add_get(self._metrics_path, self._handle_metrics)
            if self._cloud_enabled:
                app.router.add_get(f"{prefix}/health/cloud", self._handle_cloud_health)

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    def _build_middlewares(self) -> list[Any]:
        """Build middleware list based on config.

        Returns:
            List of aiohttp middleware functions.
        """
        from aiohttp import web

        from mousedroid.telemetry.auth import build_bearer_auth_middleware, build_cors_middleware

        cors_origins = self._cfg.cors_origins
        api_key = self._cfg.api_key
        auth_cfg = self._cfg.auth

        # Use the new CORS middleware from auth module
        middlewares: list[Any] = [build_cors_middleware(cors_origins)]

        # Bearer token auth takes priority if configured
        if auth_cfg is not None and auth_cfg.auth_enabled:
            middlewares.append(build_bearer_auth_middleware(auth_cfg))
        elif api_key is not None:
            # Legacy X-API-Key auth for backwards compatibility
            @web.middleware  # type: ignore[misc,untyped-decorator,unused-ignore]
            async def auth_middleware(
                request: web.Request,
                handler: Any,
            ) -> web.StreamResponse:
                """Validate API key for both REST and WebSocket requests.

                For WebSocket upgrade requests, accept the API key from either
                ``X-API-Key`` or ``?api_key=…``. Normal safe GET/HEAD browser
                navigations use the same query-param fallback because browsers
                cannot set custom headers on page navigations, MJPEG image
                requests, or WebSocket handshakes. Auth decisions stay
                centralized in middleware and share a uniform rejection path.
                """
                is_ws_upgrade = request.headers.get("Upgrade", "").lower() == "websocket"
                if is_ws_upgrade or request.method in {"GET", "HEAD"}:
                    # For WebSocket and safe browser navigations, accept key
                    # from query param OR header.
                    key = request.query.get("api_key", request.headers.get("X-API-Key", ""))
                else:
                    key = request.headers.get("X-API-Key", "")

                if key != api_key:
                    raise web.HTTPUnauthorized(text="Invalid or missing API key")

                resp: web.StreamResponse = await handler(request)
                return resp

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

    async def _handle_cloud_health(self, request: web.Request) -> web.Response:
        """GET /api/v1/health/cloud - cloud sink/export backlog health."""
        from aiohttp import web

        if not self._cloud_enabled or self._metrics is None:
            return web.json_response({"error": "cloud_metrics_disabled"}, status=503)
        return web.json_response(self._metrics.get_cloud_health_snapshot())

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

    # ------------------------------------------------------------------
    # OpenClaw mission endpoint
    # ------------------------------------------------------------------

    async def _handle_mission_post(self, request: web.Request) -> web.Response:
        """POST /api/v1/mission — OpenClaw NL command ingress.

        Wired only when both ``mission_dispatcher`` and an enabled
        :class:`OpenClawConfig` are supplied. Auth (bearer or X-API-Key)
        is enforced by the existing global middleware. This handler adds
        token-bucket rate limiting, idempotency-key dedup, prompt-injection
        rejection (delegated to the shared filter via the dispatcher),
        and a structured response carrying the dispatcher's ``trace_id``
        for end-to-end correlation.
        """
        from aiohttp import web

        peer = request.remote or "unknown"
        log = _log.bind(endpoint="mission", peer=peer)

        if not self._mission_route_enabled or self._mission_dispatcher is None:
            # Belt-and-braces; the route should not be registered when the
            # gate is closed. Log so an operator who hits this path during
            # a partial / mid-rollout deployment sees why the request was
            # refused (no silent 503s).
            log.warning("mission_endpoint_rejected", reason="openclaw_disabled")
            return web.json_response({"error": "openclaw_disabled"}, status=503)

        if self._mission_rate_limiter is not None and not await self._mission_rate_limiter.take():
            retry_after = self._mission_rate_limiter.retry_after_s()
            log.warning(
                "mission_endpoint_rejected",
                reason="rate_limited",
                retry_after_s=retry_after,
            )
            return web.json_response(
                {"error": "rate_limited", "retry_after_s": round(retry_after, 3)},
                status=429,
            )

        try:
            payload = await request.json()
        except (json.JSONDecodeError, ValueError):
            log.warning("mission_endpoint_rejected", reason="invalid_json")
            return web.json_response({"error": "invalid_json"}, status=400)

        try:
            req = MissionRequest.model_validate(payload)
        except ValidationError as exc:
            log.warning("mission_endpoint_rejected", reason="invalid_body")
            return web.json_response(
                {"error": "invalid_body", "details": exc.errors()},
                status=400,
            )

        # Idempotency: replay returns the cached body for the configured
        # window so OpenClaw can safely re-send on transient failures.
        now = time.monotonic()
        # Evict expired entries lazily (cheap; bounded by request rate).
        if self._mission_dedup:
            self._mission_dedup = {k: v for k, v in self._mission_dedup.items() if v[0] > now}
        if req.idempotency_key and req.idempotency_key in self._mission_dedup:
            cached = self._mission_dedup[req.idempotency_key][1]
            log.info(
                "mission_endpoint_dedup_hit",
                idempotency_key=req.idempotency_key,
            )
            return web.json_response(cached, status=202)

        log.info(
            "mission_endpoint_received",
            length=len(req.nl_command),
            has_idempotency_key=req.idempotency_key is not None,
        )

        try:
            result = await self._mission_dispatcher.dispatch(
                req.nl_command,
                channel=req.channel,
                peer=peer,
            )
        except InjectionRejected:
            return web.json_response(
                {"error": "invalid_command", "reason": "injection_pattern"},
                status=400,
            )
        except ValueError as exc:
            return web.json_response(
                {"error": "invalid_command", "reason": str(exc)},
                status=400,
            )
        except TimeoutError:
            log.warning("mission_endpoint_timeout")
            return web.json_response({"error": "timeout"}, status=504)
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("mission_endpoint_failed", error=f"{type(exc).__name__}:{exc}")
            return web.json_response({"error": "internal_error"}, status=500)

        body: dict[str, Any] = {
            "status": "accepted",
            "trace_id": result.trace_id,
            "command_hash": result.command_hash,
            "latency_ms": round(result.latency_ms, 3),
            "goal_vector": {
                "vx": result.goal_vector.vx_target,
                "vy": result.goal_vector.vy_target,
                "omega": result.goal_vector.omega_target,
            },
        }
        if req.idempotency_key:
            self._mission_dedup[req.idempotency_key] = (
                now + self._mission_dedup_window_s,
                body,
            )
        log.info(
            "mission_endpoint_dispatched",
            trace_id=result.trace_id,
            latency_ms=result.latency_ms,
        )
        return web.json_response(body, status=202)

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

    async def _handle_lidar_page(self, request: web.Request) -> web.Response:
        """GET /lidar — serve the static HTML polar-plot visualisation.

        The page subscribes to ``/ws`` for ``TelemetryFrame`` JSON and renders
        ``lidar_sectors`` as a polar heatmap on an HTML canvas.
        """
        from importlib import resources

        from aiohttp import web

        try:
            html = (
                resources.files("mousedroid.telemetry.static")
                .joinpath("lidar.html")
                .read_text(encoding="utf-8")
            )
        except (FileNotFoundError, ModuleNotFoundError):
            return web.Response(status=404, text="lidar_page_missing")

        return web.Response(
            body=html.encode("utf-8"),
            headers={
                "Content-Type": "text/html; charset=utf-8",
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-cache",
            },
        )

    async def _handle_camera_page(self, request: web.Request) -> web.Response:
        """GET /camera — serve the vision-feature heatmap visualisation.

        MSE-6 streams feature vectors (not raw frames); the page reshapes
        ``vision_features`` from ``/ws`` into a square heatmap.
        """
        from importlib import resources

        from aiohttp import web

        try:
            html = (
                resources.files("mousedroid.telemetry.static")
                .joinpath("camera.html")
                .read_text(encoding="utf-8")
            )
        except (FileNotFoundError, ModuleNotFoundError):
            return web.Response(status=404, text="camera_page_missing")

        return web.Response(
            body=html.encode("utf-8"),
            headers={
                "Content-Type": "text/html; charset=utf-8",
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-cache",
            },
        )

    async def _handle_camera_frame(self, request: web.Request) -> web.Response:
        """GET /camera/frame.jpg — single JPEG snapshot from the raw-frame source."""
        from aiohttp import web

        if self._raw_frame_source is None:
            return web.Response(status=404, text="raw_frame_source_unavailable")
        try:
            jpeg = await self._raw_frame_source.capture_raw_jpeg()
        except Exception as exc:  # pylint: disable=broad-except
            _log.warning("raw_frame_capture_failed", error=str(exc))
            return web.Response(status=503, text="capture_failed")
        if jpeg is None:
            return web.Response(status=503, text="no_frame")
        return web.Response(
            body=jpeg,
            headers={
                "Content-Type": "image/jpeg",
                "Cache-Control": "no-store",
            },
        )

    async def _handle_camera_stream(self, request: web.Request) -> web.StreamResponse:
        """GET /camera/stream — multipart/x-mixed-replace MJPEG stream.

        Standard browser-compatible MJPEG: repeated JPEG frames separated
        by a boundary. Runs until the client disconnects.
        """
        from aiohttp import web

        if self._raw_frame_source is None:
            return web.Response(status=404, text="raw_frame_source_unavailable")

        boundary = "mousedroidframe"
        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": f"multipart/x-mixed-replace; boundary={boundary}",
                "Cache-Control": "no-store",
                "Connection": "close",
                "Pragma": "no-cache",
            },
        )
        await resp.prepare(request)
        _log.info("camera_stream_client_connected", peer=str(request.remote))
        try:
            while self._running:
                try:
                    jpeg = await self._raw_frame_source.capture_raw_jpeg()
                except Exception as exc:  # pylint: disable=broad-except
                    _log.warning("raw_frame_capture_failed", error=str(exc))
                    await asyncio.sleep(self._raw_frame_interval_s)
                    continue
                if jpeg is None:
                    await asyncio.sleep(self._raw_frame_interval_s)
                    continue
                header = (
                    f"--{boundary}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n"
                ).encode("ascii")
                try:
                    await resp.write(header)
                    await resp.write(jpeg)
                    await resp.write(b"\r\n")
                except (ConnectionResetError, asyncio.CancelledError):
                    break
                await asyncio.sleep(self._raw_frame_interval_s)
        finally:
            _log.info("camera_stream_client_disconnected", peer=str(request.remote))
            with contextlib.suppress(Exception):
                await resp.write_eof()
        return resp

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
                        sub_queue.get(),
                        timeout=TELEMETRY_QUEUE_TIMEOUT_S,
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
                    self._queue.get(),
                    timeout=TELEMETRY_QUEUE_TIMEOUT_S,
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
                    if isinstance(gpu_temp, int | float):
                        self._metrics.set_gpu_temp_celsius(float(gpu_temp))
                safety = frame.safety
                if isinstance(safety, dict):
                    for law in safety.get("violations", []):
                        self._metrics.inc_safety_violation(str(law))
                lidar_enabled = (
                    self._lidar_max_range_m is not None or frame.lidar_sectors is not None
                )
                if lidar_enabled:
                    if frame.lidar_sectors is not None and self._lidar_max_range_m is not None:
                        self._metrics.set_lidar_sectors(
                            frame.lidar_sectors,
                            self._lidar_max_range_m,
                        )
                    if frame.lidar_min_dist_m is not None:
                        self._metrics.set_lidar_min_distance_m(frame.lidar_min_dist_m)
                    self._metrics.set_lidar_scan_points(frame.lidar_n_points)

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
