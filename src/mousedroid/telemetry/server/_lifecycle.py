"""Lifecycle mixin for ``TelemetryServer`` — construction, start/stop, mDNS.

Holds ``__init__``, ``start``/``stop``, the ``client_count``/``is_running``
properties, route + middleware registration, and mDNS/Zeroconf
registration. Split out of the former monolithic ``telemetry/server.py``;
see ``telemetry/server/__init__.py`` for how this mixin is composed with
``_RestHandlersMixin`` and ``_WebSocketHandlersMixin`` into the final
``TelemetryServer`` class.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import time
from typing import TYPE_CHECKING, Any

from mousedroid.common.async_utils import cancel_and_drain, spawn_tracked
from mousedroid.common.rate_limit import TokenBucket
from mousedroid.constants import MDNS_SERVICE_TYPE
from mousedroid.logging.setup import get_logger
from mousedroid.telemetry.protocol import LidarRawScan, TelemetryFrame
from mousedroid.telemetry.server._state import _TelemetryServerState

if TYPE_CHECKING:
    from aiohttp import web

    from mousedroid.config.schema import OpenClawConfig, TelemetryConfig
    from mousedroid.hardware.protocols import RawFrameSourceProtocol
    from mousedroid.health.monitor import HealthMonitor
    from mousedroid.orchestrator.mission_dispatcher import MissionDispatcherProtocol
    from mousedroid.telemetry.failure_recorder import FailureRecorder
    from mousedroid.telemetry.log_buffer import LogRingBuffer
    from mousedroid.telemetry.metrics import MetricsRegistry
    from mousedroid.telemetry.protocol import TelemetryPublisherProtocol
    from mousedroid.telemetry.serialization import SerializationName
    from mousedroid.telemetry.server._protocol import _ComposedServerProtocol

_log = get_logger(__name__)

_STARTUP_TIME: float = time.monotonic()


class _LifecycleMixin(_TelemetryServerState):
    """Construction, start/stop, route + middleware wiring, and mDNS."""

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
        failure_recorder: FailureRecorder | None = None,
        lidar_raw_queue: asyncio.Queue[LidarRawScan] | None = None,
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
            failure_recorder: Optional :class:`FailureRecorder` for recording
                bind failures and auth rejections as Prometheus metrics.
                Defaults to a no-op when ``None``.
            lidar_raw_queue: Optional ``asyncio.Queue[LidarRawScan]``
                produced by the publisher's raw-LiDAR channel. When
                supplied, the server spawns an additional broadcast
                loop that fans scans out to ``/ws/v1/lidar/raw``
                subscribers; otherwise the endpoint returns close-code
                4404.
        """
        from mousedroid.telemetry.failure_recorder import NullFailureRecorder

        self._failure_recorder: FailureRecorder = (
            failure_recorder if failure_recorder is not None else NullFailureRecorder()
        )
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
        # PR #4: separate publisher counter for raw LiDAR drops.
        self._reported_lidar_raw_drops = 0

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
        # Two-tier idempotency state:
        # - ``_mission_dedup``  : completed responses keyed by
        #   idempotency_key, paired with their expiry timestamp. Replays
        #   inside the dedup window return the cached body.
        # - ``_mission_inflight``: futures for in-progress dispatches.
        #   Concurrent retries with the same key block on the leader's
        #   future instead of starting a parallel dispatch — so the
        #   contract is "at most one downstream dispatch per
        #   idempotency_key per dedup window" even under burst replays.
        self._mission_dedup: dict[str, tuple[float, dict[str, Any]]] = {}
        self._mission_inflight: dict[str, asyncio.Future[tuple[int, dict[str, Any]]]] = {}

        self._ws_clients: list[web.WebSocketResponse] = []
        # PR #4: per-client serialization choice (defaults to cfg.serialization)
        # filled in by the optional hello-negotiation handshake.
        self._ws_serializations: dict[int, SerializationName] = {}
        # PR #81: set of ``id(ws)`` values for clients whose
        # ``await ws.prepare(request)`` has completed. The broadcast
        # loop only sends frames to clients in this set so frames
        # produced during the synchronous ``max_clients`` reservation
        # window (after append, before prepare) are not delivered to
        # an unprepared ``WebSocketResponse`` — that would otherwise
        # raise ``RuntimeError`` inside ``ws.send_json`` and get
        # silently swallowed by ``gather(..., return_exceptions=True)``.
        # Addresses Copilot review on PR #81.
        self._ws_prepared: set[int] = set()
        # PR #4: separate broadcast list for /ws/v1/lidar/raw clients.
        self._lidar_ws_clients: list[web.WebSocketResponse] = []
        self._lidar_ws_serializations: dict[int, SerializationName] = {}
        # PR #81: mirrors ``_ws_prepared`` for the raw-LiDAR endpoint.
        self._lidar_ws_prepared: set[int] = set()
        self._lidar_raw_queue: asyncio.Queue[LidarRawScan] | None = lidar_raw_queue
        self._latest_frame: TelemetryFrame | None = None
        self._latest_lidar_raw: LidarRawScan | None = None
        self._running = False
        self._broadcast_task: asyncio.Task[None] | None = None
        self._lidar_raw_broadcast_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._runner: web.AppRunner | None = None
        self._zeroconf: Any = None
        self._service_info: Any = None
        self._mdns_registered_event: asyncio.Event = asyncio.Event()
        self._mdns_ok: bool = False
        self._bound_port: int = self._cfg.port

        if self._metrics is not None:
            self._metrics.set_publish_hz(self._cfg.publish_hz)

    async def start(self: _ComposedServerProtocol) -> None:
        """Start the aiohttp server and background broadcast loop.

        Raises:
            TelemetryUnavailableError: When all candidate ports are exhausted.
        """
        from aiohttp import web

        from mousedroid.telemetry.exceptions import TelemetryUnavailableError

        app = web.Application(middlewares=self._build_middlewares())
        self._register_routes(app)

        self._runner = web.AppRunner(app)
        await self._runner.setup()

        strategy = self._cfg.port_discovery_strategy
        if strategy == "kernel_assigned":
            # Wrap in try/except OSError to match the ``fixed`` and
            # ``fallback_range`` strategies — an invalid host or
            # system-wide port exhaustion would otherwise raise a raw
            # OSError that bypasses the orchestrator's
            # TelemetryUnavailableError degradation handler. Addresses
            # PR #78 review (Gemini high + Copilot).
            try:
                site = web.TCPSite(self._runner, self._cfg.host, 0)
                await site.start()
            except OSError as exc:
                self._failure_recorder.record(
                    "telemetry",
                    "bind_failed",
                    level="error",
                    extra={"strategy": strategy, "host": self._cfg.host},
                )
                raise TelemetryUnavailableError(
                    f"telemetry: kernel_assigned bind on {self._cfg.host}:0 failed: {exc}"
                ) from exc
            # Read the actual OS-assigned port from the underlying socket.
            # ``site._server`` is typed Optional[asyncio.AbstractServer] in
            # aiohttp's stubs, but only the concrete asyncio.Server has
            # ``.sockets``. Using ``getattr`` with a default keeps both
            # old mypy (would flag union-attr) and new mypy (would flag
            # an unused inline type suppression) happy without sacrificing
            # the runtime contract.
            sockets = getattr(site._server, "sockets", None)
            self._bound_port = sockets[0].getsockname()[1] if sockets else self._cfg.port
            _log.info(
                "telemetry_port_bound",
                strategy=strategy,
                host=self._cfg.host,
                port=self._bound_port,
            )
        elif strategy == "fallback_range":
            max_attempts = self._cfg.port_discovery_max_attempts
            bound = False
            for offset in range(max_attempts):
                candidate = self._cfg.port + offset
                _log.info(
                    "telemetry_port_bind_attempt",
                    strategy=strategy,
                    host=self._cfg.host,
                    port=candidate,
                    attempt=offset + 1,
                    max_attempts=max_attempts,
                )
                try:
                    site = web.TCPSite(self._runner, self._cfg.host, candidate)
                    await site.start()
                    self._bound_port = candidate
                    bound = True
                    _log.info(
                        "telemetry_port_bound",
                        strategy=strategy,
                        host=self._cfg.host,
                        port=candidate,
                    )
                    break
                except OSError:
                    continue
            if not bound:
                self._failure_recorder.record(
                    "telemetry",
                    "bind_exhausted",
                    level="error",
                    extra={"port_start": self._cfg.port, "attempts": max_attempts},
                )
                raise TelemetryUnavailableError(
                    f"telemetry: all {max_attempts} candidate ports starting at "
                    f"{self._cfg.port} are in use on {self._cfg.host}"
                )
        else:  # "fixed"
            try:
                site = web.TCPSite(self._runner, self._cfg.host, self._cfg.port)
                await site.start()
                self._bound_port = self._cfg.port
                _log.info(
                    "telemetry_port_bound",
                    strategy=strategy,
                    host=self._cfg.host,
                    port=self._bound_port,
                )
            except OSError as exc:
                self._failure_recorder.record(
                    "telemetry",
                    "bind_failed",
                    level="error",
                    extra={"port": self._cfg.port},
                )
                raise TelemetryUnavailableError(
                    f"telemetry: cannot bind to {self._cfg.host}:{self._cfg.port}: {exc}"
                ) from exc

        self._running = True
        self._broadcast_task = spawn_tracked(
            self._background_tasks,
            self._broadcast_loop(),
            name=self._broadcast_loop.__name__,
        )

        # PR #4: spawn the raw LiDAR fan-out loop when a queue is wired.
        if self._lidar_raw_queue is not None:
            self._lidar_raw_broadcast_task = spawn_tracked(
                self._background_tasks,
                self._lidar_raw_broadcast_loop(),
                name=self._lidar_raw_broadcast_loop.__name__,
            )

        if self._metrics is not None:
            self._metrics.set_bound_port(self._bound_port)

        if self._cfg.mdns_enabled:
            # PR #4: bounded-wait mDNS registration. The thread-pool
            # call is launched in the background; we wait at most
            # ``mdns_register_timeout_s`` for the event to fire so a
            # slow / hung Zeroconf install can't stall startup.
            mdns_task = asyncio.create_task(self._register_mdns(), name="telemetry_mdns_register")
            self._background_tasks.add(mdns_task)
            mdns_task.add_done_callback(self._background_tasks.discard)
            try:
                await asyncio.wait_for(
                    self._mdns_registered_event.wait(),
                    timeout=self._cfg.mdns_register_timeout_s,
                )
            except asyncio.TimeoutError:
                self._failure_recorder.record(
                    "telemetry",
                    "mdns_register_timeout",
                    level="warning",
                    extra={"timeout_s": self._cfg.mdns_register_timeout_s},
                )
                _log.warning(
                    "telemetry_mdns_register_timeout",
                    timeout_s=self._cfg.mdns_register_timeout_s,
                )
                if self._metrics is not None:
                    self._metrics.set_mdns_registered(self._cfg.mdns_service_name, ok=False)

        _log.info(
            "telemetry_server_started",
            host=self._cfg.host,
            port=self._bound_port,
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
        self._ws_serializations.clear()
        self._ws_prepared.clear()

        for ws in list(self._lidar_ws_clients):
            await ws.close()
        self._lidar_ws_clients.clear()
        self._lidar_ws_serializations.clear()
        self._lidar_ws_prepared.clear()

        if self._lidar_raw_broadcast_task is not None and not self._lidar_raw_broadcast_task.done():
            self._lidar_raw_broadcast_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._lidar_raw_broadcast_task
        self._lidar_raw_broadcast_task = None

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

    def _register_routes(self: _ComposedServerProtocol, app: web.Application) -> None:
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
        # PR #4: live raw LiDAR streaming. Registered unconditionally —
        # if no publisher wired a raw queue, the handler returns 503.
        app.router.add_get(self._cfg.lidar_raw_ws_path, self._handle_lidar_raw_ws)
        app.router.add_get(f"{prefix}/logs/stream", self._handle_log_stream)
        app.router.add_get("/", self._handle_root)
        app.router.add_get("/dashboard", self._handle_dashboard_page)
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

        cors_origins = list(self._cfg.cors_origins)
        # If OpenClaw is wired and the operator named the Mac mini's
        # origin, splice it into the CORS allow-list so cross-host
        # requests from the OpenClaw dashboard succeed without forcing
        # operators to keep two YAML lists in sync.
        if (
            self._openclaw_cfg is not None
            and self._openclaw_cfg.mac_mini_origin
            and "*" not in cors_origins
            and self._openclaw_cfg.mac_mini_origin not in cors_origins
        ):
            cors_origins.append(self._openclaw_cfg.mac_mini_origin)
            _log.info(
                "telemetry_cors_origin_added",
                origin=self._openclaw_cfg.mac_mini_origin,
                source="openclaw.mac_mini_origin",
            )
        api_key = self._cfg.api_key.get_secret_value() if self._cfg.api_key is not None else None
        auth_cfg = self._cfg.auth

        # Use the new CORS middleware from auth module
        middlewares: list[Any] = [build_cors_middleware(cors_origins)]

        # Bearer token auth takes priority if configured
        if auth_cfg is not None and auth_cfg.auth_enabled:
            middlewares.append(build_bearer_auth_middleware(auth_cfg, self._failure_recorder))
        elif api_key is not None:
            # Legacy X-API-Key auth for backwards compatibility
            @web.middleware
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

                if not hmac.compare_digest(key, api_key):
                    raise web.HTTPUnauthorized(text="Invalid or missing API key")

                resp: web.StreamResponse = await handler(request)
                return resp

            middlewares.append(auth_middleware)

        return middlewares

    # ------------------------------------------------------------------
    # mDNS / Zeroconf
    # ------------------------------------------------------------------

    async def _register_mdns(self) -> None:
        """Register telemetry service via Zeroconf for LAN discovery.

        PR #4: set ``_mdns_registered_event`` whether the call succeeds
        or fails so ``start()`` can stop awaiting and continue startup
        on a bounded timeline. Outcomes are also reflected in:

        * ``telemetry_mdns_registered{service=...}`` Prometheus gauge.
        * ``FailureRecorder.record(subsystem="telemetry",
          reason="mdns_register_failed")`` on exceptions.
        """
        ok = False
        try:
            from zeroconf import ServiceInfo, Zeroconf

            # Resolved via the package namespace (not a direct name-import)
            # so that ``unittest.mock.patch("mousedroid.telemetry.server.
            # get_default_ip", ...)`` — and direct ``module.get_default_ip =
            # ...`` overrides — keep intercepting this call the same way
            # they did against the pre-split flat module. A plain
            # `from mousedroid.telemetry.network import get_default_ip`
            # would bind a private copy in THIS module's namespace that a
            # patch on ``mousedroid.telemetry.server`` cannot reach.
            from mousedroid.telemetry import server as _server_pkg

            ip = _server_pkg.get_default_ip()
            preferred_iface = self._cfg.preferred_interface
            if preferred_iface:
                preferred_ip = await _server_pkg.get_interface_ip(preferred_iface)
                if preferred_ip:
                    ip = preferred_ip
            import socket as sock

            packed_ip = sock.inet_aton(ip)

            self._service_info = ServiceInfo(
                type_=MDNS_SERVICE_TYPE,
                name=f"{self._cfg.mdns_service_name}.{MDNS_SERVICE_TYPE}",
                addresses=[packed_ip],
                port=self._bound_port,
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
            ok = True
            _log.info(
                "telemetry_mdns_registered",
                service_name=self._cfg.mdns_service_name,
                ip=ip,
                port=self._bound_port,
            )
        except ImportError:
            _log.warning("telemetry_mdns_zeroconf_not_installed")
        except Exception as exc:
            self._failure_recorder.record(
                "telemetry",
                "mdns_register_failed",
                level="warning",
                extra={"error": type(exc).__name__},
            )
            _log.warning("telemetry_mdns_registration_failed", error=str(exc))
        finally:
            self._mdns_ok = ok
            if self._metrics is not None:
                self._metrics.set_mdns_registered(self._cfg.mdns_service_name, ok=ok)
            self._mdns_registered_event.set()

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
