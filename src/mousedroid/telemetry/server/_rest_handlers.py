"""REST handler mixin for ``TelemetryServer``.

Holds the 16 REST endpoint handlers (status/sensors/health/logs/network,
the OpenClaw mission endpoint, metrics scrape, and the static
dashboard/lidar/camera pages + camera capture endpoints). Split out of the
former monolithic ``telemetry/server.py``; see ``telemetry/server/__init__.py``
for how this mixin is composed with ``_LifecycleMixin`` and
``_WebSocketHandlersMixin`` into the final ``TelemetryServer`` class.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from mousedroid.constants import MAX_LOG_ENTRIES
from mousedroid.logging.setup import get_logger
from mousedroid.orchestrator.mission_dispatcher import GATE_REJECTION_PREFIX
from mousedroid.security.injection_filter import InjectionRejected
from mousedroid.telemetry.server._models import MissionRequest
from mousedroid.telemetry.server._state import _TelemetryServerState

if TYPE_CHECKING:
    from aiohttp import web

    from mousedroid.orchestrator.mission_dispatcher import MissionDispatcherProtocol
    from mousedroid.telemetry.server._protocol import _ComposedServerProtocol

_log = get_logger(__name__)

_STARTUP_TIME: float = time.monotonic()


class _RestHandlersMixin(_TelemetryServerState):
    """The 16 REST endpoint handlers (snapshots, mission ingress, pages)."""

    async def _handle_status(self: _ComposedServerProtocol, request: web.Request) -> web.Response:
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

        # Resolved via the package namespace (not a direct name-import) so
        # that ``unittest.mock.patch("mousedroid.telemetry.server.
        # get_network_interfaces", ...)`` keeps intercepting this call the
        # same way it did against the pre-split flat module. A plain
        # `from mousedroid.telemetry.network import get_network_interfaces`
        # would bind a private copy in THIS module's namespace that a patch
        # on ``mousedroid.telemetry.server`` cannot reach.
        from mousedroid.telemetry import server as _server_pkg

        interfaces = await _server_pkg.get_network_interfaces()
        default_ip = _server_pkg.get_default_ip()
        preferred_iface = self._cfg.preferred_interface
        if preferred_iface:
            preferred_ip = await _server_pkg.get_interface_ip(preferred_iface)
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

        if self._mission_rate_limiter is not None:
            taken, retry_after = await self._mission_rate_limiter.take()
            if not taken:
                log.warning(
                    "mission_endpoint_rejected",
                    reason="rate_limited",
                    retry_after_s=retry_after,
                )
                return web.json_response(
                    {"error": "rate_limited", "retry_after_s": round(retry_after, 3)},
                    status=429,
                )

        parsed = await self._parse_mission_request(request, log)
        if isinstance(parsed, web.Response):
            return parsed
        req = parsed

        # Idempotency: two-phase dedup so concurrent retries with the
        # same key never start parallel dispatches.
        # - ``_mission_dedup`` caches the leader's *successful* (202)
        #   body for ``command_dedup_window_s`` so later replays return
        #   the cached body without touching the dispatcher.
        # - ``_mission_inflight`` carries the leader's in-progress
        #   future; followers ``await`` it so they see the same
        #   outcome (success or error) without a parallel dispatch.
        now = time.monotonic()
        if self._mission_dedup:
            self._mission_dedup = {k: v for k, v in self._mission_dedup.items() if v[0] > now}
        key = req.idempotency_key
        if key is not None and key in self._mission_dedup:
            cached = self._mission_dedup[key][1]
            log.info("mission_endpoint_dedup_hit", idempotency_key=key, kind="cached")
            return web.json_response(cached, status=202)
        if key is not None and key in self._mission_inflight:
            log.info("mission_endpoint_dedup_hit", idempotency_key=key, kind="inflight")
            try:
                leader_status, leader_body = await self._mission_inflight[key]
            except asyncio.CancelledError:
                # Cooperative cancellation of THIS follower request must
                # propagate — swallowing it here would convert a server
                # shutdown into a bogus 500 for the client.
                raise
            except Exception:
                # Leader exited abnormally; surface a 500 so the
                # follower can retry instead of seeing a stale
                # in-flight future.
                return web.json_response({"error": "internal_error"}, status=500)
            return web.json_response(leader_body, status=leader_status)

        # Reserve the in-flight slot BEFORE awaiting dispatch, so any
        # concurrent retry with the same key sees the future and waits.
        leader_future: asyncio.Future[tuple[int, dict[str, Any]]] | None = None
        if key is not None:
            leader_future = asyncio.get_running_loop().create_future()
            self._mission_inflight[key] = leader_future

        log.info(
            "mission_endpoint_received",
            length=len(req.nl_command),
            has_idempotency_key=key is not None,
        )

        try:
            status, body = await self._dispatch_mission_command(
                self._mission_dispatcher, req, peer, log
            )

            # Cache only successful 202s; transient 5xx must be
            # retryable with the same key. Client errors (400, 504)
            # propagate to in-flight followers via the future but are
            # not persisted — the request body that produced them was
            # rejected for cause and a retry deserves a fresh check.
            if key is not None and status == 202:
                self._mission_dedup[key] = (now + self._mission_dedup_window_s, body)
            if leader_future is not None:
                leader_future.set_result((status, body))
            return web.json_response(body, status=status)
        finally:
            if key is not None:
                self._mission_inflight.pop(key, None)
                # Defensive: if we somehow exited without resolving the
                # future, signal followers so they don't hang.
                if leader_future is not None and not leader_future.done():
                    leader_future.set_exception(
                        RuntimeError("mission leader exited without setting a result")
                    )

    async def _parse_mission_request(
        self, request: web.Request, log: Any
    ) -> MissionRequest | web.Response:
        """Parse and schema-validate the mission POST body.

        Returns the validated :class:`MissionRequest`, or a ready-to-send 400
        JSON response when the body is not valid JSON or fails validation. The
        caller distinguishes the two via ``isinstance(..., web.Response)``.
        """
        from aiohttp import web

        try:
            payload = await request.json()
        except (json.JSONDecodeError, ValueError):
            log.warning("mission_endpoint_rejected", reason="invalid_json")
            return web.json_response({"error": "invalid_json"}, status=400)

        try:
            return MissionRequest.model_validate(payload)
        except ValidationError as exc:
            log.warning("mission_endpoint_rejected", reason="invalid_body")
            return web.json_response(
                {"error": "invalid_body", "details": exc.errors()},
                status=400,
            )

    async def _dispatch_mission_command(
        self,
        dispatcher: MissionDispatcherProtocol,
        req: MissionRequest,
        peer: str,
        log: Any,
    ) -> tuple[int, dict[str, Any]]:
        """Dispatch a validated NL command, mapping the outcome to (status, body).

        ``channel`` is hard-coded to ``"rest"`` (NOT taken from ``req.channel``)
        so a client cannot smuggle a different channel past the dispatcher's
        ``allowed_channels`` gate — defence-in-depth alongside the schema's
        ``Literal["rest"]`` constraint. The dispatcher is passed in (already
        non-``None``-checked by the caller) rather than read off ``self`` so no
        ``-O``-stripped assert is needed to narrow the type.
        """
        try:
            result = await dispatcher.dispatch(
                req.nl_command,
                channel="rest",
                peer=peer,
            )
        except InjectionRejected:
            # Still reachable when a filter rejects directly. OpenClawSafetyGate
            # converts injection hits into a rejected ApprovalDecision instead,
            # which arrives via the ValueError branch below.
            return 400, {"error": "invalid_command", "reason": "injection_pattern"}
        except ValueError as exc:
            # dispatch() re-raises gate rejections as "<prefix><slug>". Strip the
            # prefix so the body keeps exposing the stable machine-readable slug
            # rather than leaking dispatcher-internal phrasing to the client.
            reason = str(exc).removeprefix(GATE_REJECTION_PREFIX)
            return 400, {"error": "invalid_command", "reason": reason}
        except (TimeoutError, asyncio.TimeoutError):
            # asyncio.TimeoutError aliases the builtin TimeoutError on 3.11+ but
            # is a DISTINCT exception on 3.10 (a supported CI leg). Catch both so
            # a dispatcher timeout maps to 504, not the generic 500 below —
            # matching the dual-catch in orchestrator._maybe_fire_startup_greeting.
            log.warning("mission_endpoint_timeout")
            return 504, {"error": "timeout"}
        except Exception as exc:
            log.warning("mission_endpoint_failed", error=f"{type(exc).__name__}:{exc}")
            return 500, {"error": "internal_error"}

        body = {
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
        log.info(
            "mission_endpoint_dispatched",
            trace_id=result.trace_id,
            latency_ms=result.latency_ms,
        )
        return 202, body

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

    async def _handle_root(self, request: web.Request) -> web.StreamResponse:
        """GET / — redirect to the unified dashboard, preserving the token query.

        Other LAN devices open ``http://<rover>:8080/`` (optionally with
        ``?token=``); we 302 to ``/dashboard`` carrying the query string so the
        bearer token survives the hop.
        """
        from aiohttp import web

        target = "/dashboard"
        if request.query_string:
            target = f"{target}?{request.query_string}"
        raise web.HTTPFound(target)

    async def _handle_dashboard_page(self, request: web.Request) -> web.Response:
        """GET /dashboard — serve the unified overview page.

        Single page that embeds the live camera MJPEG, the lidar polar plot,
        and a sensor-fusion panel (per-modality liveness + the ``fused``
        summary) by subscribing to ``/ws`` + ``/ws/v1/lidar/raw`` + the camera
        stream. Behind the same bearer-auth middleware as ``/lidar`` /
        ``/camera``; the page carries the token via the ``?token=`` query.
        """
        from importlib import resources

        from aiohttp import web

        try:
            html = (
                resources.files("mousedroid.telemetry.static")
                .joinpath("dashboard.html")
                .read_text(encoding="utf-8")
            )
        except (FileNotFoundError, ModuleNotFoundError):
            return web.Response(status=404, text="dashboard_page_missing")

        return web.Response(
            body=html.encode("utf-8"),
            headers={
                "Content-Type": "text/html; charset=utf-8",
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-cache",
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
        except Exception as exc:
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
                except Exception as exc:
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
