"""WebSocket handler mixin for ``TelemetryServer``.

Holds the 10 WebSocket-facing methods: the main telemetry stream, hello
negotiation, raw-LiDAR stream, live log stream, the broadcast loops that
fan frames/scans out to connected clients, the metrics-sync helpers those
loops call, and the per-client send helpers. Split out of the former
monolithic ``telemetry/server.py``; see ``telemetry/server/__init__.py``
for how this mixin is composed with ``_LifecycleMixin`` and
``_RestHandlersMixin`` into the final ``TelemetryServer`` class.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING, Any

from mousedroid.constants import (
    TELEMETRY_QUEUE_TIMEOUT_S,
    WS_CLOSE_LIDAR_RAW_UNAVAILABLE,
    WS_CLOSE_LOG_BUFFER_DISABLED,
    WS_CLOSE_MAX_CLIENTS,
    WS_CLOSE_NEGOTIATION_FAILED,
    WS_HELLO_MAX_BYTES,
)
from mousedroid.logging.setup import get_logger
from mousedroid.telemetry.protocol import TelemetryFrame
from mousedroid.telemetry.serialization import (
    SerializationName,
    build_default_ack,
    negotiate,
)
from mousedroid.telemetry.server._state import _TelemetryServerState

if TYPE_CHECKING:
    from aiohttp import web

_log = get_logger(__name__)


class _WebSocketHandlersMixin(_TelemetryServerState):
    """The 10 WebSocket handlers, broadcast loops, and send helpers."""

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket handler — stream TelemetryFrames to connected clients.

        Optionally honours a single ``hello`` negotiation message at
        connect time. If the client sends one within
        ``ws_handshake_timeout_s`` the chosen serialization overrides
        ``TelemetryConfig.serialization`` for that connection only.
        Otherwise the server falls back to the configured serialization
        — keeping legacy clients working byte-identically.

        Args:
            request: The incoming WebSocket upgrade request.

        Returns:
            The WebSocket response (kept alive until client disconnects).
        """
        from aiohttp import WSMsgType, web

        # Construct ws synchronously so the check + reservation below is
        # atomic with respect to other concurrent connects (no awaits in
        # between). Addresses Gemini medium review comment_id=3238374802
        # — previously ``await ws.prepare(request)`` yielded between
        # the count check and the append, letting two clients slip past
        # the ``max_clients`` limit.
        ws = web.WebSocketResponse()
        # --- synchronous critical section -------------------------------
        if len(self._ws_clients) >= self._cfg.max_clients:
            await ws.prepare(request)
            await ws.close(code=WS_CLOSE_MAX_CLIENTS, message=b"max_clients_reached")
            return ws
        # API key auth is enforced by auth_middleware when cfg.api_key is set.
        # Register before any await — same rationale as above: keeps
        # max_clients accounting atomic AND lets legacy tests that
        # assert ``server.client_count == 1`` immediately after
        # ``ws_connect`` keep passing. The broadcast loop checks
        # ``_ws_prepared`` (populated AFTER ``ws.prepare`` returns)
        # so frames produced inside this critical section are NOT
        # sent to an unprepared WebSocketResponse.
        self._ws_clients.append(ws)
        # --- end critical section ---------------------------------------
        peer = request.remote or "unknown"

        try:
            await ws.prepare(request)
            self._ws_prepared.add(id(ws))
            _log.info("telemetry_ws_client_connected", peer=peer, total=len(self._ws_clients))
            chosen = await self._negotiate_ws(ws)
            if ws.closed:
                # ``_negotiate_ws`` closes on negotiation failure per the
                # NegotiationResult contract. Skip the message loop.
                return ws
            self._ws_serializations[id(ws)] = chosen
            _log.info(
                "telemetry_ws_serialization_set",
                peer=peer,
                serialization=chosen,
            )
            async for msg in ws:
                if msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            self._ws_prepared.discard(id(ws))
            if ws in self._ws_clients:
                self._ws_clients.remove(ws)
            self._ws_serializations.pop(id(ws), None)
            _log.info("telemetry_ws_client_disconnected", peer=peer, total=len(self._ws_clients))

        return ws

    async def _negotiate_ws(self, ws: web.WebSocketResponse) -> SerializationName:
        """Read an optional client hello and pick a serialization.

        Listens for at most one message within
        ``TelemetryConfig.ws_handshake_timeout_s``. The message must be
        a JSON-encoded ``hello`` object (see :mod:`telemetry.serialization`).

        Soft failure modes — timeout, non-TEXT first message, oversized
        payload, unparseable JSON — fall back to the server-configured
        serialization. These are legitimately legacy clients or
        misbehaving peers and should still receive frames.

        Hard failure modes — protocol version mismatch or no
        serialization overlap — honour the ``NegotiationResult``
        contract: the server sends the ``hello_ack`` envelope with
        ``ok=False`` and the failure reason, then closes the WebSocket
        with ``WS_CLOSE_NEGOTIATION_FAILED``. Callers should check
        ``ws.closed`` after this method returns and skip the read loop.

        Args:
            ws: The accepted WebSocket response.

        Returns:
            The serialization to use for outbound frames on this
            connection. Always populated; meaningless when ``ws.closed``.
        """
        from aiohttp import WSMsgType

        server_choice: SerializationName = (
            "msgpack" if self._cfg.serialization == "msgpack" else "json"
        )
        try:
            raw = await asyncio.wait_for(
                ws.receive(),
                timeout=self._cfg.ws_handshake_timeout_s,
            )
        except asyncio.TimeoutError:
            # No hello within window — keep server default. This is the
            # normal path for legacy dashboards that never negotiate.
            ack = build_default_ack(
                serialization=server_choice,
                protocol_version=self._cfg.ws_protocol_version,
            )
            with contextlib.suppress(ConnectionResetError, RuntimeError):
                await ws.send_json(ack)
            return server_choice

        if raw.type != WSMsgType.TEXT:
            # Binary, CLOSE, or ERROR before hello — fall back silently.
            # Most non-text first messages mean the peer doesn't speak
            # our protocol or already disconnected mid-handshake.
            _log.info(
                "telemetry_ws_negotiation_skipped",
                msg_type=int(raw.type),
                fallback_serialization=server_choice,
            )
            return server_choice

        # Reject oversized hellos to keep ``_negotiate_ws`` bounded in
        # memory (aiohttp's max_msg_size only caps the global ceiling).
        data = raw.data if isinstance(raw.data, str) else ""
        if len(data.encode("utf-8")) > WS_HELLO_MAX_BYTES:
            self._failure_recorder.record(
                "telemetry",
                "ws_negotiation_oversized",
                level="warning",
                extra={"limit_bytes": WS_HELLO_MAX_BYTES, "received_bytes": len(data)},
            )
            _log.warning(
                "telemetry_ws_negotiation_oversized",
                limit_bytes=WS_HELLO_MAX_BYTES,
                received_bytes=len(data),
            )
            return server_choice

        try:
            payload = json.loads(data)
        except (TypeError, ValueError):
            _log.warning(
                "telemetry_ws_negotiation_failed",
                reason="invalid_hello",
                detail="malformed JSON",
            )
            return server_choice

        result = negotiate(
            payload,
            server_serialization=server_choice,
            server_protocol_version=self._cfg.ws_protocol_version,
            msgpack_client_lib_url=self._cfg.msgpack_client_lib_url,
        )
        with contextlib.suppress(ConnectionResetError, RuntimeError):
            await ws.send_json(result.ack_payload)
        if not result.ok:
            # Honour the NegotiationResult contract: hard failures
            # (no overlap / unsupported version) MUST close the
            # WebSocket. The ack envelope was already sent so the
            # client sees the reason; the close code disambiguates
            # from network-level failures.
            self._failure_recorder.record(
                "telemetry",
                "ws_negotiation_failed",
                level="warning",
                extra={"reason": result.reason or "unknown"},
            )
            with contextlib.suppress(ConnectionResetError, RuntimeError):
                await ws.close(
                    code=WS_CLOSE_NEGOTIATION_FAILED,
                    message=(result.reason or "negotiation_failed").encode("utf-8"),
                )
        return result.serialization

    async def _handle_lidar_raw_ws(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket handler — stream raw LiDAR scans.

        Returns HTTP-style WebSocket close code ``4404`` when the
        publisher did not wire a raw queue (so this endpoint is
        unavailable for the current deployment).

        Args:
            request: The incoming WebSocket upgrade request.

        Returns:
            The WebSocket response.
        """
        from aiohttp import WSMsgType, web

        # Construct ws synchronously so the count check + reservation
        # below is atomic. Mirrors ``_handle_ws``; addresses Gemini
        # medium review comment_id=3238374802.
        ws = web.WebSocketResponse()
        # --- synchronous critical section -------------------------------
        if self._lidar_raw_queue is None:
            await ws.prepare(request)
            await ws.close(
                code=WS_CLOSE_LIDAR_RAW_UNAVAILABLE,
                message=b"lidar_raw_unavailable",
            )
            return ws
        if len(self._lidar_ws_clients) >= self._cfg.max_clients:
            await ws.prepare(request)
            await ws.close(code=WS_CLOSE_MAX_CLIENTS, message=b"max_clients_reached")
            return ws
        self._lidar_ws_clients.append(ws)
        # --- end critical section ---------------------------------------
        peer = request.remote or "unknown"

        try:
            await ws.prepare(request)
            self._lidar_ws_prepared.add(id(ws))
            _log.info(
                "telemetry_lidar_ws_client_connected",
                peer=peer,
                total=len(self._lidar_ws_clients),
            )
            chosen = await self._negotiate_ws(ws)
            if ws.closed:
                # Negotiation closed the WS — skip the read loop.
                return ws
            self._lidar_ws_serializations[id(ws)] = chosen
            async for msg in ws:
                if msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            self._lidar_ws_prepared.discard(id(ws))
            if ws in self._lidar_ws_clients:
                self._lidar_ws_clients.remove(ws)
            self._lidar_ws_serializations.pop(id(ws), None)
            _log.info(
                "telemetry_lidar_ws_client_disconnected",
                peer=peer,
                total=len(self._lidar_ws_clients),
            )
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
            await resp.close(code=WS_CLOSE_LOG_BUFFER_DISABLED, message=b"log_buffer_disabled")
            return resp

        # Auth is enforced centrally by the middleware stack built in
        # _lifecycle.py (bearer or legacy X-API-Key) before any handler runs;
        # this route needs no additional per-handler check.
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

            self._push_frame_metrics(frame)

            dead_clients: list[web.WebSocketResponse] = []
            send_tasks = []

            for ws in self._ws_clients:
                if ws.closed:
                    dead_clients.append(ws)
                    continue
                # PR #81: skip clients that are in the registry (for
                # ``max_clients`` accounting) but whose ``ws.prepare()``
                # has not completed yet. Calling ``send_*`` on an
                # unprepared WebSocketResponse raises ``RuntimeError``
                # which would be silently swallowed by
                # ``gather(..., return_exceptions=True)``.
                if id(ws) not in self._ws_prepared:
                    continue
                serialization = self._ws_serializations.get(
                    id(ws),
                    "msgpack" if self._cfg.serialization == "msgpack" else "json",
                )
                if serialization == "msgpack":
                    send_tasks.append(self._send_msgpack(ws, data))
                else:
                    send_tasks.append(self._send_json(ws, data))

            if send_tasks:
                await asyncio.gather(*send_tasks, return_exceptions=True)

            for ws in dead_clients:
                if ws in self._ws_clients:
                    self._ws_clients.remove(ws)
                self._ws_serializations.pop(id(ws), None)

    def _push_frame_metrics(self, frame: TelemetryFrame) -> None:
        """Push live telemetry from a frame into the metrics registry.

        Non-blocking and a no-op when no registry is wired. Mirrors the
        per-field guards the broadcast loop previously inlined (health/safety
        dicts, optional LiDAR fields, per-sensor liveness). A local ``metrics``
        binding after the ``None`` check keeps the type narrowed cleanly.
        """
        metrics = self._metrics
        if metrics is None:
            return
        metrics.set_loop_time_ms(frame.loop_time_ms)
        metrics.set_battery_voltage(frame.battery_voltage)
        metrics.set_ws_client_count(len(self._ws_clients))
        self._sync_publisher_metrics()
        health = frame.health
        if isinstance(health, dict):
            gpu_temp = health.get("gpu_temp_c")
            if isinstance(gpu_temp, int | float):
                metrics.set_gpu_temp_celsius(float(gpu_temp))
        safety = frame.safety
        if isinstance(safety, dict):
            for law in safety.get("violations", []):
                metrics.inc_safety_violation(str(law))
        lidar_enabled = self._lidar_max_range_m is not None or frame.lidar_sectors is not None
        if lidar_enabled:
            if frame.lidar_sectors is not None and self._lidar_max_range_m is not None:
                metrics.set_lidar_sectors(frame.lidar_sectors, self._lidar_max_range_m)
            if frame.lidar_min_dist_m is not None:
                metrics.set_lidar_min_distance_m(frame.lidar_min_dist_m)
            metrics.set_lidar_scan_points(frame.lidar_n_points)

        # PR #4: surface per-sensor liveness via the live gauge so operators
        # can alert on stale sensors directly from Prometheus.
        if frame.sensor_liveness:
            liveness_states = {
                sensor: str(payload.get("state", "awaiting"))
                for sensor, payload in frame.sensor_liveness.items()
            }
            metrics.set_sensor_liveness(liveness_states)

    async def _lidar_raw_broadcast_loop(self) -> None:
        """Fan-out raw LiDAR scans to ``/ws/v1/lidar/raw`` clients.

        Runs only when the publisher provided a raw queue (otherwise the
        task is not started). Mirrors the main broadcast loop's
        per-client serialization, dead-client pruning, and metrics
        updates — but for the raw scan stream.
        """
        if self._lidar_raw_queue is None:
            return
        _log.info(
            "telemetry_lidar_raw_broadcast_started",
            queue_size=self._lidar_raw_queue.maxsize,
        )
        while self._running:
            try:
                scan = await asyncio.wait_for(
                    self._lidar_raw_queue.get(),
                    timeout=TELEMETRY_QUEUE_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                continue

            self._latest_lidar_raw = scan
            data = scan.to_dict()
            if self._metrics is not None:
                self._metrics.inc_lidar_raw_published()

            dead_clients: list[web.WebSocketResponse] = []
            send_tasks = []
            for ws in self._lidar_ws_clients:
                if ws.closed:
                    dead_clients.append(ws)
                    continue
                # PR #81: same prepared-gate as ``_broadcast_loop``.
                if id(ws) not in self._lidar_ws_prepared:
                    continue
                serialization = self._lidar_ws_serializations.get(
                    id(ws),
                    "msgpack" if self._cfg.serialization == "msgpack" else "json",
                )
                if serialization == "msgpack":
                    send_tasks.append(self._send_msgpack(ws, data))
                else:
                    send_tasks.append(self._send_json(ws, data))
            if send_tasks:
                await asyncio.gather(*send_tasks, return_exceptions=True)
            for ws in dead_clients:
                if ws in self._lidar_ws_clients:
                    self._lidar_ws_clients.remove(ws)
                self._lidar_ws_serializations.pop(id(ws), None)

            # Publisher drop counter -> metrics (mirrors publisher.stats).
            if self._metrics is not None and self._publisher is not None:
                dropped_total = self._publisher.stats.get("lidar_raw_dropped", 0)
                delta = dropped_total - self._reported_lidar_raw_drops
                if delta > 0:
                    self._metrics.inc_lidar_raw_dropped(delta)
                    self._reported_lidar_raw_drops = dropped_total

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
