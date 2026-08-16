"""Structural self-type for cross-mixin calls within ``TelemetryServer``.

Three methods need to call (or reference as a route callback) members that
live on a *different* mixin than the one that defines them:

* ``_LifecycleMixin.start`` spawns ``_WebSocketHandlersMixin``'s broadcast
  loops.
* ``_LifecycleMixin._register_routes`` wires up nearly every REST handler
  (``_RestHandlersMixin``) and three WebSocket handlers
  (``_WebSocketHandlersMixin``) as aiohttp route callbacks.
* ``_RestHandlersMixin._handle_status`` reads the ``client_count`` property
  (``_LifecycleMixin``).

At runtime this is fine — all three mixins are composed onto one
``TelemetryServer`` instance, so ``self.<anything>`` resolves via the
normal MRO regardless of which file defines it. For ``mypy --strict``,
though, a method's inferred ``self`` type is the class it is *defined* on,
which does not know about sibling mixins.

The fix is a ``self: <this Protocol>`` parameter override on exactly those
three methods. A concrete forward-reference to the final ``TelemetryServer``
class does NOT work here — mypy rejects it with "the erased type of self
... is not a supertype of its class" for a self-type that is a *concrete*
class defined via multiple inheritance of the very mixin being overridden.
A structural ``Protocol`` sidesteps that restriction entirely (mypy checks
structural compatibility, not nominal supertype-of-self), which is why this
class exists as a Protocol rather than importing ``TelemetryServer`` here.

This class has NO runtime behaviour and is never instantiated — it exists
purely for the three ``self:`` annotations described above. Because a
``Protocol`` cannot inherit a concrete class (only other protocols), the
shared-state attributes are restated here rather than inherited from
``_state._TelemetryServerState`` — keep the two in sync.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import asyncio

    from aiohttp import web

    from mousedroid.common.rate_limit import TokenBucket
    from mousedroid.config.schema import OpenClawConfig, TelemetryConfig
    from mousedroid.hardware.protocols import RawFrameSourceProtocol
    from mousedroid.health.monitor import HealthMonitor
    from mousedroid.orchestrator.mission_dispatcher import MissionDispatcherProtocol
    from mousedroid.telemetry.failure_recorder import FailureRecorder
    from mousedroid.telemetry.log_buffer import LogRingBuffer
    from mousedroid.telemetry.metrics import MetricsRegistry
    from mousedroid.telemetry.protocol import (
        LidarRawScan,
        TelemetryFrame,
        TelemetryPublisherProtocol,
    )
    from mousedroid.telemetry.serialization import SerializationName


class _ComposedServerProtocol(Protocol):
    """Structural view of the fully-composed ``TelemetryServer``.

    See the module docstring. Restates ``_TelemetryServerState``'s
    attributes (structurally, not by inheritance — protocols may only
    inherit other protocols) plus the specific cross-mixin methods/
    properties that ``start``, ``_register_routes``, and ``_handle_status``
    need on a sibling mixin.
    """

    # -- shared instance state (mirrors _state._TelemetryServerState) ----
    _failure_recorder: FailureRecorder
    _cfg: TelemetryConfig
    _queue: asyncio.Queue[TelemetryFrame]
    _health_monitor: HealthMonitor
    _log_buffer: LogRingBuffer | None
    _metrics: MetricsRegistry | None
    _metrics_path: str
    _publisher: TelemetryPublisherProtocol | None
    _lidar_max_range_m: float | None
    _raw_frame_source: RawFrameSourceProtocol | None
    _raw_frame_interval_s: float
    _cloud_enabled: bool
    _reported_frame_drops: int
    _reported_lidar_raw_drops: int

    _mission_dispatcher: MissionDispatcherProtocol | None
    _openclaw_cfg: OpenClawConfig | None
    _mission_route_enabled: bool
    _mission_rate_limiter: TokenBucket | None
    _mission_dedup_window_s: float
    _mission_dedup: dict[str, tuple[float, dict[str, Any]]]
    _mission_inflight: dict[str, asyncio.Future[tuple[int, dict[str, Any]]]]

    _ws_clients: list[web.WebSocketResponse]
    _ws_serializations: dict[int, SerializationName]
    _ws_prepared: set[int]
    _lidar_ws_clients: list[web.WebSocketResponse]
    _lidar_ws_serializations: dict[int, SerializationName]
    _lidar_ws_prepared: set[int]
    _lidar_raw_queue: asyncio.Queue[LidarRawScan] | None
    _latest_frame: TelemetryFrame | None
    _latest_lidar_raw: LidarRawScan | None
    _running: bool
    _broadcast_task: asyncio.Task[None] | None
    _lidar_raw_broadcast_task: asyncio.Task[None] | None
    _background_tasks: set[asyncio.Task[Any]]
    _runner: web.AppRunner | None
    _zeroconf: Any
    _service_info: Any
    _mdns_registered_event: asyncio.Event
    _mdns_ok: bool
    _bound_port: int

    # -- _LifecycleMixin members needed from _RestHandlersMixin, and from
    # -- _LifecycleMixin's own `start` (once `self` is overridden for a
    # -- method, mypy resolves EVERY `self.foo` in that method's body
    # -- through the override type — including same-mixin methods) ------
    @property
    def client_count(self) -> int: ...
    def _build_middlewares(self) -> list[Any]: ...
    def _register_routes(self, app: web.Application) -> None: ...
    async def _register_mdns(self) -> None: ...

    # -- _RestHandlersMixin members needed from _LifecycleMixin ----------
    async def _handle_status(self, request: web.Request) -> web.Response: ...
    async def _handle_sensors(self, request: web.Request) -> web.Response: ...
    async def _handle_health(self, request: web.Request) -> web.Response: ...
    async def _handle_cloud_health(self, request: web.Request) -> web.Response: ...
    async def _handle_logs(self, request: web.Request) -> web.Response: ...
    async def _handle_network(self, request: web.Request) -> web.Response: ...
    async def _handle_mission_post(self, request: web.Request) -> web.Response: ...
    async def _handle_metrics(self, request: web.Request) -> web.Response: ...
    async def _handle_root(self, request: web.Request) -> web.StreamResponse: ...
    async def _handle_dashboard_page(self, request: web.Request) -> web.Response: ...
    async def _handle_lidar_page(self, request: web.Request) -> web.Response: ...
    async def _handle_camera_page(self, request: web.Request) -> web.Response: ...
    async def _handle_camera_frame(self, request: web.Request) -> web.Response: ...
    async def _handle_camera_stream(self, request: web.Request) -> web.StreamResponse: ...

    # -- _WebSocketHandlersMixin members needed from _LifecycleMixin -----
    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse: ...
    async def _handle_lidar_raw_ws(self, request: web.Request) -> web.WebSocketResponse: ...
    async def _handle_log_stream(self, request: web.Request) -> web.WebSocketResponse: ...
    async def _broadcast_loop(self) -> None: ...
    async def _lidar_raw_broadcast_loop(self) -> None: ...
