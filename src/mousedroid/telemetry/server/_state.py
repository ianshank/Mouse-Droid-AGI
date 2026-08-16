"""Shared instance-attribute declarations for the ``TelemetryServer`` mixins.

``TelemetryServer`` (defined in ``telemetry/server/__init__.py``) is composed
from three mixins living in sibling modules — ``_lifecycle.py``,
``_rest_handlers.py``, ``_ws_handlers.py`` — so the class keeps ONE flat
method namespace (every ``server.foo(...)`` call site is unaffected by the
split) while its implementation is spread across files. Because the mixins
live in different modules but operate on one shared instance's state
(``self._cfg``, ``self._ws_clients``, ...), every mixin inherits from this
class so ``mypy --strict`` can resolve ``self.<attr>`` references in each
mixin's own methods without redeclaring every attribute in every file.

This class carries NO runtime behaviour — only ``_LifecycleMixin.__init__``
(in ``_lifecycle.py``) ever assigns real values to these attributes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


class _TelemetryServerState:
    """Type-only declarations for ``TelemetryServer``'s shared instance state.

    See the module docstring: every attribute here is actually assigned in
    ``_LifecycleMixin.__init__``; this class exists purely so sibling mixins
    can reference ``self.<attr>`` with a resolvable, correctly-typed name.
    """

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
