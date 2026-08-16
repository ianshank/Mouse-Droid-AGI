"""Telemetry server — aiohttp-based REST + WebSocket for remote monitoring.

Provides real-time sensor data streaming, health metrics, log retrieval,
and network interface information over WiFi and Ethernet connections.

All endpoints are async and run on the same event loop as the main
orchestrator. The server consumes ``TelemetryFrame`` objects from a
publisher queue and fans them out to connected WebSocket clients.

``TelemetryServer`` is composed from three mixins living in sibling
private modules — ``_lifecycle.py`` (construction, start/stop, routing,
mDNS), ``_rest_handlers.py`` (the 16 REST handlers), and
``_ws_handlers.py`` (the 10 WebSocket handlers + broadcast loops) — so the
class keeps ONE flat method namespace: every ``server.foo(...)`` call site
is unaffected by the split. ``MissionRequest`` lives in ``_models.py`` — a
plain Pydantic model with no server dependencies.

The re-exports below beyond ``TelemetryServer``/``MissionRequest``
preserve the pre-split flat module's import surface and ``dir()`` output.
``get_default_ip`` / ``get_interface_ip`` / ``get_network_interfaces`` are
load-bearing, not just cosmetic: ``_rest_handlers.py`` and ``_lifecycle.py``
resolve them dynamically through *this* package's namespace at call time
(``from mousedroid.telemetry import server as _server_pkg; _server_pkg.
get_default_ip()``) precisely so that
``unittest.mock.patch("mousedroid.telemetry.server.get_default_ip", ...)``
keeps intercepting those calls the same way it did against the pre-split
flat module.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, ValidationError

from mousedroid.common.async_utils import cancel_and_drain, spawn_tracked
from mousedroid.common.rate_limit import TokenBucket
from mousedroid.constants import (
    MAX_LOG_ENTRIES,
    MDNS_SERVICE_TYPE,
    TELEMETRY_QUEUE_TIMEOUT_S,
    WS_CLOSE_LIDAR_RAW_UNAVAILABLE,
    WS_CLOSE_LOG_BUFFER_DISABLED,
    WS_CLOSE_MAX_CLIENTS,
    WS_CLOSE_NEGOTIATION_FAILED,
    WS_HELLO_MAX_BYTES,
)
from mousedroid.logging.setup import get_logger
from mousedroid.orchestrator.mission_dispatcher import GATE_REJECTION_PREFIX
from mousedroid.security.injection_filter import InjectionRejected
from mousedroid.telemetry.network import (
    get_default_ip,
    get_interface_ip,
    get_network_interfaces,
)
from mousedroid.telemetry.protocol import LidarRawScan, TelemetryFrame
from mousedroid.telemetry.serialization import (
    SerializationName,
    build_default_ack,
    negotiate,
)
from mousedroid.telemetry.server._lifecycle import _LifecycleMixin
from mousedroid.telemetry.server._models import MissionRequest
from mousedroid.telemetry.server._rest_handlers import _RestHandlersMixin
from mousedroid.telemetry.server._ws_handlers import _WebSocketHandlersMixin

__all__ = [
    "GATE_REJECTION_PREFIX",
    "MAX_LOG_ENTRIES",
    "MDNS_SERVICE_TYPE",
    "TELEMETRY_QUEUE_TIMEOUT_S",
    "TYPE_CHECKING",
    "WS_CLOSE_LIDAR_RAW_UNAVAILABLE",
    "WS_CLOSE_LOG_BUFFER_DISABLED",
    "WS_CLOSE_MAX_CLIENTS",
    "WS_CLOSE_NEGOTIATION_FAILED",
    "WS_HELLO_MAX_BYTES",
    "Any",
    "BaseModel",
    "Field",
    "InjectionRejected",
    "LidarRawScan",
    "Literal",
    "MissionRequest",
    "SerializationName",
    "TelemetryFrame",
    "TelemetryServer",
    "TokenBucket",
    "ValidationError",
    "asyncio",
    "build_default_ack",
    "cancel_and_drain",
    "contextlib",
    "get_default_ip",
    "get_interface_ip",
    "get_logger",
    "get_network_interfaces",
    "json",
    "negotiate",
    "spawn_tracked",
    "time",
]


class TelemetryServer(_LifecycleMixin, _RestHandlersMixin, _WebSocketHandlersMixin):
    """aiohttp-based telemetry server for remote monitoring.

    Provides REST endpoints for snapshots and WebSocket endpoints for
    real-time streaming. Supports optional API key authentication,
    CORS, mDNS/Zeroconf service registration, and configurable
    serialisation (JSON or msgpack).

    Implements ``TelemetryServerProtocol``.
    """
