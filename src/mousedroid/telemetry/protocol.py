"""Telemetry protocol — data types and interfaces for remote monitoring.

Defines ``TelemetryFrame`` (immutable snapshot), ``TelemetryPublisherProtocol``
(orchestrator publishes to), and ``TelemetryServerProtocol`` (serves to clients).
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class TelemetryFrame:
    """Single telemetry snapshot published each control-loop cycle.

    All fields are plain Python types (no numpy) for direct JSON
    serialisation. NDArrays must be converted before constructing.

    Attributes:
        timestamp: Monotonic timestamp of the observation (seconds).
        distance_m: Forward ultrasonic distance (metres).
        motor_state: Motor state ``[vx, vy, omega, battery_v]``.
        vision_norm: L2 norm of the vision feature vector (scalar summary).
        audio_rms: Root-mean-square of the audio chunk (scalar summary).
        valid_mask: Per-modality validity flags.
        encoder: Encoder reading as a dict.
        battery_voltage: Battery voltage (volts).
        safety: Safety context (is_emergency, violations).
        health: Health metrics (gpu_temp, gpu_load, etc.).
        loop_time_ms: Control loop iteration time (milliseconds).
        tick_count: Monotonically increasing tick counter.
    """

    timestamp: float = 0.0
    distance_m: float = 0.0
    motor_state: list[float] = field(default_factory=list)
    vision_norm: float = 0.0
    audio_rms: float = 0.0
    valid_mask: list[float] = field(default_factory=list)
    encoder: dict[str, float] = field(default_factory=dict)
    battery_voltage: float = 0.0
    safety: dict[str, Any] = field(default_factory=dict)
    health: dict[str, Any] = field(default_factory=dict)
    lidar_min_dist_m: float | None = None
    loop_time_ms: float = 0.0
    tick_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON/msgpack encoding.

        Returns:
            Dictionary representation of all fields.
        """
        return asdict(self)


@runtime_checkable
class TelemetryPublisherProtocol(Protocol):
    """Interface for publishing telemetry from the orchestrator.

    Implementations must be non-blocking; frames are dropped when
    the internal queue is full rather than applying backpressure.
    """

    async def publish(self, frame: TelemetryFrame) -> None:
        """Publish a telemetry frame (non-blocking, drops if queue full).

        Args:
            frame: Telemetry snapshot to publish.
        """
        ...

    def get_queue(self) -> asyncio.Queue[TelemetryFrame]:
        """Return the internal queue for consumers.

        Returns:
            The ``asyncio.Queue`` that the server reads from.
        """
        ...

    @property
    def stats(self) -> dict[str, int]:
        """Publishing statistics.

        Returns:
            Dict with ``frames_published`` and ``frames_dropped`` counts.
        """
        ...


@runtime_checkable
class TelemetryServerProtocol(Protocol):
    """Interface for the telemetry web server."""

    async def start(self) -> None:
        """Start the server (bind port, begin accepting connections)."""
        ...

    async def stop(self) -> None:
        """Gracefully shut down the server."""
        ...

    @property
    def client_count(self) -> int:
        """Number of currently connected WebSocket clients."""
        ...

    @property
    def is_running(self) -> bool:
        """Whether the server is currently running."""
        ...
