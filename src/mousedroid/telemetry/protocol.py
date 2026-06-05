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
        lidar_min_dist_m: Minimum LiDAR distance reading (metres), or
            ``None`` if LiDAR is unavailable.
        lidar_sectors: Per-sector normalised distances in ``[0.0, 1.0]``,
            where ``1.0`` = ``max_range_m`` (no obstacle) and ``0.0`` =
            at-sensor obstacle. Length equals ``cfg.lidar.n_sectors``.
            ``None`` when LiDAR is disabled or feature extraction failed.
        lidar_n_points: Number of raw points in the last LiDAR scan. ``0``
            when LiDAR is stale/absent; used as a liveness signal.
        vision_features: Optional list of vision feature values for the latest
            observation. ``None`` when vision features are not published (e.g.
            summary-only telemetry mode).
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
    lidar_sectors: list[float] | None = None
    lidar_n_points: int = 0
    vision_features: list[float] | None = None
    loop_time_ms: float = 0.0
    tick_count: int = 0
    # PR #4: per-sensor liveness map distinguishes disabled/awaiting/live/
    # stale so the dashboard can render three distinct UI states instead
    # of conflating "off" with "broken". Empty when no liveness tracker
    # is wired (preserves backwards-compat for direct constructions).
    sensor_liveness: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Sensor-fusion summary derived from the fused observation bundle's
    # ``valid_mask`` + per-modality scalar magnitudes, so the dashboard can
    # render one fusion panel instead of inferring it from raw fields. Shape::
    #
    #     {
    #       "n_valid": int,        # modalities currently contributing
    #       "n_modalities": int,   # mask length (4 without lidar, 5 with)
    #       "lidar_present": bool, # whether the lidar slot exists
    #       "modalities": {        # per-modality validity (lidar only when present)
    #         "vision": bool, "ultrasonic": bool, "motor": bool,
    #         "audio": bool, "lidar": bool,
    #       },
    #       "fused_norm": float,   # bounded L2 of the per-modality summary
    #     }
    #
    # Empty dict when no observation is available (preserves backwards-compat
    # for direct constructions — mirrors ``sensor_liveness``).
    fused: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON/msgpack encoding.

        Returns:
            Dictionary representation of all fields.
        """
        return asdict(self)


def lidar_scan_to_raw(scan: Any) -> LidarRawScan:
    """Convert a ``LidarScan`` from the driver into a ``LidarRawScan``.

    Converts angles from degrees to radians and distances from mm to
    metres. The input is typed loosely (``Any``) to avoid a hard
    dependency on the sensing layer; the function expects an object
    with ``angles_deg`` (NDArray-like in degrees), ``distances_mm``
    (NDArray-like in mm), ``confidences`` (NDArray-like), ``timestamp``,
    and ``n_points`` attributes.

    Args:
        scan: Source scan in the driver's native units.

    Returns:
        A :class:`LidarRawScan` in SI units (radians, metres) ready for
        WebSocket publishing.
    """
    import math as _math

    raw_angles = list(scan.angles_deg)
    raw_distances = list(scan.distances_mm)
    raw_confidences = list(getattr(scan, "confidences", []))
    angles_rad = [float(a) * _math.pi / 180.0 for a in raw_angles]
    distances_m = [float(d) / 1000.0 for d in raw_distances]
    intensities = [float(c) / 255.0 for c in raw_confidences] if raw_confidences else []
    return LidarRawScan(
        timestamp=float(scan.timestamp),
        angles_rad=angles_rad,
        distances_m=distances_m,
        n_points=int(scan.n_points),
        scan_duration_s=0.0,
        intensities=intensities,
    )


@dataclass(frozen=True)
class LidarRawScan:
    """Single raw LiDAR scan snapshot for the live streaming endpoint.

    Carries the decoded points (angle, distance) of a complete 360° scan
    plus minimal diagnostic metadata. Independent from
    :class:`TelemetryFrame` because the raw scan stream is published at
    a different rate and to a different WebSocket endpoint.

    Attributes:
        timestamp: Monotonic timestamp of scan completion (seconds).
        angles_rad: Polar angle of each point in radians, in ``[0, 2π)``.
        distances_m: Distance (metres) at each corresponding angle.
            Same length as ``angles_rad``.
        intensities: Optional per-point intensity / confidence values in
            ``[0, 1]``. Empty when the driver does not provide them.
        n_points: Total point count in the scan (``len(angles_rad)``).
        scan_duration_s: Time taken to assemble this scan (seconds).
    """

    timestamp: float
    angles_rad: list[float]
    distances_m: list[float]
    n_points: int
    scan_duration_s: float
    intensities: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON/msgpack encoding.

        Returns:
            Dictionary representation suitable for the
            ``/ws/v1/lidar/raw`` payload.
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

    async def publish_lidar_raw(self, scan: LidarRawScan) -> None:
        """Publish a raw LiDAR scan to the streaming queue.

        Non-blocking — drops the scan if the raw queue is full.
        Implementations may rate-limit based on
        ``TelemetryConfig.lidar_raw_publish_hz``.

        Args:
            scan: Raw scan snapshot to publish.
        """
        ...

    def get_queue(self) -> asyncio.Queue[TelemetryFrame]:
        """Return the internal queue for consumers.

        Returns:
            The ``asyncio.Queue`` that the server reads from.
        """
        ...

    def get_lidar_raw_queue(self) -> asyncio.Queue[LidarRawScan]:
        """Return the raw-LiDAR streaming queue for the server to consume.

        Returns:
            The ``asyncio.Queue`` carrying ``LidarRawScan`` snapshots.
        """
        ...

    @property
    def stats(self) -> dict[str, int]:
        """Publishing statistics.

        Returns:
            Dict with at least ``frames_published``, ``frames_dropped``,
            ``lidar_raw_published``, and ``lidar_raw_dropped`` counts.
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
