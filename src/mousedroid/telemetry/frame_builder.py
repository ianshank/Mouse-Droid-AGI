"""Telemetry frame construction — decouples frame building from the orchestrator.

Centralises the observation-to-TelemetryFrame conversion so the orchestrator
doesn't need to know about frame field mapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mousedroid.constants import MOTOR_STATE_BATTERY_INDEX
from mousedroid.telemetry.protocol import TelemetryFrame

if TYPE_CHECKING:
    from mousedroid.safety.context import SafetyContext
    from mousedroid.sensing.protocol import ObservationProtocol


def build_telemetry_frame(
    observation: ObservationProtocol,
    safety_ctx: SafetyContext,
    loop_time_ms: float,
    tick_count: int,
    *,
    vision_feature_max_samples: int = 256,
) -> TelemetryFrame:
    """Build a ``TelemetryFrame`` from an observation and safety context.

    Args:
        observation: Current sensor observation bundle.
        safety_ctx: Current safety evaluation result.
        loop_time_ms: Control loop iteration time (milliseconds).
        tick_count: Monotonically increasing tick counter.
        vision_feature_max_samples: Upper bound on the number of vision
            samples encoded into ``TelemetryFrame.vision_features``.
            Sourced from ``TelemetryConfig.vision_feature_max_samples``.

    Returns:
        Fully-populated ``TelemetryFrame`` ready for publishing.
    """
    vision_arr = observation.vision_features
    vision_norm = float(np.sqrt(np.sum(vision_arr * vision_arr)))

    audio_arr = observation.audio_chunk
    audio_rms = float(np.sqrt(np.mean(audio_arr * audio_arr)))

    lidar_min_dist_m: float | None = None
    if safety_ctx.lidar_min_dist_m != float("inf"):
        lidar_min_dist_m = safety_ctx.lidar_min_dist_m

    lidar_sectors: list[float] | None = None
    lidar_features = observation.lidar_features
    if lidar_features is not None:
        lidar_sectors = lidar_features.astype(float).tolist()

    # ``lidar_n_points`` is an optional liveness attribute on concrete
    # observation bundles; fall back to ``0`` when the bundle doesn't
    # expose it (keeps the ObservationProtocol contract unchanged).
    lidar_n_points = int(getattr(observation, "lidar_n_points", 0))

    # Vision features are downsampled to a bounded payload for
    # bandwidth-friendly dashboard rendering as a heatmap. The cap is
    # supplied by the caller (see ``TelemetryConfig.vision_feature_max_samples``).
    # ``None`` when the vision modality is inactive.
    vision_features: list[float] | None = None
    if vision_arr is not None and vision_arr.size > 0:
        max_samples = max(1, vision_feature_max_samples)
        if vision_arr.size > max_samples:
            stride = vision_arr.size // max_samples
            vision_features = vision_arr[::stride][:max_samples].astype(float).tolist()
        else:
            vision_features = vision_arr.astype(float).tolist()

    motor = observation.motor_state
    battery_v = (
        float(motor[MOTOR_STATE_BATTERY_INDEX]) if motor.size > MOTOR_STATE_BATTERY_INDEX else 0.0
    )

    return TelemetryFrame(
        timestamp=observation.timestamp,
        distance_m=observation.distance_m,
        motor_state=motor.tolist(),
        vision_norm=vision_norm,
        audio_rms=audio_rms,
        valid_mask=observation.valid_mask.tolist(),
        battery_voltage=battery_v,
        safety={
            "is_emergency": safety_ctx.is_emergency,
            "violations": list(safety_ctx.law_violations),
            "forward_clearance_ok": safety_ctx.forward_clearance_ok,
            "lidar_clearance_ok": safety_ctx.lidar_clearance_ok,
        },
        lidar_min_dist_m=lidar_min_dist_m,
        lidar_sectors=lidar_sectors,
        lidar_n_points=lidar_n_points,
        vision_features=vision_features,
        loop_time_ms=loop_time_ms,
        tick_count=tick_count,
    )
