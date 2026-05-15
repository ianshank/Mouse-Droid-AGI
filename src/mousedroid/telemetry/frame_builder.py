"""Telemetry frame construction — decouples frame building from the orchestrator.

Centralises the observation-to-TelemetryFrame conversion so the orchestrator
doesn't need to know about frame field mapping.

PR #4 introduces an optional :class:`SensorLivenessTracker` parameter that
attaches a per-sensor liveness map (``disabled`` / ``awaiting`` / ``live`` /
``stale``) to every frame. This replaces the previous "0 = either disabled
or broken" silent fallback for lidar/vision data, giving dashboards three
distinct UI states to render.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mousedroid.constants import MOTOR_STATE_BATTERY_INDEX
from mousedroid.telemetry.protocol import TelemetryFrame

if TYPE_CHECKING:
    from mousedroid.safety.context import SafetyContext
    from mousedroid.sensing.protocol import ObservationProtocol
    from mousedroid.telemetry.sensor_liveness import SensorLivenessTracker


def build_telemetry_frame(
    observation: ObservationProtocol,
    safety_ctx: SafetyContext,
    loop_time_ms: float,
    tick_count: int,
    *,
    vision_feature_max_samples: int = 256,
    liveness_tracker: SensorLivenessTracker | None = None,
    now_s: float | None = None,
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
        liveness_tracker: Optional :class:`SensorLivenessTracker`. When
            provided, the builder records observations and attaches the
            resulting state map to the frame. ``None`` preserves the
            pre-PR-#4 behaviour (empty liveness dict).
        now_s: Monotonic timestamp to feed the liveness tracker. When
            ``None``, ``observation.timestamp`` is used so tests and
            replay flows remain deterministic.

    Returns:
        Fully-populated ``TelemetryFrame`` ready for publishing.
    """
    vision_arr = observation.vision_features
    vision_norm = float(np.sqrt(np.sum(vision_arr * vision_arr))) if vision_arr.size > 0 else 0.0

    audio_arr = observation.audio_chunk
    # Empty audio chunks happen in mock-mode bring-up; ``np.mean`` of an
    # empty array warns and returns NaN, so short-circuit here.
    audio_rms = float(np.sqrt(np.mean(audio_arr * audio_arr))) if audio_arr.size > 0 else 0.0

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
    # The downstream three-state ``sensor_liveness`` map distinguishes
    # "lidar disabled" from "lidar enabled but no points yet" so the
    # dashboard can render the difference.
    lidar_n_points = int(getattr(observation, "lidar_n_points", 0))

    # Vision features are downsampled to a bounded payload for
    # bandwidth-friendly dashboard rendering as a heatmap. The cap is
    # supplied by the caller (see ``TelemetryConfig.vision_feature_max_samples``).
    # ``None`` when the vision modality is inactive.
    vision_features: list[float] | None = None
    if vision_arr is not None and vision_arr.size > 0:
        max_samples = max(1, vision_feature_max_samples)
        if vision_arr.size > max_samples:
            # Uniformly-spaced indices across the full vector so we don't
            # systematically drop the tail when size is only slightly > max.
            idx = np.linspace(0, vision_arr.size - 1, max_samples).astype(np.int64)
            vision_features = vision_arr[idx].astype(float).tolist()
        else:
            vision_features = vision_arr.astype(float).tolist()

    motor = observation.motor_state
    battery_v = (
        float(motor[MOTOR_STATE_BATTERY_INDEX]) if motor.size > MOTOR_STATE_BATTERY_INDEX else 0.0
    )

    sensor_liveness: dict[str, dict[str, object]] = {}
    if liveness_tracker is not None:
        timestamp_for_liveness = now_s if now_s is not None else observation.timestamp
        if lidar_features is not None or lidar_n_points > 0:
            liveness_tracker.mark_observed("lidar", timestamp_for_liveness)
        if vision_features is not None:
            liveness_tracker.mark_observed("vision", timestamp_for_liveness)
        if audio_arr.size > 0 and audio_rms > 0.0:
            liveness_tracker.mark_observed("audio", timestamp_for_liveness)
        if motor.size > 0:
            liveness_tracker.mark_observed("motor", timestamp_for_liveness)
        snapshot = liveness_tracker.snapshot(now_s=timestamp_for_liveness)
        sensor_liveness = {name: status.to_dict() for name, status in snapshot.items()}

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
        sensor_liveness=sensor_liveness,
    )
