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
) -> TelemetryFrame:
    """Build a ``TelemetryFrame`` from an observation and safety context.

    Args:
        observation: Current sensor observation bundle.
        safety_ctx: Current safety evaluation result.
        loop_time_ms: Control loop iteration time (milliseconds).
        tick_count: Monotonically increasing tick counter.

    Returns:
        Fully-populated ``TelemetryFrame`` ready for publishing.
    """
    vision_arr = observation.vision_features
    vision_norm = float(np.sqrt(np.sum(vision_arr * vision_arr)))

    audio_arr = observation.audio_chunk
    audio_rms = float(np.sqrt(np.mean(audio_arr * audio_arr)))

    motor = observation.motor_state
    battery_v = (
        float(motor[MOTOR_STATE_BATTERY_INDEX])
        if motor.size > MOTOR_STATE_BATTERY_INDEX
        else 0.0
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
        },
        loop_time_ms=loop_time_ms,
        tick_count=tick_count,
    )
