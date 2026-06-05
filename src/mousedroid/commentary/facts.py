"""Pure, hardware-free extraction of :class:`CommentaryFacts` from an observation.

No orchestrator dependency and no I/O, so it is unit-testable in isolation. The
guards mirror ``orchestrator._voice_observe`` exactly: LiDAR min only when the
feature vector is present and non-empty; audio RMS only over a non-empty chunk;
every value ``float()``-cast and ``np.isfinite``-checked so an empty / NaN / inf
modality degrades to a safe ``0.0`` plus a ``*_valid=False`` flag rather than
poisoning the gate or raising.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mousedroid.commentary.protocol import CommentaryFacts

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mousedroid.sensing.protocol import ObservationProtocol

# motor_state layout is [vx, vy, omega, battery_v]; guard each index against a
# short vector (mirrors the orchestrator's ``motor_state.size > INDEX`` guards).
_IDX_VX = 0
_IDX_VY = 1
_IDX_OMEGA = 2
_IDX_BATTERY = 3
# Neutral battery default when the motor vector is too short to carry it; matches
# the SafetyContext default so downstream "low battery" logic is consistent.
_DEFAULT_BATTERY_V = 12.0


def _finite(value: float, default: float = 0.0) -> float:
    """Return ``value`` as a finite float, falling back to ``default``."""
    out = float(value)
    return out if np.isfinite(out) else default


def _motor(motor_state: NDArray[np.float32], idx: int, default: float = 0.0) -> float:
    """Read ``motor_state[idx]`` defensively (short vectors -> ``default``)."""
    if motor_state.size <= idx:
        return default
    return _finite(motor_state[idx], default)


def _audio_rms(audio_chunk: NDArray[np.float32]) -> tuple[float, bool]:
    """Compute RMS energy of ``audio_chunk``; return ``(rms, valid)``.

    Guards the empty-array case BEFORE computing (so no ``RuntimeWarning`` from
    ``np.mean`` over size 0) and rejects non-finite results.
    """
    if audio_chunk.size == 0:
        return (0.0, False)
    rms = float(np.sqrt(np.mean(np.square(audio_chunk, dtype=np.float64))))
    if not np.isfinite(rms):
        return (0.0, False)
    return (rms, True)


def _min_clearance(
    lidar_features: NDArray[np.float32] | None,
    forward_distance_m: float,
) -> tuple[float, bool]:
    """Return ``(min_clearance_m, lidar_valid)``.

    Uses the LiDAR 360 minimum when available; otherwise falls back to the
    forward ultrasonic distance with ``lidar_valid=False`` so composers can
    avoid over-claiming a full clearance picture.
    """
    if lidar_features is not None and lidar_features.size > 0:
        m = float(np.min(lidar_features))
        if np.isfinite(m):
            return (m, True)
    return (forward_distance_m, False)


def extract_commentary_facts(
    observation: ObservationProtocol,
    *,
    novelty: float | None,
    is_emergency: bool,
    embedding: NDArray[np.float32] | None = None,
) -> CommentaryFacts:
    """Marshal a grounded :class:`CommentaryFacts` from one observation.

    Args:
        observation: The fused observation bundle for this tick.
        novelty: The freshly-sampled curiosity novelty, or ``None`` when no
            curiosity module exists (distinct from a genuine ``0.0``).
        is_emergency: Whether this tick is flagged as an emergency.
        embedding: Optional RSSM embedding of this moment (Phase-1 recognition
            key). ``None`` (default) when recognition is disabled — keeps the
            Phase-0 facts byte-identical.

    Returns:
        A frozen :class:`CommentaryFacts`. Never raises on degraded sensors.
    """
    motor_state = observation.motor_state
    forward_distance_m = _finite(observation.distance_m, 0.0)
    min_clearance_m, lidar_valid = _min_clearance(observation.lidar_features, forward_distance_m)
    audio_rms, audio_valid = _audio_rms(observation.audio_chunk)
    vx = _motor(motor_state, _IDX_VX)
    vy = _motor(motor_state, _IDX_VY)
    speed_mps = float(np.hypot(vx, vy))
    return CommentaryFacts(
        min_clearance_m=min_clearance_m,
        forward_distance_m=forward_distance_m,
        audio_rms=audio_rms,
        speed_mps=speed_mps,
        turn_rate=_motor(motor_state, _IDX_OMEGA),
        battery_v=_motor(motor_state, _IDX_BATTERY, _DEFAULT_BATTERY_V),
        novelty=None if novelty is None else _finite(novelty, 0.0),
        is_emergency=is_emergency,
        lidar_valid=lidar_valid,
        audio_valid=audio_valid,
        timestamp=_finite(observation.timestamp, 0.0),
        embedding=embedding,
    )


__all__ = ["extract_commentary_facts"]
