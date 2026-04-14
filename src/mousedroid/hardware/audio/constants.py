"""Shared audio constants for hardware drivers and voice synthesis."""

from __future__ import annotations

INT16_MAX_F: float = 32768.0
"""Scale factor for int16 <-> float32 audio conversion."""

POWER_CLIP_MAX: float = 1e20
"""Maximum power spectral density value to prevent overflow."""

LOG_FLOOR: float = 1e-10
"""Minimum value before log scaling to avoid log(0)."""
