"""Shared utilities and constants for communication drivers."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# ESP32 protocol constants (shared between serial and WiFi drivers)
# ---------------------------------------------------------------------------

MAX_PWM: int = 255
"""Maximum PWM value for motor control (8-bit range)."""

ESP32_CMD_TYPE_VELOCITY: int = 1
"""ESP32 JSON command type for velocity control."""

ESP32_CMD_TYPE_STOP: int = 0
"""ESP32 JSON command type for emergency stop."""

ESP32_CMD_TYPE_BATTERY: int = 2
"""ESP32 JSON command type for battery voltage query."""


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value between lo and hi.

    Args:
        value: Value to clamp.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Clamped value.
    """
    return max(lo, min(hi, value))
