"""Shared utilities and constants for communication drivers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.comms.protocol import EncoderReading
    from mousedroid.config.schema import ESP32Config

_log = get_logger(__name__)

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


def build_velocity_cmd(
    vx: float,
    vy: float,
    omega: float,
    cfg: ESP32Config,
) -> dict[str, int]:
    """Convert physical velocity setpoints to an ESP32 PWM command dict.

    Scales each axis to the ``[-MAX_PWM, MAX_PWM]`` integer range and packs
    the result into a command dictionary ready for transmission.

    Args:
        vx: Forward velocity in m/s.
        vy: Lateral velocity in m/s.
        omega: Angular velocity in rad/s.
        cfg: ESP32 configuration supplying ``max_velocity_mps`` and
            ``max_omega_rads``.

    Returns:
        Command dictionary with keys ``T``, ``vx``, ``vy``, ``omega``.
    """
    max_vel = cfg.max_velocity_mps
    pwm_vx = int(clamp(vx / max_vel, -1.0, 1.0) * MAX_PWM)
    pwm_vy = int(clamp(vy / max_vel, -1.0, 1.0) * MAX_PWM)
    pwm_omega = int(clamp(omega / cfg.max_omega_rads, -1.0, 1.0) * MAX_PWM)
    _log.debug(
        "velocity_cmd_built",
        vx_pwm=pwm_vx,
        vy_pwm=pwm_vy,
        omega_pwm=pwm_omega,
    )
    return {
        "T": ESP32_CMD_TYPE_VELOCITY,
        "vx": pwm_vx,
        "vy": pwm_vy,
        "omega": pwm_omega,
    }


def parse_encoder_reading(data: dict[str, Any]) -> EncoderReading:
    """Parse a raw ESP32 JSON response into an ``EncoderReading``.

    Missing keys fall back to ``0.0``.

    Args:
        data: Raw JSON dictionary from the ESP32.

    Returns:
        Populated ``EncoderReading`` dataclass.
    """
    from mousedroid.comms.protocol import EncoderReading  # avoid circular at import time

    expected_keys = {"lv", "rv", "ox", "oy", "h", "ts"}
    missing = expected_keys - data.keys()
    if missing:
        _log.debug("encoder_fields_missing", missing_keys=sorted(missing))

    return EncoderReading(
        left_velocity_mps=float(data.get("lv", 0.0)),
        right_velocity_mps=float(data.get("rv", 0.0)),
        odometry_x_m=float(data.get("ox", 0.0)),
        odometry_y_m=float(data.get("oy", 0.0)),
        heading_rad=float(data.get("h", 0.0)),
        timestamp=float(data.get("ts", 0.0)),
    )
