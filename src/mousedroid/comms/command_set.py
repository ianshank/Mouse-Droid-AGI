"""ESP32 firmware command-set codecs (F-025).

The rover's original drivers speak a private JSON protocol
(``{"T":1,"vx","vy","omega"}``) that no committed firmware implements —
stock Waveshare ``General_Driver`` firmware (``waveshareteam/
ugv_base_general``) reads different commands entirely, and the legacy
battery poll ``{"T":2}`` is stock ``CMD_SET_MOTOR_PID``: a motor-controller
*write*. This module isolates every command-build and response-parse
decision behind a small strategy protocol so the transports
(:class:`~mousedroid.comms.serial_driver.SerialESP32Driver`,
:class:`~mousedroid.comms.wifi_driver.WiFiESP32Driver`) and the resilience
wrapper stay command-set-agnostic.

Selection is driven by :attr:`ESP32Config.command_set`
(``"legacy"`` default — byte-identical pre-F-025 behaviour) via
:func:`resolve_command_codec`. Codecs are stateless module-level
singletons; per-connection state (e.g. the lateral-velocity warn latch)
lives on the driver instance.

Stock protocol facts verified against vendor source (2026-08-07):
``json_cmd.h`` defines ``CMD_ROS_CTRL``=13 (``{"T":13,"X":<m/s>,
"Z":<rad/s>}``), ``CMD_HEART_BEAT_SET``=136 (``{"T":136,"cmd":<ms>}``),
``CMD_BASE_FEEDBACK``=130 → one ``FEEDBACK_BASE_INFO``=1001 reply;
``ugv_advance.h::baseInfoFeedback()`` writes keys ``L``/``R`` (wheel
speeds), ``r``/``p``/``y`` (IMU), ``temp`` and ``v`` (bus voltage); stock
firmware defines NO dedicated e-stop command, so stop is
``{"T":13,"X":0,"Z":0}``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from mousedroid.comms._utils import (
    ESP32_CMD_TYPE_BATTERY,
    ESP32_CMD_TYPE_STOP,
    build_velocity_cmd,
    clamp,
    parse_encoder_reading,
)
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mousedroid.comms.protocol import EncoderReading
    from mousedroid.config.schema import ESP32Config

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Stock Waveshare General_Driver command constants (json_cmd.h)
# ---------------------------------------------------------------------------

WAVESHARE_CMD_ROS_CTRL: Final[int] = 13  # hardcoded-ok: vendor protocol constant (json_cmd.h)
"""``CMD_ROS_CTRL`` — velocity in physical units: ``{"T":13,"X":<m/s>,"Z":<rad/s>}``."""

WAVESHARE_CMD_BASE_FEEDBACK: Final[int] = 130  # hardcoded-ok: vendor protocol constant
"""``CMD_BASE_FEEDBACK`` — polls one ``FEEDBACK_BASE_INFO`` frame."""

WAVESHARE_CMD_HEART_BEAT_SET: Final[int] = 136  # hardcoded-ok: vendor protocol constant
"""``CMD_HEART_BEAT_SET`` — arms the chassis failsafe: ``{"T":136,"cmd":<window ms>}``."""

WAVESHARE_FEEDBACK_BASE_INFO: Final[int] = 1001  # hardcoded-ok: vendor protocol constant
"""``FEEDBACK_BASE_INFO`` — the telemetry frame carrying ``L``/``R``/``v``."""

_MS_PER_SECOND: Final[float] = 1000.0  # hardcoded-ok: unit conversion, not a tunable
"""Milliseconds per second — heartbeat-window derivation factor."""

MIN_HEARTBEAT_WINDOW_MS: Final[int] = 1  # hardcoded-ok: smallest window the wire can express
"""Floor for the derived heartbeat window.

A window of ``0`` is NOT "arm the failsafe as tightly as possible" — stock
firmware reads ``{"T":136,"cmd":0}`` as *disable the heartbeat*. Without this
floor a config typo (``keepalive_hz`` >= 6000 rounds the derivation to zero)
silently disarms the very failsafe the feature exists to provide, while
``heartbeat_enabled`` still reads ``True`` and the driver still logs
``esp32_heartbeat_armed``. Clamping keeps the invariant honest: *enabled means
actually armed*. Operators who want the failsafe off set
``heartbeat_enabled=False`` — the one explicit way to express that."""


def _coerce_float(raw: Any) -> float | None:
    """Coerce a wire value to ``float``; ``None`` when it is absent or junk.

    Every numeric field arriving from the ESP32 crosses this helper. UART
    noise, a brown-out mid-frame and firmware churn all produce well-formed
    JSON carrying a non-numeric value, and a bare ``float()`` on that raises
    out of the codec, up through ``_query_data`` and into the 30 Hz sensor
    read — turning a recoverable comms glitch into a crashed tick.

    Callers choose their own fallback because the two consumers differ:
    a missing battery reading is ``None`` (semantically "no telemetry",
    which the safety layer screens), while a missing wheel speed is ``0.0``
    (the encoder-less chassis reports zero regardless).

    Args:
        raw: The value as parsed from the response frame.

    Returns:
        The coerced float, or ``None`` when absent or uncoercible.
    """
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def worst_case_command_gap_s(cfg: ESP32Config) -> float:
    """Longest gap between commands the driver considers normal-but-slow.

    Three configured budgets can each stall the command stream, and the
    heartbeat must out-wait the largest of them or the firmware halts the
    wheels during ordinary operation:

    * ``1 / keepalive_hz`` — the nominal command cadence.
    * ``command_timeout_s`` — one timed-out read blocks for this long.
    * ``degraded_poll_interval_s`` — degraded mode *deliberately* throttles
      to one probe per interval to cut serial traffic.

    Deriving from the max keeps the failsafe self-consistent: an operator who
    tightens the timeouts automatically tightens the window, with no second
    number to remember and nothing hardcoded against a sibling field.

    Args:
        cfg: ESP32 configuration supplying the three budgets.

    Returns:
        Worst-case gap in seconds.
    """
    return max(
        1.0 / cfg.keepalive_hz,
        cfg.command_timeout_s,
        cfg.degraded_poll_interval_s,
    )


def heartbeat_window_ms(cfg: ESP32Config) -> int:
    """Derive the chassis heartbeat window from the driver's own budgets.

    ``window_ms = ceil(worst_case_command_gap_s * 1000 *
    heartbeat_window_multiple)``, floored at
    :data:`MIN_HEARTBEAT_WINDOW_MS`. With the shipped defaults the worst case
    is ``degraded_poll_interval_s`` (1.0 s), so the window is 3000 ms: a hung
    host halts the wheels within three seconds, while neither a timed-out
    read (500 ms) nor a degraded-mode probe cycle (1000 ms) can trip it.

    Earlier revisions derived from ``keepalive_hz`` alone, which yielded a
    300 ms window — shorter than *two* of the driver's own blocking budgets,
    so the failsafe would fire during conditions the host considers normal.
    ``ESP32Config`` still warns if a hand-tuned multiple reintroduces that.

    ``ceil`` rather than ``round``: banker's rounding would turn a 0.5 ms
    window into 0 (i.e. *disabled* — see :data:`MIN_HEARTBEAT_WINDOW_MS`) and
    a 2.5 ms window into 2. Rounding up never disarms anything.

    Args:
        cfg: ESP32 configuration supplying the budgets and the multiple.

    Returns:
        Window length in whole milliseconds (firmware takes an int), never
        below :data:`MIN_HEARTBEAT_WINDOW_MS`.
    """
    derived = math.ceil(
        worst_case_command_gap_s(cfg) * _MS_PER_SECOND * cfg.heartbeat_window_multiple
    )
    return max(MIN_HEARTBEAT_WINDOW_MS, derived)


@runtime_checkable
class ESP32CommandCodec(Protocol):
    """Strategy protocol for building/parsing one firmware's command set.

    Implementations MUST be stateless (module-level singletons are shared
    across driver instances); any per-connection state belongs on the
    driver. All payloads are read-only mappings — legacy implementations
    may return ``dict[str, int]`` (covariant under ``Mapping[str, float]``),
    keeping wire serialization byte-identical.
    """

    supports_lateral: bool
    """Whether the command set can express a lateral (vy) velocity axis."""

    def build_velocity(
        self, vx: float, vy: float, omega: float, cfg: ESP32Config
    ) -> Mapping[str, float]:
        """Build the velocity command for physical setpoints."""
        ...

    def build_stop(self) -> Mapping[str, float]:
        """Build the strongest stop command the firmware understands."""
        ...

    def battery_query(self) -> Mapping[str, float] | None:
        """Command to elicit a battery reading, or ``None`` for read-only."""
        ...

    def encoder_query(self) -> Mapping[str, float] | None:
        """Command to elicit an encoder/telemetry frame, or ``None``."""
        ...

    def parse_battery(self, data: Mapping[str, Any]) -> float | None:
        """Extract battery volts, or ``None`` when the frame carries none.

        ``None`` means *no reading available* (timeout, degraded-mode skip,
        wrong frame type) — semantically distinct from a genuine low
        voltage, which the safety layer must not confuse for a flat pack.
        """
        ...

    def parse_encoders(self, data: Mapping[str, Any]) -> EncoderReading:
        """Extract an :class:`EncoderReading` from a response frame."""
        ...

    def connect_commands(self, cfg: ESP32Config) -> list[Mapping[str, float]]:
        """Commands to send once per successful transport connect."""
        ...


class LegacyCommandCodec:
    """Pre-F-025 private protocol — pure delegation to ``_utils``.

    Every payload and parse is byte-identical to the historical driver:
    velocity is the PWM-scaled ``{"T":1,"vx","vy","omega"}`` dict, stop is
    ``{"T":0}``, the battery poll is ``{"T":2}`` reading key ``"v"``, and
    encoder reads send nothing (the firmware pushes frames).
    """

    supports_lateral: bool = True

    def build_velocity(
        self, vx: float, vy: float, omega: float, cfg: ESP32Config
    ) -> dict[str, int]:
        """Delegate to :func:`build_velocity_cmd` (PWM ints)."""
        return build_velocity_cmd(vx, vy, omega, cfg)

    def build_stop(self) -> dict[str, int]:
        """Legacy emergency stop ``{"T":0}``."""
        return {"T": ESP32_CMD_TYPE_STOP}

    def battery_query(self) -> dict[str, int]:
        """Legacy battery poll ``{"T":2}``."""
        return {"T": ESP32_CMD_TYPE_BATTERY}

    def encoder_query(self) -> None:
        """Legacy encoder reads send nothing — the firmware pushes frames."""
        return None

    def parse_battery(self, data: Mapping[str, Any]) -> float | None:
        """Read key ``"v"``; ``None`` when the frame carries no voltage.

        Legacy firmware answers the ``{"T":2}`` poll with a ``"v"`` key, so a
        frame without one is a non-answer rather than a zero-volt rover.
        Returning ``None`` (instead of the historical fabricated ``0.0``)
        lets the driver distinguish "no telemetry" from "flat pack".
        """
        return _coerce_float(data.get("v"))

    def parse_encoders(self, data: Mapping[str, Any]) -> EncoderReading:
        """Delegate to :func:`parse_encoder_reading` (keys ``lv``/``rv``/…)."""
        return parse_encoder_reading(dict(data))

    def connect_commands(self, cfg: ESP32Config) -> list[Mapping[str, float]]:
        """Legacy firmware has no arming commands — send nothing."""
        return []


class WaveshareStockCodec:
    """Stock ``General_Driver`` command set (``ugv_base_general``).

    Velocity goes out as ``CMD_ROS_CTRL`` in physical units (the firmware
    owns its own motor mapping — no host-side PWM scaling); stop is a
    zero-velocity ``CMD_ROS_CTRL`` (stock defines no e-stop command),
    backed by the ``CMD_HEART_BEAT_SET`` failsafe armed at connect.
    Battery and wheel telemetry are polled with ``CMD_BASE_FEEDBACK`` and
    parsed from the ``FEEDBACK_BASE_INFO`` frame, **T-gated**: frames
    arrive on a shared stream (no per-command ACKs), so a non-1001 frame
    parses to the same silent-zero the legacy contract produced, with a
    DEBUG breadcrumb for smoke triage.
    """

    supports_lateral: bool = False

    def build_velocity(
        self, vx: float, vy: float, omega: float, cfg: ESP32Config
    ) -> dict[str, float]:
        """Clamped physical-unit ``{"T":13,"X","Z"}`` (``vy`` has no axis).

        The lateral term is intentionally absent — WAVE ROVER is
        differential/skid-steer and ``CMD_ROS_CTRL`` has no lateral key.
        The driver owns the operator-facing warn (``supports_lateral``).
        """
        del vy  # no lateral axis in CMD_ROS_CTRL; driver warns via supports_lateral
        max_vel = cfg.max_velocity_mps
        return {
            "T": WAVESHARE_CMD_ROS_CTRL,
            "X": clamp(vx, -max_vel, max_vel),
            "Z": clamp(omega, -cfg.max_omega_rads, cfg.max_omega_rads),
        }

    def build_stop(self) -> dict[str, float]:
        """Zero-velocity ``CMD_ROS_CTRL`` — stock has no e-stop command."""
        return {"T": WAVESHARE_CMD_ROS_CTRL, "X": 0.0, "Z": 0.0}

    def battery_query(self) -> dict[str, int]:
        """Poll one telemetry frame — a READ, unlike the legacy PID-write."""
        return {"T": WAVESHARE_CMD_BASE_FEEDBACK}

    def encoder_query(self) -> dict[str, int]:
        """Stock frames must be polled; nothing is pushed unsolicited."""
        return {"T": WAVESHARE_CMD_BASE_FEEDBACK}

    def parse_battery(self, data: Mapping[str, Any]) -> float | None:
        """Voltage key ``"v"`` from a T-gated ``FEEDBACK_BASE_INFO`` frame.

        Returns ``None`` for an empty payload (timeout / degraded-mode skip)
        or any non-1001 frame — stock firmware streams several frame types
        with no per-command ACKs, so reading a wrong-typed frame is routine
        and must not be reported as a zero-volt battery.
        """
        if not self._is_base_info(data):
            return None
        return _coerce_float(data.get("v"))

    def parse_encoders(self, data: Mapping[str, Any]) -> EncoderReading:
        """Map ``L``/``R`` wheel speeds; odometry/heading stay zero.

        WAVE ROVER is encoder-less — ``L``/``R`` echo commanded speed, and
        the chassis has no odometry source. IMU-derived heading (keys
        ``r``/``p``/``y`` in the same frame) is deliberately NOT consumed
        here; that is the audit-R4 sensing feature, not a comms concern.

        ``_is_base_info`` gates only the ``T`` key, so a well-typed 1001
        frame can still carry a malformed wheel speed. Both fields go
        through :func:`_coerce_float` for that reason — a raise here would
        propagate out of the 30 Hz sensor read.
        """
        from mousedroid.comms.protocol import EncoderReading  # deferred: circular at import

        if not self._is_base_info(data):
            return EncoderReading()
        left = _coerce_float(data.get("L"))
        right = _coerce_float(data.get("R"))
        return EncoderReading(
            left_velocity_mps=0.0 if left is None else left,
            right_velocity_mps=0.0 if right is None else right,
        )

    def connect_commands(self, cfg: ESP32Config) -> list[Mapping[str, float]]:
        """Arm the chassis heartbeat failsafe (window from ``keepalive_hz``)."""
        if not cfg.heartbeat_enabled:
            return []
        return [
            {
                "T": WAVESHARE_CMD_HEART_BEAT_SET,
                "cmd": heartbeat_window_ms(cfg),
            }
        ]

    @staticmethod
    def _is_base_info(data: Mapping[str, Any]) -> bool:
        """T-gate: ``True`` only for a ``FEEDBACK_BASE_INFO`` frame.

        Empty payloads (timeout / degraded-skip) stay silent, matching the
        legacy silent-zero contract; a *wrong-typed* frame leaves a DEBUG
        breadcrumb (this sits on the 30 Hz read path — DEBUG per the
        ``log_command_dispatch`` precedent).
        """
        if not data:
            return False
        try:
            frame_type = int(data.get("T", -1))
        except (TypeError, ValueError):
            frame_type = -1
        if frame_type != WAVESHARE_FEEDBACK_BASE_INFO:
            _log.debug("esp32_stock_frame_mismatch", got_type=frame_type)
            return False
        return True


LEGACY_CODEC: Final[LegacyCommandCodec] = LegacyCommandCodec()
"""Shared stateless legacy codec instance."""

WAVESHARE_STOCK_CODEC: Final[WaveshareStockCodec] = WaveshareStockCodec()
"""Shared stateless stock-Waveshare codec instance."""


def command_set_supports_lateral(cfg: ESP32Config) -> bool:
    """Whether the configured command set can execute a lateral velocity.

    Lets callers outside :mod:`mousedroid.comms` (notably the orchestrator's
    action seam) ask the question without reaching through the driver into
    private codec state, and keeps the answer owned by the codec rather than
    duplicated as a ``command_set == ...`` literal at each call site.

    Args:
        cfg: ESP32 configuration carrying the ``command_set`` selector.

    Returns:
        ``True`` when the command set has a lateral axis.
    """
    return resolve_command_codec(cfg).supports_lateral


def resolve_command_codec(cfg: ESP32Config) -> ESP32CommandCodec:
    """Select the codec for ``cfg.command_set``.

    The legacy value is the un-guarded fall-through (the
    ``build_llm_gateway`` dispatch pattern), so a future selector value
    can never silently break the pre-F-025 path.

    Args:
        cfg: ESP32 configuration carrying the ``command_set`` selector.

    Returns:
        The shared stateless codec singleton for the selected set.
    """
    if cfg.command_set == "waveshare_stock":
        return WAVESHARE_STOCK_CODEC
    # Default / legacy path.
    return LEGACY_CODEC
