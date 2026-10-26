"""ESP32 liveness probe behind the preflight ``esp32`` check (F-059, D-3)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings

_log = get_logger(__name__)


@dataclass(frozen=True)
class Esp32ProbeResult:
    """Outcome of up to ``esp32.max_consecutive_timeouts`` polls plus one battery read.

    The polls and the read go through the public driver protocol.

    Attributes:
        responded: ``True`` when a stock ``FEEDBACK_BASE_INFO`` frame arrived
            within the poll budget.
        frame_type: The witnessed frame's ``T`` value, or ``None`` when the
            board did not respond.
        battery_v: Bus voltage in volts, or ``None`` when the board did not
            respond or its frame carried no usable voltage. The driver's
            ``0.0`` "no reading" sentinel is reported as ``None``, never as
            a flat pack.
        elapsed_s: Seconds from the start of the probe (driver build and
            connect included, disconnect excluded) to the verdict.
        detail: One-line operator-facing summary for preflight and MCP output.
            It names ``esp32.serial_port`` as the *configured* port: USB-C
            discovery may have opened a different by-id path, and the driver
            protocol does not expose the one in use.
    """

    responded: bool
    frame_type: int | None
    battery_v: float | None
    elapsed_s: float
    detail: str


def _configured_port_note(serial_port: str) -> str:
    """Name ``serial_port`` as the configured literal, not as the port that was opened.

    ``build_esp32_driver`` hands the serial driver a *copy* of the ESP32 config whose
    ``serial_port`` USB-C discovery may have replaced with a by-id path, and
    ``ESP32CommProtocol`` exposes no accessor for the port in use. The probe therefore
    knows only what was configured, and says so rather than present it as the endpoint
    that answered (or stayed silent).

    Args:
        serial_port: ``cfg.esp32.serial_port``.

    Returns:
        A fragment for the operator-facing summary.
    """
    return f"configured serial_port={serial_port} (USB-C discovery may substitute a by-id path)"


async def collect_esp32_probe(cfg: Settings) -> Esp32ProbeResult:
    """Connect, poll for one FEEDBACK_BASE_INFO frame, always disconnect.

    Public path only: under ``waveshare_stock`` ``read_encoders()`` sends the
    codec's ``encoder_query()`` (CMD_BASE_FEEDBACK) and ``imu_valid`` is set
    only by a parsed T:1001 frame, so it is the witness. Up to
    ``esp32.max_consecutive_timeouts`` polls (the driver's own "link is dead"
    budget) because stock shares one stream across frame types. Legacy has no
    polled frame to witness, so it is refused rather than fabricated.

    ``connect()`` arms the chassis heartbeat (T:136) when ``heartbeat_enabled``;
    the probe is motion-free but not wire-passive.

    A silent or wrong-typed board is an ordinary ``responded=False`` result,
    not an exception. Driver failures from ``connect()`` or from a poll
    propagate to the caller, and ``disconnect()`` runs either way: the serial
    driver starts its I/O executor before the open call, so a port that failed
    to open still has one to release. ``disconnect()`` is safe on a driver that
    never connected.

    Callers must gate on both ``cfg.mock_hardware`` and ``cfg.esp32.enabled``
    before calling: against a mock-backed config the result reads as a silent
    board.

    Args:
        cfg: Fully resolved settings. The driver is built from ``cfg`` through
            the factory, so mock, disabled and USB-C-discovery behaviour is
            the application's own.

    Returns:
        The probe verdict.

    Raises:
        ValueError: If ``cfg.esp32.command_set`` is not ``waveshare_stock``.
    """
    from mousedroid.comms.command_set import WAVESHARE_FEEDBACK_BASE_INFO
    from mousedroid.factory import build_esp32_driver

    esp32 = cfg.esp32
    if esp32.command_set != "waveshare_stock":
        msg = f"collect_esp32_probe needs command_set='waveshare_stock', got {esp32.command_set!r}"
        raise ValueError(msg)
    t0 = time.monotonic()
    driver = build_esp32_driver(cfg)
    try:
        await driver.connect()
        for polls in range(1, esp32.max_consecutive_timeouts + 1):
            if (await driver.read_encoders()).imu_valid:
                voltage = await driver.get_battery_voltage()
                battery_v = voltage if voltage > 0.0 else None
                elapsed = time.monotonic() - t0
                shown = "unavailable" if battery_v is None else f"{battery_v:.2f}"
                detail = (
                    f"FEEDBACK_BASE_INFO after {polls} poll(s); battery_v={shown}; "
                    f"{_configured_port_note(esp32.serial_port)}"
                )
                _log.info(
                    "esp32_probe_complete",
                    responded=True,
                    polls=polls,
                    battery_v=battery_v,
                    elapsed_s=elapsed,
                )
                return Esp32ProbeResult(
                    True, WAVESHARE_FEEDBACK_BASE_INFO, battery_v, elapsed, detail
                )
        elapsed = time.monotonic() - t0
        detail = (
            f"no FEEDBACK_BASE_INFO frame in {esp32.max_consecutive_timeouts} polls at "
            f"{esp32.serial_baud} baud; {_configured_port_note(esp32.serial_port)}"
        )
        _log.warning(
            "esp32_probe_complete",
            responded=False,
            polls=esp32.max_consecutive_timeouts,
            configured_port=esp32.serial_port,
            baud=esp32.serial_baud,
            elapsed_s=elapsed,
        )
        return Esp32ProbeResult(False, None, None, elapsed, detail)
    finally:
        await driver.disconnect()
