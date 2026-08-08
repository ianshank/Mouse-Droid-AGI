"""Serial (UART) ESP32 communication driver for Wave Rover motor control.

Implements ``ESP32CommProtocol`` over a serial connection to the ESP32.

Includes adaptive timeout: after ``max_consecutive_timeouts`` empty reads
the serial timeout degrades to ``degraded_timeout_s`` to avoid blocking
the orchestrator loop.  Any successful read restores the original timeout.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import TYPE_CHECKING, Any

from mousedroid.comms.base_driver import BaseESP32Driver
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mousedroid.config.schema import ESP32Config

try:
    import serial as _serial_mod
except ImportError:  # pragma: no cover
    _serial_mod = None

_log = get_logger(__name__)


class SerialESP32Driver(BaseESP32Driver):
    """ESP32 driver using serial UART, implementing ``ESP32CommProtocol``.

    All blocking serial I/O is delegated to ``asyncio.to_thread``.
    High-level protocol methods (``send_velocity``, ``read_encoders``,
    ``get_battery_voltage``, ``emergency_stop``) are inherited from
    ``BaseESP32Driver``.

    Adaptive timeout:
        Tracks consecutive empty reads.  After exceeding
        ``cfg.max_consecutive_timeouts`` the serial read timeout is
        reduced to ``cfg.degraded_timeout_s`` to prevent the 500 ms
        blocking window from dominating the orchestrator loop.  A
        successful read resets the counter and restores the original
        timeout.
    """

    def __init__(self, cfg: ESP32Config) -> None:
        """Initialise serial driver from config.

        Args:
            cfg: ESP32 communication configuration.
        """
        super().__init__(cfg)
        self._port: str = cfg.serial_port
        self._baud: int = cfg.serial_baud
        self._serial: Any = None

        # Adaptive timeout state
        self._normal_timeout: float = cfg.command_timeout_s
        self._degraded_timeout: float = cfg.degraded_timeout_s
        self._max_consecutive_timeouts: int = cfg.max_consecutive_timeouts
        self._consecutive_timeouts: int = 0
        self._is_degraded: bool = False
        self._last_probe_time: float = 0.0
        self._degraded_poll_interval: float = cfg.degraded_poll_interval_s
        # Serialises the send-then-read pair in ``_query_data`` (see there).
        # Created lazily-safe here: the driver is constructed inside the
        # running loop by the factory, and asyncio.Lock() no longer binds a
        # loop at construction time on the supported Python versions.
        self._io_lock: asyncio.Lock = asyncio.Lock()

    async def connect(self) -> None:
        """Open serial connection to ESP32."""
        if _serial_mod is None:
            msg = "pyserial is not installed — install mousedroid[hardware]"
            raise RuntimeError(msg)
        self._serial = await asyncio.to_thread(self._open_serial)
        self._connected = True
        self._consecutive_timeouts = 0
        self._is_degraded = False
        _log.info("serial_esp32_connected", port=self._port, baud=self._baud)
        # F-025: codec connect-time commands (chassis heartbeat under
        # waveshare_stock; empty under legacy — zero extra writes).
        #
        # Rollback on failure is mandatory: this is the first I/O between
        # opening the port and returning, so an exception here would leave a
        # live handle behind. ResilientESP32Driver retries connect(), and each
        # retry re-runs _open_serial() — without the close we would leak one
        # fd per attempt AND advertise _connected=True with the failsafe
        # unarmed. Close first, then re-raise so the retry starts clean.
        try:
            await self._arm_command_set()
        except Exception:
            _log.warning(
                "serial_esp32_arm_failed_rolling_back",
                port=self._port,
                command_set=self._cfg.command_set,
            )
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._close_serial)
            self._serial = None
            self._connected = False
            raise

    def _open_serial(self) -> Any:  # pragma: no cover
        """Open the serial port (blocking)."""
        return _serial_mod.Serial(
            self._port,
            self._baud,
            timeout=self._timeout,
        )

    async def disconnect(self) -> None:
        """Close serial connection to ESP32."""
        if self._serial is not None:
            await asyncio.to_thread(self._close_serial)
        self._connected = False
        _log.info("serial_esp32_disconnected")

    def _close_serial(self) -> None:  # pragma: no cover
        """Close the serial port (blocking)."""
        self._serial.close()
        self._serial = None

    # ------------------------------------------------------------------
    # Transport implementation
    # ------------------------------------------------------------------

    async def _send_command(self, cmd: Mapping[str, float]) -> None:
        """Write a JSON command to the serial port.

        Args:
            cmd: Command payload to serialise and send.
        """
        await self._send_json(cmd)

    async def _query_data(
        self,
        resource: str,
        cmd: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Send an optional command then read one JSON line from serial.

        For serial, the ESP32 responds on the same channel so a preceding
        command is required when the device needs to be polled (e.g. battery).

        Args:
            resource: Logical resource name (unused for serial, present for
                interface compatibility).
            cmd: Optional command to send before reading the response.

        Returns:
            Parsed JSON response dictionary.
        """
        # The send-then-read pair MUST be atomic. ``SensorManager`` gathers
        # read_encoders() and get_battery_voltage() concurrently, and each
        # leg yields control at its ``asyncio.to_thread`` boundary — without
        # this lock two ``readline()`` calls run in different OS threads on
        # the same pyserial handle (not thread-safe: a single line can be
        # split between them), replies can pair with the wrong request, and
        # the degraded-mode counters below are mutated from both tasks.
        async with self._io_lock:
            if self.should_skip_read():
                # In degraded mode, throttle actual reads to the configured
                # poll interval so we reduce serial traffic instead of busy-
                # polling with a short timeout. Skip both the transmit and the
                # read so we do not queue unread responses in the serial buffer.
                return {}
            if cmd is not None:
                await self._send_json(cmd)
            return await self._read_json()

    # ------------------------------------------------------------------
    # Adaptive timeout
    # ------------------------------------------------------------------

    @property
    def is_degraded(self) -> bool:
        """Whether the driver is in degraded (fast-timeout) mode."""
        return self._is_degraded

    @property
    def consecutive_timeouts(self) -> int:
        """Number of consecutive empty reads."""
        return self._consecutive_timeouts

    def should_skip_read(self) -> bool:
        """Return True if we are degraded and the poll interval hasn't elapsed."""
        if not self._is_degraded:
            return False
        return (time.monotonic() - self._last_probe_time) < self._degraded_poll_interval

    def _enter_degraded(self) -> None:
        """Switch serial timeout to degraded value."""
        if self._is_degraded:
            return
        self._is_degraded = True
        if self._serial is not None:
            self._serial.timeout = self._degraded_timeout
        _log.warning(
            "esp32_entering_degraded_mode",
            consecutive_timeouts=self._consecutive_timeouts,
            degraded_timeout_s=self._degraded_timeout,
        )

    def _exit_degraded(self) -> None:
        """Restore serial timeout to normal value."""
        if not self._is_degraded:
            return
        self._is_degraded = False
        self._consecutive_timeouts = 0
        if self._serial is not None:
            self._serial.timeout = self._normal_timeout
        _log.info("esp32_recovered_from_degraded")

    def _record_timeout(self) -> None:
        """Record an empty/timeout read and enter degraded mode if threshold exceeded."""
        self._consecutive_timeouts += 1
        if self._consecutive_timeouts >= self._max_consecutive_timeouts:
            self._enter_degraded()

    def _record_success(self) -> None:
        """Record a successful read, exiting degraded mode if active."""
        if self._is_degraded:
            self._exit_degraded()
        else:
            self._consecutive_timeouts = 0

    # ------------------------------------------------------------------
    # Low-level serial helpers
    # ------------------------------------------------------------------

    async def _send_json(self, data: Mapping[str, Any]) -> None:
        """Serialise and send JSON over serial.

        Args:
            data: Payload mapping to send as JSON.
        """
        payload = json.dumps(dict(data)).encode() + b"\n"
        await asyncio.to_thread(self._write_bytes, payload)

    def _write_bytes(self, payload: bytes) -> None:  # pragma: no cover
        """Write raw bytes to serial port (blocking).

        Args:
            payload: Bytes to write.
        """
        self._serial.write(payload)

    async def _read_json(self) -> dict[str, Any]:
        """Read one JSON line from serial.

        Tracks adaptive timeout state: empty reads increment the timeout
        counter; successful reads reset it and restore normal timeout. The
        raw decoded line is logged at DEBUG so operators can grep
        ``esp32_raw_line`` to triage protocol-level mismatches (firmware
        version drift, partial framing, non-JSON output) without rewiring
        the driver.

        Returns:
            Parsed JSON dictionary.
        """
        self._last_probe_time = time.monotonic()
        raw = await asyncio.to_thread(self._read_line)
        if not raw:
            self._record_timeout()
            return {}
        # Truncation length is config-driven (cfg.debug_log_max_chars) so
        # operators triaging firmware-protocol drift can widen the window
        # without editing source. Default 200 stays compact for normal smoke
        # runs; bump via MOUSEDROID_ESP32__DEBUG_LOG_MAX_CHARS for triage.
        truncate = self._cfg.debug_log_max_chars
        _log.debug("esp32_raw_line", line=raw[:truncate], len=len(raw))
        self._record_success()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            _log.warning(
                "esp32_non_json_response",
                line=raw[:truncate],
                error=str(exc),
            )
            return {}
        if not isinstance(parsed, dict):
            _log.warning(
                "esp32_response_not_object",
                line=raw[:truncate],
                got=type(parsed).__name__,
            )
            return {}
        return parsed

    def _read_line(self) -> str:  # pragma: no cover
        """Read one line from serial port (blocking).

        Uses ``errors="replace"`` so a garbled byte from firmware churn,
        UART noise, or a partial flash never raises ``UnicodeDecodeError``
        out of the ``asyncio.to_thread`` wrapper. The replacement char
        survives into the downstream ``json.loads`` which then emits the
        existing ``esp32_non_json_response`` warning path.

        Returns:
            Decoded line string.
        """
        line: bytes = self._serial.readline()
        return line.decode(errors="replace").strip()
