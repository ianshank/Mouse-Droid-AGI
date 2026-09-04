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
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Final, TypeVar

from mousedroid.comms.base_driver import BaseESP32Driver
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mousedroid.config.schema import ESP32Config

try:
    import serial as _serial_mod
except ImportError:  # pragma: no cover
    _serial_mod = None

_log = get_logger(__name__)

_IoResult = TypeVar("_IoResult")

#: Worker count for the per-driver serial I/O executor.
#:
#: Deliberately **not** a config field: this is not a tuning knob, it is the
#: mechanism that serialises access to a single non-thread-safe pyserial
#: handle. Raising it reintroduces exactly the interleaving the executor
#: exists to prevent, so it must not be reachable from YAML or an env var.
_SERIAL_IO_WORKERS: Final[int] = 1

#: Thread-name prefix for the serial I/O worker, so a stack dump or ``py-spy``
#: capture attributes a blocked ``readline()`` to this driver rather than to an
#: anonymous ``asyncio_N`` thread from the shared default executor.
_SERIAL_IO_THREAD_PREFIX: Final[str] = "mousedroid-serial"


class SerialESP32Driver(BaseESP32Driver):
    """ESP32 driver using serial UART, implementing ``ESP32CommProtocol``.

    All blocking serial I/O is delegated to a per-driver single-worker
    executor via ``_run_io`` (never ``asyncio.to_thread``), so operations
    against the non-thread-safe pyserial handle stay ordered even when the
    awaiting coroutine is cancelled.
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
        # Single-worker executor owning every blocking call against
        # ``self._serial``.  See ``_run_io`` for why this is not the default
        # executor and why it is load-bearing alongside ``_io_lock``.
        # Constructed lazily so a driver that is never connected (mock-mode
        # wiring, unit tests) never spawns a thread.
        self._io_executor: ThreadPoolExecutor | None = None

    async def _run_io(self, fn: Callable[..., _IoResult], *args: Any) -> _IoResult:
        """Run one blocking serial operation on this driver's single I/O thread.

        Replaces ``asyncio.to_thread`` for every call that touches
        ``self._serial``.  Two properties matter, and neither is available from
        the default executor:

        1. **Ordering that survives cancellation.**  ``_io_lock`` gives mutual
           exclusion between coroutines, but cancelling a task parked on an
           executor future releases the ``async with`` *immediately* while the
           OS thread keeps running — so a lock alone lets an ``emergency_stop``
           write reach the port while a ``readline()`` still owns it.  With a
           single worker the write is *queued behind* the in-flight read
           instead, because a cancelled future does not cancel already-
           submitted work.
        2. **Isolation from unrelated blocking work.**  The default executor is
           shared with every other ``asyncio.to_thread`` call in the process;
           a saturated pool would delay motor I/O behind unrelated jobs.

        ``_io_lock`` is still required for *atomicity*: ``_query_data`` submits
        two operations (send, then read) and a foreign write submitted between
        them would be correctly ordered yet still pair the wrong reply with the
        request.  The lock closes that window; the executor closes the
        cancellation window.  Both are load-bearing.

        Args:
            fn: Blocking callable to execute on the I/O thread.
            *args: Positional arguments forwarded to *fn*.

        Returns:
            Whatever *fn* returns.
        """
        if self._io_executor is None:
            self._io_executor = ThreadPoolExecutor(
                max_workers=_SERIAL_IO_WORKERS,
                thread_name_prefix=f"{_SERIAL_IO_THREAD_PREFIX}-{self._port}",
            )
            _log.debug(
                "serial_io_executor_started",
                port=self._port,
                max_workers=_SERIAL_IO_WORKERS,
            )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._io_executor, fn, *args)

    def _shutdown_io_executor(self) -> None:
        """Release the serial I/O thread, if one was ever started.

        ``wait=False`` so shutdown never blocks the event loop: any operation
        still in flight owns a handle that ``_close_serial`` has already
        released, and the worker exits once it returns.
        """
        executor = self._io_executor
        if executor is None:
            return
        self._io_executor = None
        executor.shutdown(wait=False)
        _log.debug("serial_io_executor_stopped", port=self._port)

    async def connect(self) -> None:
        """Open serial connection to ESP32."""
        if _serial_mod is None:
            msg = "pyserial is not installed — install mousedroid[hardware]"
            raise RuntimeError(msg)
        self._serial = await self._run_io(self._open_serial)
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
        except BaseException:
            # BaseException, not Exception: _arm_command_set yields at the
            # executor boundary inside _send_command, and
            # orchestrator.start() awaits connect(). A CancelledError raised
            # in that window derives from BaseException, so an `except
            # Exception` here would skip the rollback and leak exactly the
            # open handle (with _connected=True) this block exists to prevent.
            _log.warning(
                "serial_esp32_arm_failed_rolling_back",
                port=self._port,
                command_set=self._cfg.command_set,
            )
            # Clear driver state SYNCHRONOUSLY, before the next await. The
            # rollback itself has an await in it, and `contextlib.suppress(
            # Exception)` cannot hold a CancelledError delivered there — it is
            # a BaseException. Resetting after that await meant a cancellation
            # landing mid-rollback skipped the reset and the executor shutdown
            # entirely, leaving `_connected=True` over a live handle: the exact
            # leak this block exists to prevent, just through a narrower window.
            #
            # The handle is snapshotted rather than closed via `_close_serial`
            # so the worker thread never races the `self._serial = None` above
            # (that helper dereferences `self._serial` when it runs, not when
            # it is submitted).
            handle, self._serial = self._serial, None
            self._connected = False
            try:
                if handle is not None:
                    # shield() so the close still completes on the I/O thread
                    # when this coroutine is cancelled at the await. It stays
                    # routed through _run_io so it queues behind any read
                    # _arm_command_set left in flight.
                    with contextlib.suppress(Exception):
                        await asyncio.shield(self._run_io(handle.close))
            finally:
                self._shutdown_io_executor()
            raise

    def _open_serial(self) -> Any:  # pragma: no cover
        """Open the serial port (blocking)."""
        return _serial_mod.Serial(
            self._port,
            self._baud,
            timeout=self._timeout,
        )

    async def disconnect(self) -> None:
        """Close serial connection to ESP32.

        The close is submitted through :meth:`_run_io` so it queues behind any
        in-flight read rather than pulling the handle out from under a
        ``readline()`` running on the I/O thread.  The executor is released
        afterwards; a later :meth:`connect` transparently starts a new one.
        """
        if self._serial is not None:
            await self._run_io(self._close_serial)
        self._connected = False
        self._shutdown_io_executor()
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

        Takes ``_io_lock`` for the same reason :meth:`_query_data` does. This
        path was previously unlocked, which meant a ``send_velocity`` or
        ``emergency_stop`` write could land *between* a query's send and its
        read — the reply-mispairing hazard the query's own comment describes,
        reached from the one direction the lock did not cover. Every caller of
        this method (``send_velocity``, ``emergency_stop``, ``_arm_command_set``
        in :class:`BaseESP32Driver`) is a plain write with no reply to consume,
        so holding the lock for a single submit adds no latency beyond waiting
        out an in-flight query.

        Args:
            cmd: Command payload to serialise and send.
        """
        async with self._io_lock:
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
        # leg yields control at its executor boundary — without this lock two
        # ``readline()`` calls interleave on the same pyserial handle, replies
        # pair with the wrong request, and the degraded-mode counters below are
        # mutated from both tasks. ``_send_command`` holds the same lock, so a
        # velocity or emergency-stop write can no longer split this pair.
        #
        # The lock alone is not sufficient: cancelling a task parked on the
        # executor future releases this ``async with`` while the OS thread is
        # still inside ``readline()``. Ordering across that window is provided
        # by the single-worker executor in ``_run_io`` — see its docstring.
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
        await self._run_io(self._write_bytes, payload)

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
        raw = await self._run_io(self._read_line)
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
        out of the ``_run_io`` executor wrapper. The replacement char
        survives into the downstream ``json.loads`` which then emits the
        existing ``esp32_non_json_response`` warning path.

        Returns:
            Decoded line string.
        """
        line: bytes = self._serial.readline()
        return line.decode(errors="replace").strip()
