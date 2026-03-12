"""Serial (UART) ESP32 communication driver for Wave Rover motor control.

Implements ``ESP32CommProtocol`` over a serial connection to the ESP32.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from mousedroid.comms.base_driver import BaseESP32Driver
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
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

    async def connect(self) -> None:
        """Open serial connection to ESP32."""
        if _serial_mod is None:
            msg = "pyserial is not installed — install mousedroid[hardware]"
            raise RuntimeError(msg)
        self._serial = await asyncio.to_thread(self._open_serial)
        self._connected = True
        _log.info("serial_esp32_connected", port=self._port, baud=self._baud)

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

    async def _send_command(self, cmd: dict[str, int]) -> None:
        """Write a JSON command to the serial port.

        Args:
            cmd: Command dictionary to serialise and send.
        """
        await self._send_json(cmd)

    async def _query_data(
        self,
        resource: str,
        cmd: dict[str, int] | None = None,
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
        if cmd is not None:
            await self._send_json(cmd)
        return await self._read_json()

    # ------------------------------------------------------------------
    # Low-level serial helpers
    # ------------------------------------------------------------------

    async def _send_json(self, data: dict[str, Any]) -> None:
        """Serialise and send JSON over serial.

        Args:
            data: Dictionary to send as JSON.
        """
        payload = json.dumps(data).encode() + b"\n"
        await asyncio.to_thread(self._write_bytes, payload)

    def _write_bytes(self, payload: bytes) -> None:  # pragma: no cover
        """Write raw bytes to serial port (blocking).

        Args:
            payload: Bytes to write.
        """
        self._serial.write(payload)

    async def _read_json(self) -> dict[str, Any]:
        """Read one JSON line from serial.

        Returns:
            Parsed JSON dictionary.
        """
        raw = await asyncio.to_thread(self._read_line)
        if not raw:
            return {}
        return json.loads(raw)  # type: ignore[no-any-return]

    def _read_line(self) -> str:  # pragma: no cover
        """Read one line from serial port (blocking).

        Returns:
            Decoded line string.
        """
        line: bytes = self._serial.readline()
        return line.decode().strip()
