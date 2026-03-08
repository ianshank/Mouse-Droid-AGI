"""Serial (UART) ESP32 communication driver for Wave Rover motor control.

Implements ``ESP32CommProtocol`` over a serial connection to the ESP32.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from mousedroid.comms._utils import clamp as _clamp
from mousedroid.comms.protocol import EncoderReading
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ESP32Config

try:
    import serial as _serial_mod
except ImportError:  # pragma: no cover
    _serial_mod = None

_log = get_logger(__name__)

_ESP32_CMD_TYPE_VELOCITY: int = 1
_ESP32_CMD_TYPE_STOP: int = 0
_MAX_PWM: int = 255


class SerialESP32Driver:
    """ESP32 driver using serial UART, implementing ``ESP32CommProtocol``.

    All blocking serial I/O is delegated to ``asyncio.to_thread``.
    """

    def __init__(self, cfg: ESP32Config) -> None:
        """Initialise serial driver from config.

        Args:
            cfg: ESP32 communication configuration.
        """
        self._cfg = cfg
        self._port: str = cfg.serial_port
        self._baud: int = cfg.serial_baud
        self._timeout: float = cfg.command_timeout_s
        self._serial: Any = None
        self._connected: bool = False
        self._last_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

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

    async def send_velocity(self, vx: float, vy: float, omega: float) -> None:
        """Send velocity command as PWM values over serial.

        Args:
            vx: Forward velocity in m/s.
            vy: Lateral velocity in m/s.
            omega: Angular velocity in rad/s.
        """
        max_vel = self._cfg.max_velocity_mps
        pwm_vx = int(_clamp(vx / max_vel, -1.0, 1.0) * _MAX_PWM)
        pwm_vy = int(_clamp(vy / max_vel, -1.0, 1.0) * _MAX_PWM)
        pwm_omega = int(_clamp(omega / self._cfg.max_omega_rads, -1.0, 1.0) * _MAX_PWM)
        cmd: dict[str, int] = {
            "T": _ESP32_CMD_TYPE_VELOCITY,
            "vx": pwm_vx,
            "vy": pwm_vy,
            "omega": pwm_omega,
        }
        await self._send_json(cmd)
        self._last_velocity = (vx, vy, omega)
        _log.debug("serial_velocity_sent", vx=vx, vy=vy, omega=omega)

    async def read_encoders(self) -> EncoderReading:
        """Read encoder data as JSON from serial.

        Returns:
            ``EncoderReading`` parsed from ESP32 response.
        """
        data = await self._read_json()
        return EncoderReading(
            left_velocity_mps=float(data.get("lv", 0.0)),
            right_velocity_mps=float(data.get("rv", 0.0)),
            odometry_x_m=float(data.get("ox", 0.0)),
            odometry_y_m=float(data.get("oy", 0.0)),
            heading_rad=float(data.get("h", 0.0)),
            timestamp=float(data.get("ts", 0.0)),
        )

    async def get_battery_voltage(self) -> float:
        """Query battery voltage from ESP32 ADC.

        Returns:
            Battery voltage in volts.
        """
        cmd: dict[str, int] = {"T": 2}
        await self._send_json(cmd)
        data = await self._read_json()
        return float(data.get("v", 0.0))

    async def emergency_stop(self) -> None:
        """Send emergency stop command over serial."""
        cmd: dict[str, int] = {"T": _ESP32_CMD_TYPE_STOP}
        await self._send_json(cmd)
        self._last_velocity = (0.0, 0.0, 0.0)
        _log.warning("serial_emergency_stop")

    # ------------------------------------------------------------------
    # Internal helpers
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
