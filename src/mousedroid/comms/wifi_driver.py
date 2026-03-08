"""WiFi (HTTP) ESP32 communication driver for Wave Rover motor control.

Implements ``ESP32CommProtocol`` over HTTP using only stdlib ``urllib``.
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import TYPE_CHECKING, Any

from mousedroid.comms.protocol import EncoderReading
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ESP32Config

_log = get_logger(__name__)

_ESP32_CMD_TYPE_VELOCITY: int = 1
_ESP32_CMD_TYPE_STOP: int = 0
_MAX_PWM: int = 255


class WiFiESP32Driver:
    """ESP32 driver using WiFi HTTP, implementing ``ESP32CommProtocol``.

    All blocking HTTP I/O is delegated to ``asyncio.to_thread``.
    """

    def __init__(self, cfg: ESP32Config) -> None:
        """Initialise WiFi driver from config.

        Args:
            cfg: ESP32 communication configuration.
        """
        self._cfg = cfg
        self._base_url: str = f"http://{cfg.wifi_host}:{cfg.wifi_port}"
        self._timeout: float = cfg.command_timeout_s
        self._connected: bool = False
        self._last_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    async def connect(self) -> None:
        """Mark connection as established (HTTP is stateless)."""
        self._connected = True
        _log.info("wifi_esp32_connected", base_url=self._base_url)

    async def disconnect(self) -> None:
        """Mark connection as closed."""
        self._connected = False
        _log.info("wifi_esp32_disconnected")

    async def send_velocity(self, vx: float, vy: float, omega: float) -> None:
        """Send velocity command as PWM values over HTTP POST.

        Args:
            vx: Forward velocity in m/s.
            vy: Lateral velocity in m/s.
            omega: Angular velocity in rad/s.
        """
        max_vel = self._cfg.max_velocity_mps
        pwm_vx = int(_clamp(vx / max_vel, -1.0, 1.0) * _MAX_PWM)
        pwm_vy = int(_clamp(vy / max_vel, -1.0, 1.0) * _MAX_PWM)
        pwm_omega = int(
            _clamp(omega / self._cfg.max_omega_rads, -1.0, 1.0) * _MAX_PWM
        )
        cmd: dict[str, int] = {
            "T": _ESP32_CMD_TYPE_VELOCITY,
            "vx": pwm_vx,
            "vy": pwm_vy,
            "omega": pwm_omega,
        }
        await self._post_json("/cmd", cmd)
        self._last_velocity = (vx, vy, omega)
        _log.debug("wifi_velocity_sent", vx=vx, vy=vy, omega=omega)

    async def read_encoders(self) -> EncoderReading:
        """Read encoder data via HTTP GET.

        Returns:
            ``EncoderReading`` parsed from ESP32 response.
        """
        data = await self._get_json("/encoders")
        return EncoderReading(
            left_velocity_mps=float(data.get("lv", 0.0)),
            right_velocity_mps=float(data.get("rv", 0.0)),
            odometry_x_m=float(data.get("ox", 0.0)),
            odometry_y_m=float(data.get("oy", 0.0)),
            heading_rad=float(data.get("h", 0.0)),
            timestamp=float(data.get("ts", 0.0)),
        )

    async def get_battery_voltage(self) -> float:
        """Query battery voltage via HTTP GET.

        Returns:
            Battery voltage in volts.
        """
        data = await self._get_json("/battery")
        return float(data.get("v", 0.0))

    async def emergency_stop(self) -> None:
        """Send emergency stop command over HTTP POST."""
        cmd: dict[str, int] = {"T": _ESP32_CMD_TYPE_STOP}
        await self._post_json("/cmd", cmd)
        self._last_velocity = (0.0, 0.0, 0.0)
        _log.warning("wifi_emergency_stop")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post_json(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """HTTP POST JSON to ESP32.

        Args:
            path: URL path.
            data: Dictionary to send as JSON body.

        Returns:
            Parsed JSON response.
        """
        return await asyncio.to_thread(self._blocking_post, path, data)

    def _blocking_post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        """Perform blocking HTTP POST.

        Args:
            path: URL path.
            data: Dictionary to send as JSON body.

        Returns:
            Parsed JSON response.
        """
        url = f"{self._base_url}{path}"
        payload = json.dumps(data).encode()
        req = urllib.request.Request(  # noqa: S310
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
            body = resp.read().decode()
        if not body.strip():
            return {}
        return json.loads(body)  # type: ignore[no-any-return]

    async def _get_json(self, path: str) -> dict[str, Any]:
        """HTTP GET JSON from ESP32.

        Args:
            path: URL path.

        Returns:
            Parsed JSON response.
        """
        return await asyncio.to_thread(self._blocking_get, path)

    def _blocking_get(self, path: str) -> dict[str, Any]:  # pragma: no cover
        """Perform blocking HTTP GET.

        Args:
            path: URL path.

        Returns:
            Parsed JSON response.
        """
        url = f"{self._base_url}{path}"
        req = urllib.request.Request(url, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
            body = resp.read().decode()
        if not body.strip():
            return {}
        return json.loads(body)  # type: ignore[no-any-return]


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value between lo and hi.

    Args:
        value: Value to clamp.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Clamped value.
    """
    return max(lo, min(hi, value))
