"""WiFi (HTTP) ESP32 communication driver for Wave Rover motor control.

Implements ``ESP32CommProtocol`` over HTTP using only stdlib ``urllib``.
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import TYPE_CHECKING, Any

from mousedroid.comms.base_driver import BaseESP32Driver
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ESP32Config

_log = get_logger(__name__)


class WiFiESP32Driver(BaseESP32Driver):
    """ESP32 driver using WiFi HTTP, implementing ``ESP32CommProtocol``.

    All blocking HTTP I/O is delegated to ``asyncio.to_thread``.
    High-level protocol methods (``send_velocity``, ``read_encoders``,
    ``get_battery_voltage``, ``emergency_stop``) are inherited from
    ``BaseESP32Driver``.
    """

    def __init__(self, cfg: ESP32Config) -> None:
        """Initialise WiFi driver from config.

        Args:
            cfg: ESP32 communication configuration.
        """
        super().__init__(cfg)
        self._base_url: str = f"http://{cfg.wifi_host}:{cfg.wifi_port}"

    async def connect(self) -> None:
        """Mark connection as established (HTTP is stateless)."""
        self._connected = True
        _log.info("wifi_esp32_connected", base_url=self._base_url)

    async def disconnect(self) -> None:
        """Mark connection as closed."""
        self._connected = False
        _log.info("wifi_esp32_disconnected")

    # ------------------------------------------------------------------
    # Transport implementation
    # ------------------------------------------------------------------

    async def _send_command(self, cmd: dict[str, int]) -> None:
        """POST a JSON command to the ESP32 ``/cmd`` endpoint.

        Args:
            cmd: Command dictionary to serialise and POST.
        """
        await self._post_json("/cmd", cmd)

    async def _query_data(
        self,
        resource: str,
        cmd: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """GET JSON data from an ESP32 endpoint.

        The ``cmd`` argument is accepted for interface compatibility but is
        unused — HTTP is stateless so each resource has its own endpoint.

        Args:
            resource: Endpoint path component (e.g. ``"encoders"``,
                ``"battery"``).
            cmd: Ignored for HTTP transport.

        Returns:
            Parsed JSON response dictionary.
        """
        return await self._get_json(f"/{resource}")

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
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
        req = urllib.request.Request(  # noqa: S310 — scheme is a fixed http:// literal (see _base_url)
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310 — fixed http:// scheme
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
        req = urllib.request.Request(url, method="GET")  # noqa: S310 — fixed http:// scheme (see _base_url)
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310 — fixed http:// scheme
            body = resp.read().decode()
        if not body.strip():
            return {}
        return json.loads(body)  # type: ignore[no-any-return]
