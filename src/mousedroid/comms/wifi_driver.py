"""WiFi (HTTP) ESP32 communication driver for Wave Rover motor control.

Implements ``ESP32CommProtocol`` over HTTP using only stdlib ``urllib``.
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import TYPE_CHECKING, Any, cast

from mousedroid.comms.base_driver import BaseESP32Driver
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

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
        # F-025: no-op under legacy (empty command list); the stock command
        # set cannot reach this transport (schema validator rejects
        # command_set='waveshare_stock' + protocol='wifi' at load time).
        await self._arm_command_set()

    async def disconnect(self) -> None:
        """Mark connection as closed."""
        self._connected = False
        _log.info("wifi_esp32_disconnected")

    # ------------------------------------------------------------------
    # Transport implementation
    # ------------------------------------------------------------------

    async def _send_command(self, cmd: Mapping[str, float]) -> None:
        """POST a JSON command to the ESP32 ``/cmd`` endpoint.

        Args:
            cmd: Command payload to serialise and POST.
        """
        await self._post_json("/cmd", cmd)

    async def _query_data(
        self,
        resource: str,
        cmd: Mapping[str, float] | None = None,
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

    async def _post_json(self, path: str, data: Mapping[str, Any]) -> dict[str, Any]:
        """HTTP POST JSON to ESP32.

        Args:
            path: URL path.
            data: Payload mapping to send as JSON body.

        Returns:
            Parsed JSON response.
        """
        return await asyncio.to_thread(self._blocking_post, path, data)

    def _blocking_post(
        self, path: str, data: Mapping[str, Any]
    ) -> dict[str, Any]:  # pragma: no cover
        """Perform blocking HTTP POST.

        Args:
            path: URL path.
            data: Payload mapping to send as JSON body.

        Returns:
            Parsed JSON response.
        """
        url = f"{self._base_url}{path}"
        payload = json.dumps(dict(data)).encode()
        req = urllib.request.Request(  # noqa: S310 — scheme is a fixed http:// literal (see _base_url)
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310 — fixed http:// scheme
            # errors="replace": a garbled byte (firmware churn / brown-out) must
            # never raise UnicodeDecodeError out of the asyncio.to_thread wrapper;
            # the replacement char flows into the JSON guard below. Mirrors the
            # serial driver's decode hygiene.
            body = resp.read().decode(errors="replace")
        return self._decode_json_object(body, path=path)

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
            # errors="replace": see _blocking_post — never raise out of to_thread.
            body = resp.read().decode(errors="replace")
        return self._decode_json_object(body, path=path)

    def _decode_json_object(self, body: str, *, path: str) -> dict[str, Any]:
        """Parse ``body`` as a JSON object, guarding against degraded payloads.

        The ESP32 firmware is contracted to return JSON objects, but a brown-out
        or firmware-churn response could be (a) non-JSON bytes (truncated frame,
        an HTTP error page, UART noise) or (b) a syntactically-valid but
        non-mapping shape (bare list/string/number). Both are degraded conditions
        that must NOT surface as a raw ``json.JSONDecodeError`` out of the
        ``asyncio.to_thread`` wrapper, nor as a non-mapping returned to callers
        expecting ``dict`` semantics (which would manifest later as a confusing
        ``AttributeError``/``KeyError``). Instead we log a structured warning and
        return an empty mapping (the same shape used for an empty body) so the
        protocol layer degrades gracefully — mirroring the serial driver's
        ``esp32_non_json_response`` / ``esp32_response_not_object`` contract.

        Args:
            body: Raw decoded HTTP response body.
            path: URL path the body came from (for log context).

        Returns:
            The parsed JSON object, or ``{}`` for an empty/malformed/non-object
            payload.
        """
        if not body.strip():
            return {}
        truncate = self._cfg.debug_log_max_chars
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            _log.warning(
                "wifi_esp32_non_json_response",
                path=path,
                body=body[:truncate],
                error=str(exc),
            )
            return {}
        if isinstance(decoded, dict):
            return cast("dict[str, Any]", decoded)
        _log.warning(
            "wifi_esp32_unexpected_json_shape",
            path=path,
            shape=type(decoded).__name__,
        )
        return {}
