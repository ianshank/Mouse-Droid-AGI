"""Abstract base class for ESP32 communication drivers.

Both ``SerialESP32Driver`` and ``WiFiESP32Driver`` inherit this base to share
the high-level command/response logic.  Subclasses only implement the
transport layer (serial I/O vs HTTP).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from mousedroid.comms._utils import (
    ESP32_CMD_TYPE_BATTERY,
    ESP32_CMD_TYPE_STOP,
    build_velocity_cmd,
    parse_encoder_reading,
)
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.comms.protocol import EncoderReading
    from mousedroid.config.schema import ESP32Config

_log = get_logger(__name__)


class BaseESP32Driver(ABC):
    """Shared base class for ESP32 drivers implementing ``ESP32CommProtocol``.

    Subclasses provide transport-specific ``connect``, ``disconnect``,
    ``_send_command``, and ``_query_data`` implementations.  All high-level
    protocol methods (``send_velocity``, ``read_encoders``,
    ``get_battery_voltage``, ``emergency_stop``) are implemented here.
    """

    def __init__(self, cfg: ESP32Config) -> None:
        """Initialise shared driver state from config.

        Args:
            cfg: ESP32 communication configuration.
        """
        self._cfg = cfg
        self._timeout: float = cfg.command_timeout_s
        self._connected: bool = False
        self._last_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Abstract transport interface — implemented by each subclass
    # ------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Establish the transport connection to the ESP32."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the transport connection to the ESP32."""

    @abstractmethod
    async def _send_command(self, cmd: dict[str, int]) -> None:
        """Send a fire-and-forget command to the ESP32.

        Args:
            cmd: Command dictionary (e.g. ``{"T": 0}`` for stop).
        """

    @abstractmethod
    async def _query_data(
        self,
        resource: str,
        cmd: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Fetch JSON data from the ESP32.

        For transports that require an explicit request command (e.g. serial),
        ``cmd`` is sent first.  For stateless transports (e.g. HTTP GET), the
        ``resource`` string identifies the endpoint and ``cmd`` is ignored.

        Args:
            resource: Logical resource name (e.g. ``"encoders"``, ``"battery"``).
            cmd: Optional preceding command (used by serial transport).

        Returns:
            Parsed JSON response dictionary.
        """

    # ------------------------------------------------------------------
    # Shared high-level protocol methods
    # ------------------------------------------------------------------

    async def send_velocity(self, vx: float, vy: float, omega: float) -> None:
        """Send velocity command as PWM values.

        Converts physical velocity setpoints to the integer PWM range and
        dispatches the command via the transport layer. Emits the uniform
        ``command_dispatch`` INFO event (via :func:`log_command_dispatch`)
        so operators grepping smoke logs see one consistent record shape
        regardless of which transport (serial / wifi / mock / resilient
        wrapper) fielded the call. The original per-call DEBUG event
        ``esp32_velocity_sent`` is retained for backwards compatibility
        with existing log-greps.

        Args:
            vx: Forward velocity in m/s.
            vy: Lateral velocity in m/s.
            omega: Angular velocity in rad/s.
        """
        cmd = build_velocity_cmd(vx, vy, omega, self._cfg)
        await self._send_command(cmd)
        self._last_velocity = (vx, vy, omega)
        # Uniform INFO-level dispatch event for smoke-triage greps.
        # Uses the concrete subclass name so the resilient wrapper still
        # surfaces the *inner* transport in ``driver=`` for traceability.
        log_command_dispatch(
            driver_name=type(self).__name__,
            vx=vx,
            vy=vy,
            omega=omega,
        )
        # Per-driver DEBUG event preserved (existing log-grep recipes still work).
        _log.debug("esp32_velocity_sent", vx=vx, vy=vy, omega=omega)

    async def read_encoders(self) -> EncoderReading:
        """Read encoder data from ESP32.

        Returns:
            ``EncoderReading`` parsed from the ESP32 response.
        """
        data = await self._query_data("encoders")
        return parse_encoder_reading(data)

    async def get_battery_voltage(self) -> float:
        """Query battery voltage from ESP32 ADC.

        Returns:
            Battery voltage in volts.
        """
        data = await self._query_data("battery", {"T": ESP32_CMD_TYPE_BATTERY})
        return float(data.get("v", 0.0))

    async def emergency_stop(self) -> None:
        """Send emergency stop command and zero stored velocity."""
        await self._send_command({"T": ESP32_CMD_TYPE_STOP})
        self._last_velocity = (0.0, 0.0, 0.0)
        _log.warning("esp32_emergency_stop")


def log_command_dispatch(*, driver_name: str, vx: float, vy: float, omega: float) -> None:
    """Emit a structured ``command_dispatch`` DEBUG event used by smoke triage.

    Centralised so every driver (Serial, WiFi, Mock, Resilient) can emit the
    same event shape — operators grepping smoke logs for ``command_dispatch``
    get a uniform record regardless of which driver fielded the call.

    Emitted at DEBUG (not INFO) because ``BaseESP32Driver.send_velocity``
    AND ``MockESP32Driver.send_velocity`` invoke this on every orchestrator
    tick (30 Hz). An INFO emission rate of 30/s saturates Windows / Linux
    structlog processors enough to push the 5-second e2e orchestrator
    integration test past its pytest-timeout on Python 3.10 (CI run
    26718393212 — passed on 3.11/3.12 with more headroom, failed on
    3.10). Operators wanting to see the events during a smoke run set
    ``LOG_LEVEL=DEBUG`` or filter the structured stream — the legacy
    per-driver events (``esp32_velocity_sent`` / ``mock_velocity_sent``)
    were already DEBUG for the same reason.
    """
    _log.debug("command_dispatch", driver=driver_name, vx=vx, vy=vy, omega=omega)
