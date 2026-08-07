"""Abstract base class for ESP32 communication drivers.

Both ``SerialESP32Driver`` and ``WiFiESP32Driver`` inherit this base to share
the high-level command/response logic.  Subclasses only implement the
transport layer (serial I/O vs HTTP).

Command building and response parsing are delegated to the command-set
codec selected by ``cfg.command_set`` (F-025 —
:mod:`mousedroid.comms.command_set`), so this class stays firmware-agnostic:
the default ``legacy`` codec reproduces the historical private protocol
byte-for-byte, while ``waveshare_stock`` speaks stock ``General_Driver``
firmware.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from mousedroid.comms.command_set import heartbeat_window_ms, resolve_command_codec
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mousedroid.comms.command_set import ESP32CommandCodec
    from mousedroid.comms.protocol import EncoderReading
    from mousedroid.config.schema import ESP32Config

_log = get_logger(__name__)


class BaseESP32Driver(ABC):
    """Shared base class for ESP32 drivers implementing ``ESP32CommProtocol``.

    Subclasses provide transport-specific ``connect``, ``disconnect``,
    ``_send_command``, and ``_query_data`` implementations.  All high-level
    protocol methods (``send_velocity``, ``read_encoders``,
    ``get_battery_voltage``, ``emergency_stop``) are implemented here and
    delegate command shapes to the resolved command-set codec.
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
        # F-025: codec resolved once from cfg.command_set — "legacy" default
        # reproduces the pre-selector protocol byte-for-byte.
        self._codec: ESP32CommandCodec = resolve_command_codec(cfg)
        # Once-per-connection latch for the lateral-velocity warning. Lives
        # on the driver (codecs are stateless shared singletons) and is
        # re-armed by _arm_command_set() on every successful connect, so
        # resilience-wrapper reconnects surface the warning again.
        self._lateral_warn_emitted: bool = False

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
    async def _send_command(self, cmd: Mapping[str, float]) -> None:
        """Send a fire-and-forget command to the ESP32.

        Args:
            cmd: Command payload (e.g. ``{"T": 0}`` for the legacy stop).
                Read-only mapping so codecs may return ``dict[str, int]``
                (legacy PWM) or ``dict[str, float]`` (stock physical units).
        """

    @abstractmethod
    async def _query_data(
        self,
        resource: str,
        cmd: Mapping[str, float] | None = None,
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
        """Send a velocity command via the selected command-set codec.

        Under the ``legacy`` codec this is the historical PWM-scaled dict;
        under ``waveshare_stock`` it is ``CMD_ROS_CTRL`` in physical units.
        Emits the uniform ``command_dispatch`` DEBUG event (via
        :func:`log_command_dispatch`) so operators grepping smoke logs see
        one consistent record shape regardless of which transport (serial /
        wifi / mock / resilient wrapper) fielded the call. The original
        per-call DEBUG event ``esp32_velocity_sent`` is retained for
        backwards compatibility with existing log-greps.

        Args:
            vx: Forward velocity in m/s.
            vy: Lateral velocity in m/s.
            omega: Angular velocity in rad/s.
        """
        if vy != 0.0 and not self._codec.supports_lateral:
            self._warn_lateral_unsupported(vy)
        cmd = self._codec.build_velocity(vx, vy, omega, self._cfg)
        await self._send_command(cmd)
        self._last_velocity = (vx, vy, omega)
        # Uniform DEBUG-level dispatch event for smoke-triage greps.
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

        The codec supplies both the optional poll command (stock firmware
        must be polled with ``CMD_BASE_FEEDBACK``; legacy sends nothing)
        and the response parser.

        Returns:
            ``EncoderReading`` parsed from the ESP32 response.
        """
        data = await self._query_data("encoders", self._codec.encoder_query())
        return self._codec.parse_encoders(data)

    async def get_battery_voltage(self) -> float:
        """Query battery voltage from the ESP32.

        Under ``legacy`` this sends the historical ``{"T":2}`` poll; under
        ``waveshare_stock`` it polls ``CMD_BASE_FEEDBACK`` (a read — the
        legacy poll is stock ``CMD_SET_MOTOR_PID``, a motor-controller
        write) and parses the ``FEEDBACK_BASE_INFO`` frame.

        Returns:
            Battery voltage in volts (0.0 on timeout / unparseable frame).
        """
        data = await self._query_data("battery", self._codec.battery_query())
        return self._codec.parse_battery(data)

    async def emergency_stop(self) -> None:
        """Send the strongest stop the firmware understands; zero velocity."""
        await self._send_command(self._codec.build_stop())
        self._last_velocity = (0.0, 0.0, 0.0)
        _log.warning("esp32_emergency_stop")

    # ------------------------------------------------------------------
    # Command-set arming (called by transports at the end of connect())
    # ------------------------------------------------------------------

    async def _arm_command_set(self) -> None:
        """Send the codec's connect-time commands and reset warn latches.

        Under ``waveshare_stock`` this arms the chassis heartbeat failsafe
        (``CMD_HEART_BEAT_SET``) so the firmware halts the motors on its own
        if the host wedges; under ``legacy`` the command list is empty and
        the connect sequence is byte-identical to pre-F-025. Also re-arms
        the once-per-connection lateral warning so a reconnect surfaces it
        again.
        """
        self._lateral_warn_emitted = False
        commands = self._codec.connect_commands(self._cfg)
        for cmd in commands:
            await self._send_command(cmd)
        if commands:
            _log.info(
                "esp32_heartbeat_armed",
                command_set=self._cfg.command_set,
                window_ms=heartbeat_window_ms(self._cfg),
            )

    def _warn_lateral_unsupported(self, vy: float) -> None:
        """Surface a dropped lateral setpoint — WARNING once, DEBUG after.

        ``send_velocity`` runs on the 30 Hz control path, so only the first
        occurrence per connection logs at WARNING (the
        :func:`log_command_dispatch` rate rationale); subsequent drops leave
        a DEBUG breadcrumb for smoke triage.
        """
        if self._lateral_warn_emitted:
            _log.debug("esp32_lateral_velocity_unsupported", vy=vy)
            return
        self._lateral_warn_emitted = True
        _log.warning(
            "esp32_lateral_velocity_unsupported",
            vy=vy,
            command_set=self._cfg.command_set,
        )


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
