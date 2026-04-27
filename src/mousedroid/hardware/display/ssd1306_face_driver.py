"""SSD1306 128x64 I²C OLED driver rendering MSE-6 facial expressions.

All blocking I²C work is bridged through :func:`asyncio.to_thread` so the
30 Hz orchestrator loop is never stalled. A background blink task adds a
lifelike idle animation; it is cancelled cleanly on :meth:`stop`. Concurrent
state changes (explicit ``show_expression`` vs. blink loop) are serialised
through a single :class:`asyncio.Lock`.

The ``luma.oled`` and ``smbus2`` imports happen lazily inside
:meth:`start` so that the unit-test environment can patch them without
having the libraries installed.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

from mousedroid.hardware.display.expressions import (
    Expression,
    render_expression,
    render_text,
)
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import FaceDisplayConfig

_log = get_logger(__name__)


class SSD1306FaceDriver:
    """Hardware driver for the SSD1306 OLED expression panel.

    Implements :class:`FaceDisplayProtocol`. Construction is cheap; all
    device probing and resource allocation happens in :meth:`start`.
    """

    def __init__(self, cfg: FaceDisplayConfig) -> None:
        """Initialise the driver with a validated config.

        Args:
            cfg: Validated face-display config.
        """
        self._cfg = cfg
        self._device: Any | None = None
        self._lock = asyncio.Lock()
        self._current: Expression | None = None
        self._blink_task: asyncio.Task[None] | None = None
        self._started = False
        _log.info(
            "ssd1306_face_init",
            i2c_bus=cfg.i2c_bus,
            i2c_address=cfg.i2c_address,
            width=cfg.width,
            height=cfg.height,
        )

    @property
    def started(self) -> bool:
        """Whether the device has been initialised."""
        return self._started

    @classmethod
    def probe(cls, cfg: FaceDisplayConfig) -> None:
        """Eagerly probe the configured I²C bus + address.

        Used by the factory to surface missing libraries, missing
        ``/dev/i2c-N`` device nodes, and dead panels before the orchestrator
        starts. Raises the original :class:`ImportError` / :class:`OSError`
        when the probe cannot be completed; the factory then decides whether
        to fall back to the mock based on
        :attr:`FaceDisplayConfig.fallback_to_mock_on_error`.

        Args:
            cfg: Validated face-display config.
        """
        from smbus2 import SMBus

        _log.debug(
            "face_display_probe_start",
            i2c_bus=cfg.i2c_bus,
            i2c_address=cfg.i2c_address,
        )
        try:
            with SMBus(cfg.i2c_bus) as bus:
                bus.read_byte(cfg.i2c_address)
        except OSError as exc:
            # Add an actionable hint that distinguishes "no responder on the
            # configured address" from "bus node missing" without changing
            # the exception type the factory checks against.
            responders = SSD1306FaceDriver._scan_i2c_bus(cfg.i2c_bus)
            hint_parts = [
                f"errno={exc.errno}",
                f"bus={cfg.i2c_bus}",
                f"addr=0x{cfg.i2c_address:02x}",
            ]
            if responders is None:
                hint_parts.append("bus node missing or inaccessible")
            elif not responders:
                hint_parts.append(
                    "no I2C responders on this bus; check OLED power, "
                    "ribbon orientation, and that the device is wired to "
                    "the configured bus"
                )
            else:
                addrs = ", ".join(f"0x{a:02x}" for a in responders)
                hint_parts.append(
                    f"responders found at: {addrs}; update "
                    f"face_display.i2c_address or i2c_bus in config"
                )
            raise OSError(exc.errno, f"{exc.strerror or exc} :: {' | '.join(hint_parts)}") from exc
        _log.debug(
            "face_display_probe_success",
            i2c_bus=cfg.i2c_bus,
            i2c_address=cfg.i2c_address,
        )

    @staticmethod
    def _scan_i2c_bus(bus_id: int) -> list[int] | None:
        """Return the list of responding 7-bit addresses on ``bus_id``.

        Returns ``None`` when the bus device node cannot be opened. The scan
        skips reserved address ranges (0x00-0x02 and 0x78-0x7F) and is best
        effort: any per-address read error is treated as "no responder".
        """
        from smbus2 import SMBus

        try:
            bus_ctx = SMBus(bus_id)
        except (FileNotFoundError, PermissionError, OSError):
            return None
        responders: list[int] = []
        with bus_ctx as bus:
            for addr in range(0x03, 0x78):
                try:
                    bus.read_byte(addr)
                except OSError:
                    continue
                responders.append(addr)
        return responders

    async def start(self) -> None:
        """Probe the I²C bus and initialise the SSD1306 device."""
        await asyncio.to_thread(type(self).probe, self._cfg)
        self._device = await asyncio.to_thread(self._init_device)
        self._started = True
        _log.info(
            "face_display_started",
            i2c_bus=self._cfg.i2c_bus,
            i2c_address=self._cfg.i2c_address,
            width=self._cfg.width,
            height=self._cfg.height,
        )
        await self.show_text(self._cfg.boot_message)
        if self._cfg.idle_blink_interval_s > 0.0:
            self._blink_task = asyncio.create_task(self._blink_loop())

    async def stop(self) -> None:
        """Cancel the blink task, clear the panel, and release the device."""
        if not self._started:
            return
        if self._blink_task is not None:
            self._blink_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._blink_task
            self._blink_task = None
        async with self._lock:
            if self._device is not None:
                await asyncio.to_thread(self._device.clear)
            self._device = None
        self._started = False
        _log.info("face_display_stopped")

    async def show_expression(self, expression: Expression) -> None:
        """Render an expression to the panel under the lock."""
        async with self._lock:
            await self._render_expression(expression)

    async def show_text(self, message: str) -> None:
        """Render a short status message to the panel under the lock."""
        async with self._lock:
            if self._device is None:
                return
            frame = render_text(message, self._cfg.width, self._cfg.height)
            await asyncio.to_thread(self._device.display, frame)
        _log.debug("face_display_text_rendered", message=message)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _render_expression(self, expression: Expression) -> None:
        if self._device is None:
            return
        prev = self._current
        frame = render_expression(expression, self._cfg.width, self._cfg.height)
        await asyncio.to_thread(self._device.display, frame)
        self._current = expression
        if prev != expression:
            _log.info(
                "face_expression_changed",
                old=prev.value if prev is not None else None,
                new=expression.value,
            )

    async def _blink_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._cfg.idle_blink_interval_s)
                async with self._lock:
                    previous = self._current
                    if previous is None or previous is Expression.EMERGENCY:
                        continue
                    await self._render_expression(Expression.BLINK)
                await asyncio.sleep(self._cfg.blink_close_duration_s)
                async with self._lock:
                    # Skip restore if a concurrent show_expression replaced
                    # _current while the eyes were closed.
                    if self._current is Expression.BLINK:
                        await self._render_expression(previous)
        except asyncio.CancelledError:
            return

    def _init_device(self) -> Any:
        from luma.core.interface.serial import i2c
        from luma.oled.device import ssd1306

        serial = i2c(port=self._cfg.i2c_bus, address=self._cfg.i2c_address)
        return ssd1306(
            serial,
            width=self._cfg.width,
            height=self._cfg.height,
            rotate=self._cfg.rotate,
        )
