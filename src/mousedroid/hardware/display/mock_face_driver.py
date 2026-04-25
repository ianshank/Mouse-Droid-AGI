"""In-memory mock face display for CI and ``mock_hardware`` mode.

Implements :class:`FaceDisplayProtocol` and records the full sequence of
expressions and text frames it has been asked to render so that tests can
assert against the call history without standing up a real I²C device.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from mousedroid.hardware.display.expressions import Expression
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import FaceDisplayConfig

_log = get_logger(__name__)


class MockFaceDriver:
    """Minimal in-memory face driver implementing :class:`FaceDisplayProtocol`.

    The mock skips the blink animation by default to keep tests deterministic,
    but exposes :meth:`tick_blink` for tests that want to drive the blink path
    explicitly.
    """

    def __init__(self, cfg: FaceDisplayConfig) -> None:
        """Initialise the mock driver.

        Args:
            cfg: Validated face-display config (used only to read panel size).
        """
        self._cfg = cfg
        self._lock = asyncio.Lock()
        self.expressions: list[Expression] = []
        self.texts: list[str] = []
        self.history: list[str] = []
        self.current: Expression | None = None
        self._started = False
        _log.info(
            "mock_face_init",
            width=cfg.width,
            height=cfg.height,
        )

    @property
    def started(self) -> bool:
        """Whether ``start()`` has been called and ``stop()`` has not."""
        return self._started

    async def start(self) -> None:
        """Mark the mock display as started and record the lifecycle event."""
        self._started = True
        self.history.append("start")
        _log.info("mock_face_started")

    async def stop(self) -> None:
        """Mark the mock display as stopped (idempotent)."""
        if not self._started:
            return
        self._started = False
        self.history.append("stop")
        _log.info("mock_face_stopped")

    async def show_expression(self, expression: Expression) -> None:
        """Record the requested expression and update :attr:`current`."""
        async with self._lock:
            self.current = expression
            self.expressions.append(expression)
            self.history.append(f"expr:{expression.value}")
        _log.debug("mock_face_expression", expression=expression.value)

    async def show_text(self, message: str) -> None:
        """Record the requested text frame."""
        async with self._lock:
            self.texts.append(message)
            self.history.append(f"text:{message}")
        _log.debug("mock_face_text", message=message)

    async def tick_blink(self) -> None:
        """Drive a blink-open-restore cycle on demand for tests."""
        previous = self.current
        await self.show_expression(Expression.BLINK)
        if previous is not None:
            await self.show_expression(previous)
