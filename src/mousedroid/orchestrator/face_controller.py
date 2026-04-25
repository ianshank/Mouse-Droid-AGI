"""Face controller — maps cognitive affect + safety state to expressions.

The orchestrator calls :meth:`FaceController.update` once per tick with the
BDI affect tuple and the current :class:`SafetyContext.is_emergency` flag.
The controller applies a configurable precedence policy and a minimum
dwell time (hysteresis) before delegating to the underlying
:class:`FaceDisplayProtocol`.

All thresholds come from :class:`FaceDisplayConfig`; this module contains
no magic numbers.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from mousedroid.hardware.display.expressions import Expression
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import FaceDisplayConfig
    from mousedroid.hardware.protocols import FaceDisplayProtocol

_log = get_logger(__name__)


class FaceController:
    """Drive a :class:`FaceDisplayProtocol` from BDI affect + safety state."""

    def __init__(
        self,
        driver: FaceDisplayProtocol,
        cfg: FaceDisplayConfig,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Initialise the controller.

        Args:
            driver: Concrete face-display implementation.
            cfg: Validated face-display config (provides every threshold).
            clock: Monotonic-time source (override-able for tests).
        """
        self._driver = driver
        self._cfg = cfg
        self._clock = clock if clock is not None else time.monotonic
        self._last_expr: Expression | None = None
        self._last_change_ts: float = -float("inf")
        self._last_update_ts: float = -float("inf")
        self._idle_started_ts: float | None = None

    @property
    def current(self) -> Expression | None:
        """Most-recently-rendered expression (``None`` before first update)."""
        return self._last_expr

    async def start(self) -> None:
        """Start the underlying driver and reset hysteresis state."""
        self._last_expr = None
        self._last_change_ts = -float("inf")
        self._last_update_ts = -float("inf")
        self._idle_started_ts = None
        await self._driver.start()
        await self._driver.show_expression(Expression.NEUTRAL)
        self._last_expr = Expression.NEUTRAL
        self._last_change_ts = self._clock()
        _log.info("face_controller_started")

    async def stop(self) -> None:
        """Stop the underlying driver."""
        await self._driver.stop()
        _log.info("face_controller_stopped")

    async def update(
        self,
        valence: float,
        arousal: float,
        is_emergency: bool,
        is_idle: bool,
    ) -> None:
        """Map affect + safety state to an :class:`Expression` and render it.

        Args:
            valence: BDI affect valence in ``[-1, 1]``.
            arousal: BDI affect arousal in ``[-1, 1]``.
            is_emergency: Whether the safety monitor has signalled a halt.
            is_idle: Whether the agent is currently idle (no commanded motion).
        """
        now = self._clock()
        self._track_idle(now=now, is_idle=is_idle)
        candidate = self._classify(
            valence=valence,
            arousal=arousal,
            is_emergency=is_emergency,
            now=now,
        )
        # Throttle non-emergency updates to cfg.refresh_hz so the I²C bus and
        # log volume are bounded even when the orchestrator ticks faster than
        # the configured refresh rate.  Emergency always bypasses the throttle.
        if candidate is not Expression.EMERGENCY:
            min_interval = 1.0 / self._cfg.refresh_hz
            if now - self._last_update_ts < min_interval:
                _log.debug(
                    "face_update_throttled",
                    refresh_hz=self._cfg.refresh_hz,
                    candidate=candidate.value,
                )
                return
        self._last_update_ts = now
        target = self._apply_dwell(candidate=candidate, now=now)
        if target == self._last_expr:
            _log.debug(
                "face_tick_update",
                valence=valence,
                arousal=arousal,
                emergency=is_emergency,
                expression=target.value,
                changed=False,
            )
            return

        await self._driver.show_expression(target)
        _log.info(
            "face_expression_changed",
            old=self._last_expr.value if self._last_expr is not None else None,
            new=target.value,
            valence=valence,
            arousal=arousal,
        )
        self._last_expr = target
        self._last_change_ts = now

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _track_idle(self, *, now: float, is_idle: bool) -> None:
        if is_idle:
            if self._idle_started_ts is None:
                self._idle_started_ts = now
        else:
            self._idle_started_ts = None

    def _classify(
        self,
        *,
        valence: float,
        arousal: float,
        is_emergency: bool,
        now: float,
    ) -> Expression:
        cfg = self._cfg
        if is_emergency:
            return Expression.EMERGENCY
        if (
            self._idle_started_ts is not None
            and (now - self._idle_started_ts) >= cfg.idle_sleepy_after_s
        ):
            return Expression.SLEEPY
        if valence <= cfg.angry_valence_max and arousal >= cfg.angry_arousal_min:
            return Expression.ANGRY
        if arousal >= cfg.arousal_alert_min:
            return Expression.ALERT
        if valence >= cfg.valence_happy_min:
            return Expression.HAPPY
        if valence <= cfg.valence_sad_max:
            return Expression.SAD
        if arousal <= cfg.arousal_sleepy_max:
            return Expression.SLEEPY
        return Expression.NEUTRAL

    def _apply_dwell(self, *, candidate: Expression, now: float) -> Expression:
        if self._last_expr is None or candidate is Expression.EMERGENCY:
            return candidate
        if candidate == self._last_expr:
            return candidate
        elapsed = now - self._last_change_ts
        if elapsed < self._cfg.min_dwell_s:
            _log.debug(
                "face_update_skipped_dwell",
                candidate=candidate.value,
                current=self._last_expr.value,
                remaining_s=self._cfg.min_dwell_s - elapsed,
            )
            return self._last_expr
        return candidate
