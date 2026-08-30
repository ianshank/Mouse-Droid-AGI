"""MouseDroid orchestrator — background cadence loop mixin.

Handles sensor recovery, memory consolidation, on-device learning, and growth distillation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.safety.context import SafetyContext

_log = get_logger(__name__)


class _BackgroundCadenceMixin:
    """Background cadence loops for the orchestrator."""

    async def _try_sensor_recovery(self, safety_ctx: SafetyContext) -> bool:
        """Attempt sensor recovery if the emergency is due to sensor degradation.

        Only runs when valid_sensor_count is below threshold and the
        configured recovery_attempts > 0.

        Args:
            safety_ctx: Current safety context.

        Returns:
            True if a recovery was attempted, False otherwise.
        """
        max_attempts = self._cfg.safety.sensor_recovery_attempts  # type: ignore[attr-defined]
        if max_attempts <= 0:
            return False
        if safety_ctx.valid_sensor_count >= self._cfg.safety.min_valid_sensors:  # type: ignore[attr-defined]
            return False

        _log.warning(
            "sensor_recovery_starting",
            valid_sensors=safety_ctx.valid_sensor_count,
            required=self._cfg.safety.min_valid_sensors,  # type: ignore[attr-defined]
            max_attempts=max_attempts,
        )

        for attempt in range(max_attempts):
            recovered = await self._sensor_manager.recovery_attempt()  # type: ignore[attr-defined]
            if recovered > 0:
                _log.info(
                    "sensor_recovery_success",
                    attempt=attempt + 1,
                    recovered=recovered,
                )
                return True
            if attempt < max_attempts - 1:
                await self._clock.sleep(self._cfg.safety.sensor_recovery_delay_s)  # type: ignore[attr-defined]

        _log.error("sensor_recovery_exhausted", attempts=max_attempts)
        return False

    async def _run_slow_cadence_loop(
        self,
        *,
        interval: float,
        started_event: str,
        failed_event: str,
        cycle_fn: Callable[[], Awaitable[object]],
        should_continue: Callable[[], bool] | None = None,
    ) -> None:
        """Shared body for the off-loop slow-cadence background loops.

        ``_on_device_update_loop``, ``_growth_distill_loop``, and
        ``_consolidation_loop`` were structurally identical modulo names: log a
        start event, then forever sleep-wake-run-a-cycle, logging (not raising)
        on a failed cycle so a transient error never kills the background task.
        Each caller keeps its OWN pre-loop guard clause (coordinator-absent /
        config-absent) so the loop body here is entered only once the caller has
        already decided to run — tests assert those guards return immediately
        without ever reaching a real ``clock.sleep``, which folding the guard in
        here would break.

        Args:
            interval: Seconds between cycles, read from the caller's config.
            started_event: Literal event name logged once at entry. Passed as a
                plain string, never f-string-built here — AGENTS.md invariant 3
                ("no f-string log messages") applies to a shared helper exactly
                as much as to a single call site; each caller supplies its own
                literal (e.g. ``"on_device_update_loop_started"``), so grepping
                the CALL SITE for the exact event name still works.
            failed_event: Literal event name logged on a failed cycle, same
                literal-per-caller discipline as ``started_event``.
            cycle_fn: The awaitable cycle body to run each wake-up.
            should_continue: Optional per-iteration continuation check, evaluated
                after each sleep and before ``cycle_fn``. ``None`` (the default)
                means "run forever" — matches ``_on_device_update_loop`` and
                ``_growth_distill_loop``, which have no in-loop exit condition.
                ``_consolidation_loop`` passes one to preserve its ``break`` on a
                cleared memory tier.
        """
        _log.info(started_event, interval_s=interval)
        while True:
            await self._clock.sleep(interval)  # type: ignore[attr-defined]
            if should_continue is not None and not should_continue():
                break
            try:
                await cycle_fn()
            except Exception:
                _log.warning(failed_event, exc_info=True)

    async def _consolidation_loop(self) -> None:
        """Background loop that consolidates episodic memory into semantic index.

        Runs at the interval specified by ``cfg.memory.consolidation_interval_s``.
        Automatically cancelled by ``stop()``.
        """
        interval = self._cfg.memory.consolidation_interval_s  # type: ignore[attr-defined]

        async def _cycle() -> None:
            memory_tier = self._memory_tier  # type: ignore[attr-defined]
            if memory_tier is None:  # pragma: no cover - should_continue already guards this
                return
            count = await asyncio.to_thread(memory_tier.consolidation.consolidate)
            if count > 0:
                _log.debug(
                    "consolidation_cycle_complete",
                    records_consolidated=count,
                    semantic_size=memory_tier.semantic.size,
                )

        await self._run_slow_cadence_loop(
            interval=interval,
            started_event="consolidation_loop_started",
            failed_event="consolidation_cycle_failed",
            cycle_fn=_cycle,
            should_continue=lambda: self._memory_tier is not None,  # type: ignore[attr-defined]
        )

    def _on_device_learning_enabled(self) -> bool:
        """Return ``True`` only when the WS1 on-device block is enabled.

        Uses an explicit ``None`` / attribute check (no ``assert``) so the gate
        survives ``-O`` (PYTHONOPTIMIZE=1, the Jetson Docker default).
        """
        on_device_cfg = self._cfg.on_device_learning  # type: ignore[attr-defined]
        return on_device_cfg is not None and on_device_cfg.enabled

    async def _on_device_update_loop(self) -> None:
        """Background slow-cadence loop driving the WS3 update coordinator.

        Runs at ``cfg.on_device_learning.check_interval_s`` OUTSIDE the 30 Hz
        hot loop. Each tick the coordinator probes the fresh-record count and,
        when armed, produces + persists a SHA-256-stamped candidate slot — the
        bounded torch update is offloaded via ``asyncio.to_thread`` inside the
        coordinator so the event loop is never blocked. Automatically cancelled
        by ``stop()``. A failed cycle is logged and the loop keeps running so a
        transient replay-store / disk error never kills on-device learning.
        """
        on_device_cfg = self._cfg.on_device_learning  # type: ignore[attr-defined]
        if on_device_cfg is None or self._on_device_coordinator is None:  # type: ignore[attr-defined]
            return
        coordinator = self._on_device_coordinator  # type: ignore[attr-defined]
        await self._run_slow_cadence_loop(
            interval=on_device_cfg.check_interval_s,
            started_event="on_device_update_loop_started",
            failed_event="on_device_update_cycle_failed",
            cycle_fn=coordinator.maybe_update,
        )

    def _growth_enabled(self) -> bool:
        """Return ``True`` only when the growth-distillation block is enabled.

        Explicit ``None`` / attribute check (no ``assert``) so the gate survives
        ``-O`` (PYTHONOPTIMIZE=1, the Jetson Docker default).
        """
        growth_cfg = self._cfg.growth  # type: ignore[attr-defined]
        return growth_cfg is not None and growth_cfg.enabled

    async def _growth_distill_loop(self) -> None:
        """Background slow-cadence loop driving the growth-distillation coordinator.

        Runs at ``cfg.growth.check_interval_s`` OUTSIDE the 30 Hz hot loop. Each
        tick the coordinator probes the fresh-record count and, when armed,
        distils the VLA teacher into the compact student and persists a
        SHA-256-stamped slot — all torch work is offloaded via ``asyncio.to_thread``
        inside the coordinator so the event loop is never blocked. Automatically
        cancelled by ``stop()``. A failed cycle is logged and the loop keeps
        running so a transient error never kills distillation.
        """
        growth_cfg = self._cfg.growth  # type: ignore[attr-defined]
        if growth_cfg is None or self._growth_coordinator is None:  # type: ignore[attr-defined]
            return
        coordinator = self._growth_coordinator  # type: ignore[attr-defined]
        await self._run_slow_cadence_loop(
            interval=growth_cfg.check_interval_s,
            started_event="growth_distill_loop_started",
            failed_event="growth_distill_cycle_failed",
            cycle_fn=coordinator.maybe_distill,
        )
