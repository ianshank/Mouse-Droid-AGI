"""MouseDroid orchestrator — lifecycle management mixin.

Handles startup, shutdown, background task management, and health checks.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from mousedroid.common.async_utils import cancel_and_drain, spawn_tracked
from mousedroid.logging.setup import get_logger
from mousedroid.orchestrator._state import _OrchestratorState

_log = get_logger(__name__)


class _LifecycleMixin(_OrchestratorState):
    """Lifecycle management for the orchestrator."""

    async def start(self) -> None:
        """Start all subsystems."""
        _log.info("orchestrator_starting")
        # F-014 follow-up: surface the RESOLVED mock-hardware boolean in the
        # boot log so an operator reading container logs can tell at a glance
        # whether real or mock drivers were wired. ``health_check`` already
        # exposes this, but that is an on-demand API response, not a boot
        # artefact — this is the one boot-time touch-point.
        _log.info("mock_hardware_resolved", value=self._cfg.mock_hardware)
        if self._hailo_runtime is not None:
            await self._hailo_runtime.start()
        await self._esp32.connect()
        await self._sensor_manager.start()
        if self._cognitive_core is not None:
            await self._cognitive_core.start()
        if self._telemetry_server is not None:
            from mousedroid.telemetry.exceptions import TelemetryUnavailableError

            try:
                await self._telemetry_server.start()
            except TelemetryUnavailableError:
                _log.warning("telemetry_start_degraded", exc_info=True)
                self._telemetry_server = None
        # PR #4: start the mock telemetry source after the server is
        # up so its synthesised payloads land on a live publisher queue.
        # Wrapped in try/except so a buggy mock source never blocks
        # production startup.
        if self._mock_telemetry_source is not None:
            try:
                await self._mock_telemetry_source.start()
                _log.info("mock_telemetry_source_running")
            except Exception:
                _log.warning("mock_telemetry_source_start_failed", exc_info=True)
        if self._mcp_server is not None:
            await self._mcp_server.start()
        if self._llm_gateway is not None:
            try:
                await self._llm_gateway.start()
            except RuntimeError:
                _log.warning("llm_gateway_start_failed", exc_info=True)
        if self._voice_engine is not None:
            await self._voice_engine.start()
            await self._voice_lifecycle("startup")
        if self._face_controller is not None:
            await self._face_controller.start()
        if self._experience_logger is not None:
            self._experience_logger.open()
        await self._start_cloud_subsystems()
        self._spawn_slow_background_tasks()
        # Harness journal (background writer task). NullJournal is a no-op.
        await self._journal.start()
        # Issue #109 — one-shot MSE-6 greeting, fired here (AFTER the voice
        # engine is started above, BEFORE the 30 Hz loop begins). This is
        # the only greeting touch-point; the hot loop never sees it.
        await self._maybe_fire_startup_greeting()
        self._running = True
        _log.info("orchestrator_started")

    async def _maybe_fire_startup_greeting(self) -> None:
        """Fire the startup greeting once iff configured + wired.

        Gated on ``cfg.greeting`` being a non-None enabled config with
        ``fire_on_startup=True`` AND a greeter present. Wrapped in
        try/except so a greeting failure (TTS / speaker hiccup) is logged
        and swallowed — it must NEVER block orchestrator startup. Runs
        OUTSIDE the 30 Hz control loop (one-shot at ``start()``), so the
        hot loop stays byte-identical when the flag is off.
        """
        greeting_cfg = self._cfg.greeting
        if (
            greeting_cfg is None
            or not greeting_cfg.enabled
            or not greeting_cfg.fire_on_startup
            or self._greeter is None
        ):
            return
        try:
            # Bound the greeting with a config-driven timeout: a hung TTS engine
            # or blocked ALSA device must never wedge bring-up, since this runs
            # before the 30 Hz loop starts. On timeout the greeting is abandoned
            # and startup proceeds.
            await asyncio.wait_for(
                self._greeter.greet(),
                timeout=greeting_cfg.startup_timeout_s,
            )
        except (TimeoutError, asyncio.TimeoutError):
            # asyncio.TimeoutError is an alias for the builtin TimeoutError on
            # Python 3.11+, but a DISTINCT exception on 3.10 (a supported CI
            # leg). Catching both keeps the precise greeting_startup_timeout
            # event firing on every supported interpreter — matching the
            # dual-catch pattern in common/tools/registry.py + voice/rocky.py.
            _log.warning(
                "greeting_startup_timeout",
                timeout_s=greeting_cfg.startup_timeout_s,
            )
            return
        except asyncio.CancelledError:
            # Cooperative cancellation MUST propagate — never swallow it.
            # ``CancelledError`` subclasses ``BaseException`` (not
            # ``Exception``) on every supported interpreter, so the broad
            # ``except Exception`` below would not eat it — this explicit
            # re-raise is defensive documentation of the contract. Matches
            # the LLM gateway/composite cancellation contract.
            raise
        except Exception:
            # A flaky speaker / TTS must never crash bring-up.
            _log.warning("greeting_startup_failed", exc_info=True)
            return
        _log.info("greeting_startup_complete", name_count=len(greeting_cfg.names))

    async def _start_cloud_subsystems(self) -> None:
        """Start the cloud sink, exporters, Firestore sync, and OTA weight pollers.

        Each guard is a no-op when the subsystem is unwired. Every cloud
        collaborator's ``start()`` — like every poller's ``start()`` — is
        wrapped so a boot-time failure (an unreachable GCP backend, a missing
        ``google-cloud-*`` SDK the collaborator's own deferred import only
        discovers here, HF Hub unreachable, etc.) can't block the orchestrator
        from coming up; an empty poller mapping skips that loop entirely, so
        default deployments pay zero cost.
        """
        if self._cloud_sink is not None:
            try:
                await self._cloud_sink.start()
            except Exception:
                _log.warning("cloud_sink_start_failed", exc_info=True)
        if self._cloud_experience_exporter is not None:
            try:
                await self._cloud_experience_exporter.start()
            except Exception:
                _log.warning("cloud_experience_exporter_start_failed", exc_info=True)
        if self._cloud_metrics_exporter is not None:
            try:
                await self._cloud_metrics_exporter.start()
            except Exception:
                _log.warning("cloud_metrics_exporter_start_failed", exc_info=True)
        if self._cloud_firestore_sync is not None:
            try:
                await self._cloud_firestore_sync.start()
            except Exception:
                _log.warning("cloud_firestore_sync_start_failed", exc_info=True)
        for poller in self._weight_update_pollers.values():
            try:
                await poller.start()
            except Exception:
                _log.warning("cloud_weight_update_poller_start_failed", exc_info=True)

    def _spawn_slow_background_tasks(self) -> None:
        """Spawn the memory-consolidation and on-device-learning slow tasks.

        Both spawns are gated: consolidation only when a memory tier is wired,
        the on-device replay-trigger task only when the coordinator is wired AND
        on-device learning is enabled. Both gates absent keeps the lifecycle
        byte-identical to pre-WS3. Runs OUTSIDE the 30 Hz loop.
        """
        if self._memory_tier is not None:
            self._consolidation_task = spawn_tracked(
                self._consolidation_tasks,
                self._consolidation_loop(),
                name=self._consolidation_loop.__name__,
            )
        if self._on_device_coordinator is not None and self._on_device_learning_enabled():
            self._on_device_task = spawn_tracked(
                self._on_device_tasks,
                self._on_device_update_loop(),
                name=self._on_device_update_loop.__name__,
            )
        if self._growth_coordinator is not None and self._growth_enabled():
            self._growth_task = spawn_tracked(
                self._growth_tasks,
                self._growth_distill_loop(),
                name=self._growth_distill_loop.__name__,
            )

    async def stop(self) -> None:
        """Stop all subsystems gracefully.

        Actuator teardown (:meth:`_halt_actuators`) runs in a ``finally`` so a
        failure anywhere in software shutdown cannot leave the motors running.
        Before this structure the emergency stop sat ~20 statements downstream
        of ``_drain_background_tasks``, so any raise in between — a drain
        timeout, a wedged voice engine, a telemetry server that would not
        close — skipped it and left the last commanded velocity latched.
        """
        _log.info("orchestrator_stopping")
        self._running = False
        try:
            await self._drain_background_tasks()
            await self._stop_cloud_subsystems()
            await self._stop_software_subsystems()
        finally:
            await self._halt_actuators()
            await self._stop_residual_subsystems()
        _log.info("orchestrator_stopped")

    async def _stop_software_subsystems(self) -> None:
        """Stop the non-actuator subsystems in reverse-start (LIFO) order.

        Every guard is a no-op when the subsystem was never wired. Raising here
        is acceptable — :meth:`stop`'s ``finally`` still halts the actuators.
        """
        if self._experience_logger is not None:
            self._experience_logger.close()
        if self._face_controller is not None:
            await self._face_controller.stop()
        if self._voice_engine is not None:
            await self._voice_lifecycle("shutdown")
            await self._voice_engine.stop()
        if self._mcp_server is not None:
            await self._mcp_server.stop()
        # PR #4: stop the mock telemetry source BEFORE the server so
        # synthetic payloads drain cleanly into the broadcast loop.
        if self._mock_telemetry_source is not None:
            with contextlib.suppress(Exception):
                await self._mock_telemetry_source.stop()
            _log.info("mock_telemetry_source_stopped_via_orchestrator")
        if self._telemetry_server is not None:
            await self._telemetry_server.stop()
        if self._cognitive_core is not None:
            await self._cognitive_core.stop()
        if self._llm_gateway is not None:
            await self._llm_gateway.stop()

    async def _halt_actuators(self) -> None:
        """Halt the motors and release the sensor/actuator transports.

        Safety-critical and best-effort: each step is attempted even if an
        earlier one raised, because a failure to stop the sensor manager must
        not prevent the serial port from being closed (and vice versa). The
        emergency stop is issued first and its failure is logged at ``error``.
        """
        try:
            await self._esp32.emergency_stop()
        except Exception:
            _log.error("shutdown_emergency_stop_failed", exc_info=True)
        try:
            await self._sensor_manager.stop()
        except Exception:
            _log.warning("shutdown_sensor_manager_stop_failed", exc_info=True)
        try:
            await self._esp32.disconnect()
        except Exception:
            _log.warning("shutdown_esp32_disconnect_failed", exc_info=True)

    async def _stop_residual_subsystems(self) -> None:
        """Stop the accelerator runtime and drain the harness journal.

        Runs after :meth:`_halt_actuators` so terminal journal events persist
        last. Best-effort for the same reason as the actuator teardown.
        """
        if self._hailo_runtime is not None:
            try:
                await self._hailo_runtime.stop()
            except Exception:
                _log.warning("shutdown_hailo_runtime_stop_failed", exc_info=True)
        try:
            await self._journal.stop()
        except Exception:
            _log.warning("shutdown_journal_stop_failed", exc_info=True)

    async def _drain_background_tasks(self) -> None:
        """Cancel + drain the consolidation, on-device, and cloud-publish tasks.

        Each guard is a no-op when the task was never spawned. The consolidation
        task is drained via its tracking set when present, else cancelled
        directly; the on-device and cloud-publish task sets drain unconditionally
        (empty sets are no-ops).
        """
        if self._consolidation_task is not None:
            if self._consolidation_task in self._consolidation_tasks:
                await cancel_and_drain(self._consolidation_tasks)
            elif not self._consolidation_task.done():
                self._consolidation_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._consolidation_task
            self._consolidation_tasks.discard(self._consolidation_task)
            self._consolidation_task = None
        # Phase 6 WS3 — drain the on-device slow task (no-op when never spawned).
        if self._on_device_task is not None:
            await cancel_and_drain(self._on_device_tasks)
            self._on_device_task = None
        # Growth pillar — drain the distillation slow task (no-op when never spawned).
        if self._growth_task is not None:
            await cancel_and_drain(self._growth_tasks)
            self._growth_task = None
        await cancel_and_drain(self._cloud_publish_tasks)

    async def _stop_cloud_subsystems(self) -> None:
        """Stop the OTA weight pollers, Firestore sync, exporters, and cloud sink.

        Each poller's ``stop`` is wrapped so a stuck in-flight download can't
        block shutdown of the others or of the orchestrator; an empty mapping
        skips the loop. Guards are no-ops when the subsystem is unwired. Order
        is the reverse of ``_start_cloud_subsystems`` (LIFO teardown). Note
        ``cloud_metrics_exporter`` exposes ``stop()``, not ``close()`` —
        matching :class:`CloudMetricsExporterProtocol`'s teardown method name.
        """
        for poller in self._weight_update_pollers.values():
            try:
                await poller.stop()
            except Exception:
                _log.warning("cloud_weight_update_poller_stop_failed", exc_info=True)
        if self._cloud_firestore_sync is not None:
            await self._cloud_firestore_sync.close()
        if self._cloud_metrics_exporter is not None:
            await self._cloud_metrics_exporter.stop()
        if self._cloud_experience_exporter is not None:
            await self._cloud_experience_exporter.close()
        if self._cloud_sink is not None:
            await self._cloud_sink.flush()
            await self._cloud_sink.close()

    async def run(self) -> None:
        """Run the main loop at configured control rate.

        Each tick is wrapped in ``asyncio.wait_for`` with
        ``cfg.loop.tick_timeout_s`` as the deadline.  A timeout or
        uncaught exception triggers ``emergency_stop`` on the ESP32 to
        halt the motors immediately.
        """
        control_period = 1.0 / self._cfg.loop.control_hz
        tick_timeout = self._cfg.loop.tick_timeout_s
        _log.info(
            "main_loop_starting",
            control_hz=self._cfg.loop.control_hz,
            tick_timeout_s=tick_timeout,
        )

        while self._running:
            tick_start = self._clock.monotonic()
            try:
                await asyncio.wait_for(self.tick(), timeout=tick_timeout)
            except asyncio.TimeoutError:
                _log.critical(
                    "tick_timeout",
                    timeout_s=tick_timeout,
                    elapsed_s=self._clock.monotonic() - tick_start,
                )
                await self._esp32.emergency_stop()
                await self._voice_lifecycle("error")
            except Exception:
                _log.exception("tick_error")
                await self._esp32.emergency_stop()
                await self._voice_lifecycle("error")
            else:
                # Successful tick — notify watchdog
                if self._watchdog is not None:
                    self._watchdog.notify()

            elapsed = self._clock.monotonic() - tick_start
            sleep_time = max(0.0, control_period - elapsed)
            if sleep_time > 0:
                await self._clock.sleep(sleep_time)

    async def health_check(self) -> dict[str, object]:
        """Run a quick health check of all subsystems.

        Returns:
            Health status dict.
        """
        return {
            "status": "ok",
            "platform": str(self._cfg.platform),
            "mock_hardware": self._cfg.mock_hardware,
            "agents": [a.name for a in self._agents],
        }

    async def dispatch_tool(self, name: str, **kwargs: Any) -> Any:
        """Dispatch a named tool via the tool registry.

        Args:
            name: Tool name to dispatch.
            **kwargs: Keyword arguments forwarded to the tool handler.

        Returns:
            Tool result.

        Raises:
            KeyError: If no tool registry is configured.
        """
        if self._tool_registry is None:
            raise KeyError("Tool registry not configured")
        return await self._tool_registry.dispatch(name, **kwargs)
