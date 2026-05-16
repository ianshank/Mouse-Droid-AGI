"""MouseDroid orchestrator — main sense-plan-act loop.

Platform-agnostic via dependency injection. All components injected
through constructor, wired by factory functions.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from mousedroid.common.actions import normalize_action_numpy
from mousedroid.common.async_utils import cancel_and_drain, spawn_tracked
from mousedroid.common.time.protocol import ClockProtocol, RealClock
from mousedroid.constants import (
    DEFAULT_BATTERY_VOLTAGE,
    MILLISECONDS_PER_SECOND,
    MOTOR_STATE_BATTERY_INDEX,
)
from mousedroid.harness.protocol import HookPhase, TickContext
from mousedroid.logging.setup import get_logger
from mousedroid.telemetry.frame_builder import build_telemetry_frame

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mousedroid.agents.base import AgentProtocol
    from mousedroid.cloud.protocol import (
        CloudExperienceExporterProtocol,
        CloudTelemetrySinkProtocol,
        PendingWeightUpdate,
        WeightUpdatePollerProtocol,
    )
    from mousedroid.cognitive.cognitive_core import CognitiveCore
    from mousedroid.comms.protocol import ESP32CommProtocol
    from mousedroid.config.schema import Settings
    from mousedroid.curiosity.protocol import CuriosityProtocol
    from mousedroid.experience.logger import ExperienceLogger
    from mousedroid.hardware.accelerator.hailo_runtime import HailoRuntimeProtocol
    from mousedroid.harness.journal.protocol import JournalProtocol
    from mousedroid.harness.protocol import (
        HookRegistryProtocol,
        TaskTrackerProtocol,
    )
    from mousedroid.health.watchdog import WatchdogProtocol
    from mousedroid.llm_gateway.mission_parser import MissionParserProtocol
    from mousedroid.llm_gateway.protocol import GoalVector, LLMGatewayProtocol
    from mousedroid.mcp.protocol import MCPServerProtocol
    from mousedroid.memory.exporter import MemoryExporterProtocol
    from mousedroid.memory.tier import MemoryTier
    from mousedroid.orchestrator.face_controller import FaceController
    from mousedroid.orchestrator.mission_dispatcher import MissionDispatcherProtocol
    from mousedroid.safety.context import SafetyContext
    from mousedroid.safety.projector_protocol import SafetyActionProjectorProtocol
    from mousedroid.safety.protocol import SafetyMonitorProtocol
    from mousedroid.sensing.manager import SensorManager
    from mousedroid.sensing.protocol import ObservationProtocol
    from mousedroid.skills.delegator import SkillDelegator
    from mousedroid.telemetry.failure_recorder import FailureRecorder
    from mousedroid.telemetry.metrics import MetricsRegistry
    from mousedroid.telemetry.protocol import TelemetryPublisherProtocol, TelemetryServerProtocol
    from mousedroid.vla.policy import VLAPolicyProtocol
    from mousedroid.voice.protocol import VoiceEngineProtocol
    from mousedroid.world_model.protocol import WorldModelProtocol

_log = get_logger(__name__)


class MouseDroidOrchestrator:
    """Main orchestrator — runs the sense-plan-act loop.

    All components are injected via constructor. No hardcoded types.
    """

    def __init__(
        self,
        world_model: WorldModelProtocol,
        agents: list[AgentProtocol],
        safety_monitor: SafetyMonitorProtocol,
        esp32: ESP32CommProtocol,
        sensor_manager: SensorManager,
        cfg: Settings,
        cognitive_core: CognitiveCore | None = None,
        telemetry_publisher: TelemetryPublisherProtocol | None = None,
        telemetry_server: TelemetryServerProtocol | None = None,
        voice_engine: VoiceEngineProtocol | None = None,
        hailo_runtime: HailoRuntimeProtocol | None = None,
        memory_tier: MemoryTier | None = None,
        experience_logger: ExperienceLogger | None = None,
        curiosity_module: CuriosityProtocol | None = None,
        llm_gateway: LLMGatewayProtocol | None = None,
        mission_parser: MissionParserProtocol | None = None,
        watchdog: WatchdogProtocol | None = None,
        cloud_sink: CloudTelemetrySinkProtocol | None = None,
        cloud_experience_exporter: CloudExperienceExporterProtocol | None = None,
        tool_registry: Any | None = None,
        face_controller: FaceController | None = None,
        mcp_server: MCPServerProtocol | None = None,
        vla_policy: VLAPolicyProtocol | None = None,
        hook_registry: HookRegistryProtocol | None = None,
        task_tracker: TaskTrackerProtocol | None = None,
        journal: JournalProtocol | None = None,
        skill_delegator: SkillDelegator | None = None,
        memory_exporter: MemoryExporterProtocol | None = None,
        mission_dispatcher: MissionDispatcherProtocol | None = None,
        clock: ClockProtocol | None = None,
        failure_recorder: FailureRecorder | None = None,
        liveness_tracker: Any | None = None,
        mock_telemetry_source: Any | None = None,
        metrics: MetricsRegistry | None = None,
        weight_update_poller: WeightUpdatePollerProtocol | None = None,
        weight_update_pollers: Mapping[str, WeightUpdatePollerProtocol] | None = None,
        weight_update_loader: Callable[[PendingWeightUpdate], object] | None = None,
        safety_projector: SafetyActionProjectorProtocol | None = None,
    ) -> None:
        """Initialise orchestrator with all components.

        Args:
            world_model: World model for latent dynamics.
            agents: List of navigation agents.
            safety_monitor: Safety monitor.
            esp32: ESP32 communication driver.
            sensor_manager: Sensor manager for concurrent sensor reads.
            cfg: Root settings.
            cognitive_core: Optional CognitiveCore for BDI/metacognitive loop.
            telemetry_publisher: Optional telemetry publisher for remote monitoring.
            telemetry_server: Optional telemetry server for remote connections.
            voice_engine: Optional Rocky voice engine for audio output.
            hailo_runtime: Optional Hailo-8 accelerator runtime for lifecycle management.
            memory_tier: Optional layered memory tier for episodic/semantic/working memory.
            experience_logger: Optional LMDB-backed experience logger.
            curiosity_module: Optional intrinsic curiosity module (ICM).
            llm_gateway: Optional LLM gateway for NL command translation.
            mission_parser: Optional rule-based NL mission parser.
            watchdog: Optional watchdog notifier for liveness signalling.
            cloud_sink: Optional GCP Pub/Sub telemetry sink for cloud streaming.
            cloud_experience_exporter: Optional GCS experience batch exporter.
            tool_registry: Optional tool registry for runtime tool dispatch.
            face_controller: Optional MSE-6 face-display controller. When
                supplied, the orchestrator drives it from the BDI affect
                signal and ``SafetyContext.is_emergency`` once per tick.
            mcp_server: Optional MCP server. When supplied, started after
                the telemetry server during ``start()`` and stopped just
                before it during ``stop()``. Runs as its own supervised
                background tasks; never blocks the control loop.
            vla_policy: Optional VLA policy (Phase 3a). When supplied and
                ``cfg.loop.policy_selector`` selects it, the orchestrator
                routes action selection through the policy with a per-tick
                inference timeout (``cfg.loop.inference_timeout_s``,
                defaulting to ``1.0 / control_hz``).
            hook_registry: Optional :class:`HookRegistryProtocol`. When
                ``None``, a :class:`NullHookRegistry` is used so the
                30 Hz hot loop is byte-identical to the legacy path.
            task_tracker: Optional :class:`TaskTrackerProtocol`. When
                ``None``, no task tracking is performed.
            journal: Optional :class:`JournalProtocol`. When ``None``,
                a :class:`NullJournal` is used (no-op start/stop/append).
            skill_delegator: Optional :class:`SkillDelegator` used by the
                MCP bridge / skills to dispatch tasks to sub-agents.
            memory_exporter: Optional :class:`MemoryExporterProtocol`. When
                supplied together with a non-None ``memory_tier`` and a
                non-None ``mission_dispatcher``, the orchestrator runs a
                snapshot export from the POST_TICK path on the cadence
                set by :attr:`OpenClawConfig.export_every_n_ticks`.
                ``None`` keeps existing deployments byte-identical.
            mission_dispatcher: Optional :class:`MissionDispatcherProtocol`.
                When supplied, its ``mission_just_completed`` flag gates
                the export hook so MEMORY.md is only refreshed after a
                channel-driven mission completes.
            clock: Optional :class:`ClockProtocol` for time and sleep
                primitives. Defaults to :class:`RealClock` (production).
                Pass a :class:`MockClock` in tests to control simulated time
                without wall-clock delays.
            failure_recorder: Optional :class:`FailureRecorder` for emitting
                structured failure events and Prometheus counters. Defaults
                to a :class:`NullFailureRecorder` (no-op) when ``None``.
            liveness_tracker: Optional :class:`SensorLivenessTracker`.
                When supplied, the orchestrator threads it through
                :func:`build_telemetry_frame` so every published frame
                carries a per-sensor liveness map
                (``disabled`` / ``awaiting`` / ``live`` / ``stale``).
                ``None`` (default) preserves byte-identical behaviour
                with an empty ``sensor_liveness`` field.
            mock_telemetry_source: Optional ``MockTelemetrySource`` (or
                any object exposing ``async start()`` / ``async stop()``).
                When supplied, started after telemetry server during
                ``start()`` and stopped just before it during ``stop()``
                so the dashboard renders synthetic motion in mock mode.
            metrics: Optional :class:`MetricsRegistry`. When supplied,
                ``_try_vla_action()`` calls ``inc_vla_timeout(mode=cfg.vla.backend)``
                on the timeout branch so operators see VLA fallback events
                in Prometheus. ``None`` (default) preserves byte-identical
                pre-PR-A2.1 behavior.
            weight_update_poller: Optional Tier C1
                :class:`WeightUpdatePollerProtocol`. When supplied, the
                orchestrator polls ``pending_update`` once per tick AFTER
                ``_select_action`` and atomically swaps the live world-model
                / policy with the newly downloaded artifact. ``None`` (default)
                preserves byte-identical pre-Tier-C1 behavior. Legacy single-
                poller kwarg retained for one minor-version window for
                backwards compatibility — folded into
                ``self._weight_update_pollers`` under the poller's
                ``_engine_type`` (or ``"policy"`` if absent) so internal
                handling is uniform between the two shapes. Prefer
                ``weight_update_pollers`` for new call sites.
            weight_update_pollers: Optional Tier C1.2 mapping
                ``{engine_type: WeightUpdatePollerProtocol}``. When supplied,
                each tick the orchestrator iterates all pollers (in insertion
                order — ``policy`` before ``world_model`` is the documented
                contract) and applies any pending update. Supersedes
                ``weight_update_poller`` for multi-engine OTA deployments.
                Empty mapping (``{}``) and ``None`` both preserve byte-
                identical pre-Tier-C1 behaviour (swap helper short-circuits).
            weight_update_loader: Optional callable
                ``(PendingWeightUpdate) -> engine``. Invoked inside
                ``_apply_pending_weight_update`` to materialise the new
                engine BEFORE the reference swap (so a load failure does
                NOT corrupt the live model). Tests inject a stub; production
                wires a loader that lazy-imports ``onnxruntime`` and builds
                a fresh engine of the same type as the live one. Must be
                supplied alongside ``weight_update_poller`` — the poller
                downloads, the loader materialises.
            safety_projector: Optional :class:`SafetyActionProjectorProtocol`
                (Tier C2 / C2.1). When supplied, every action returned by
                :meth:`_select_action` is run through the projector at the
                tick-level seam, so all four ``_select_action`` return
                branches (cognitive / VLA / VLA-strict-timeout / nav_agent)
                are clamped uniformly. ``None`` (default, gated by
                ``cfg.safety.projector.enabled=false``) makes the seam a
                pure pass-through so existing deployments are byte-identical.
        """
        if not agents:
            msg = "At least one agent is required"
            raise ValueError(msg)

        self._world_model = world_model
        self._agents = agents
        self._safety_monitor = safety_monitor
        self._esp32 = esp32
        self._sensor_manager = sensor_manager
        self._cognitive_core = cognitive_core
        self._telemetry_publisher = telemetry_publisher
        self._telemetry_server = telemetry_server
        self._voice_engine = voice_engine
        self._hailo_runtime = hailo_runtime
        self._memory_tier = memory_tier
        self._experience_logger = experience_logger
        self._curiosity_module = curiosity_module
        self._llm_gateway = llm_gateway
        self._mission_parser = mission_parser
        self._watchdog = watchdog
        self._cloud_sink = cloud_sink
        self._cloud_experience_exporter = cloud_experience_exporter
        self._tool_registry = tool_registry
        self._face_controller = face_controller
        self._mcp_server = mcp_server
        self._vla_policy = vla_policy
        self._cfg = cfg
        # ---- Agent harness (all opt-in, defaults preserve legacy behaviour) --
        # When ``cfg.harness is None`` the orchestrator runs against no-op
        # registries / journal so the 30 Hz hot loop is byte-identical.
        from mousedroid.harness.hooks import NullHookRegistry  # local: avoid cycle
        from mousedroid.harness.journal.null_journal import NullJournal

        self._hook_registry: HookRegistryProtocol = (
            hook_registry if hook_registry is not None else NullHookRegistry()
        )
        self._task_tracker: TaskTrackerProtocol | None = task_tracker
        self._journal: JournalProtocol = journal if journal is not None else NullJournal()
        self._skill_delegator: SkillDelegator | None = skill_delegator
        self._memory_exporter: MemoryExporterProtocol | None = memory_exporter
        self._mission_dispatcher: MissionDispatcherProtocol | None = mission_dispatcher
        self._memory_export_every_n: int = (
            cfg.openclaw.export_every_n_ticks if cfg.openclaw is not None else 0
        )
        self._clock: ClockProtocol = clock if clock is not None else RealClock()
        from mousedroid.telemetry.failure_recorder import NullFailureRecorder

        self._failure_recorder: FailureRecorder = (
            failure_recorder if failure_recorder is not None else NullFailureRecorder()
        )
        self._liveness_tracker: Any | None = liveness_tracker
        self._mock_telemetry_source: Any | None = mock_telemetry_source
        self._metrics = metrics
        # Tier C1 / C1.2 — OTA weight-update wiring. An empty mapping (or
        # both kwargs left at ``None``) keeps the pre-C1 tick path byte-
        # identical (swap helper short-circuits). The legacy single
        # ``weight_update_poller=`` kwarg stays on the constructor for one
        # minor-version window for backwards compatibility; it is folded
        # into ``self._weight_update_pollers`` under the poller's
        # ``_engine_type`` attribute (defaulting to ``"policy"`` when the
        # attribute is missing) so the rest of the orchestrator only ever
        # sees the mapping shape.
        self._weight_update_pollers: dict[str, WeightUpdatePollerProtocol] = dict(
            weight_update_pollers or {}
        )
        if weight_update_poller is not None and not self._weight_update_pollers:
            engine_type = getattr(weight_update_poller, "_engine_type", "policy")
            self._weight_update_pollers[engine_type] = weight_update_poller
        elif weight_update_poller is not None and self._weight_update_pollers:
            _log.warning(
                "weight_update_poller_kwarg_ignored",
                reason="weight_update_pollers mapping takes precedence",
            )
        self._weight_update_loader: Callable[[PendingWeightUpdate], object] | None = (
            weight_update_loader
        )
        # Tier C2 / C2.1 — soft-constraint safety projector. ``None`` (the
        # default, gated by ``cfg.safety.projector.enabled=False``) makes
        # ``_maybe_project_action`` a no-op so existing deployments produce
        # byte-identical actions to pre-C2.
        self._safety_projector: SafetyActionProjectorProtocol | None = safety_projector
        self._running = False
        self._tick_count: int = 0
        self._consolidation_task: asyncio.Task[None] | None = None
        self._consolidation_tasks: set[asyncio.Task[Any]] = set()
        # Strong-reference set for fire-and-forget cloud publishes. Keeping
        # the reference prevents premature GC of the asyncio.Task; entries
        # are evicted by spawn_tracked's done-callback as tasks resolve.
        self._cloud_publish_tasks: set[asyncio.Task[Any]] = set()

        # Latent state (combined_dim = hidden_dim + cfc_hidden_dim for dual-stream)
        _combined_hidden_dim = cfg.model.hidden_dim + cfg.model.cfc_hidden_dim
        self._h = torch.zeros(1, _combined_hidden_dim)
        self._z = torch.zeros(1, cfg.model.latent_dim)
        self._prev_action = torch.zeros(1, cfg.model.action_dim)
        # Rolling buffer of (h, z) tuples for latent NaN recovery.
        self._latent_buffer: deque[tuple[torch.Tensor, torch.Tensor]] = deque(
            maxlen=cfg.model.latent_recovery_buffer_size
        )

    async def start(self) -> None:
        """Start all subsystems."""
        _log.info("orchestrator_starting")
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
        if self._cloud_sink is not None:
            await self._cloud_sink.start()
        if self._cloud_experience_exporter is not None:
            await self._cloud_experience_exporter.start()
        # Tier C1 / C1.2 — start every wired OTA poller as part of the
        # orchestrator lifecycle. Wrapped in try/except so a poller failure
        # (HF Hub unreachable at boot, etc.) can't block the orchestrator
        # from coming up. Each poller's own ``start`` is a no-op when
        # ``poll_interval_s = 0.0``, so default deployments pay zero cost.
        # An empty mapping skips the loop entirely. (Copilot 3253293644 /
        # 3253309972.)
        for poller in self._weight_update_pollers.values():
            try:
                await poller.start()
            except Exception:  # pylint: disable=broad-except
                _log.warning("cloud_weight_update_poller_start_failed", exc_info=True)
        if self._memory_tier is not None:
            self._consolidation_task = spawn_tracked(
                self._consolidation_tasks,
                self._consolidation_loop(),
                name=self._consolidation_loop.__name__,
            )
        # Harness journal (background writer task). NullJournal is a no-op.
        await self._journal.start()
        self._running = True
        _log.info("orchestrator_started")

    async def stop(self) -> None:
        """Stop all subsystems gracefully."""
        _log.info("orchestrator_stopping")
        self._running = False
        if self._consolidation_task is not None:
            if self._consolidation_task in self._consolidation_tasks:
                await cancel_and_drain(self._consolidation_tasks)
            elif not self._consolidation_task.done():
                self._consolidation_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._consolidation_task
            self._consolidation_tasks.discard(self._consolidation_task)
            self._consolidation_task = None
        await cancel_and_drain(self._cloud_publish_tasks)
        # Tier C1 / C1.2 — stop every wired OTA poller. Wrapped in
        # try/except so a stuck in-flight download on one poller can't
        # block shutdown of the others or of the orchestrator itself.
        # An empty mapping skips the loop entirely.
        for poller in self._weight_update_pollers.values():
            try:
                await poller.stop()
            except Exception:  # pylint: disable=broad-except
                _log.warning("cloud_weight_update_poller_stop_failed", exc_info=True)
        if self._cloud_experience_exporter is not None:
            await self._cloud_experience_exporter.close()
        if self._cloud_sink is not None:
            await self._cloud_sink.flush()
            await self._cloud_sink.close()
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
        await self._esp32.emergency_stop()
        await self._sensor_manager.stop()
        await self._esp32.disconnect()
        if self._hailo_runtime is not None:
            await self._hailo_runtime.stop()
        # Drain and stop the harness journal last so terminal events persist.
        await self._journal.stop()
        _log.info("orchestrator_stopped")

    async def tick(self) -> None:
        """Execute one sense-plan-act cycle.

        When the agent harness is configured, hooks fire at five phases —
        ``pre_tick``, ``pre_action``, ``post_action``, ``post_tick``, and
        ``on_error`` — and active tasks are evaluated once per tick. With
        ``Settings.harness=None`` every harness call is a constant-time
        no-op, so the legacy behaviour is bit-identical.
        """
        loop_start = self._clock.monotonic()
        ctx = TickContext(
            tick_index=self._tick_count,
            timestamp_s=loop_start,
            prev_action=self._prev_action,
        )
        try:
            observation = await self._sensor_manager.read_all()
            loop_time_ms = (self._clock.monotonic() - loop_start) * MILLISECONDS_PER_SECOND

            safety_ctx = self._safety_monitor.evaluate(observation, loop_time_ms)

            self._update_world_model(observation)

            ctx.observation = observation
            ctx.safety_ctx = safety_ctx
            ctx.loop_time_ms = loop_time_ms
            if self._task_tracker is not None:
                ctx.active_tasks = tuple(s.id for s in self._task_tracker.active())
            await self._hook_registry.run_phase(HookPhase.PRE_TICK, ctx)

            if safety_ctx.is_emergency:
                # Attempt sensor recovery before emergency stop if sensors degraded
                if await self._try_sensor_recovery(safety_ctx):
                    # Re-read after recovery — sensors may have come back
                    observation = await self._sensor_manager.read_all()
                    loop_time_ms = (self._clock.monotonic() - loop_start) * MILLISECONDS_PER_SECOND
                    safety_ctx = self._safety_monitor.evaluate(observation, loop_time_ms)
                    ctx.observation = observation
                    ctx.safety_ctx = safety_ctx
                    ctx.loop_time_ms = loop_time_ms

                if safety_ctx.is_emergency:
                    await self._esp32.emergency_stop()
                    await self._voice_event("emergency_stop", observation)
                    await self._update_face(safety_ctx=safety_ctx, action=None)
                    _log.warning("emergency_stop_triggered")
                    self._tick_count += 1
                    await self._publish_telemetry(observation, safety_ctx, loop_time_ms)
                    await self._hook_registry.run_phase(HookPhase.POST_TICK, ctx)
                    return

            action = self._select_action(safety_ctx, observation, loop_time_ms)
            # Tier C2 / C2.1 — geometric safety projection seam. Wrapping
            # at the ``tick()`` call site (NOT inside ``_select_action``)
            # ensures all four return sites in ``_select_action`` — cognitive,
            # VLA, VLA-strict-timeout, nav_agent — get clamped uniformly.
            # ``_maybe_project_action`` is a no-op when the projector is
            # disabled, preserving byte-identical pre-PR behaviour. Runs
            # BEFORE the Tier C1 OTA swap so the projector clamps the action
            # produced by the (pre-swap) policy weights this tick saw.
            action = self._maybe_project_action(action, safety_ctx)
            # Tier C1 — atomic OTA swap. Runs AFTER ``_select_action`` so the
            # current tick saw one consistent weight set for both
            # ``_update_world_model`` and ``_select_action``. No-op when the
            # poller is not wired or has no pending update. Returns ``True``
            # iff a world-model swap performed a recurrent-state reset; in
            # that case ``_prev_action`` has already been zeroed by the
            # helper (preserving device + dtype) and MUST NOT be overwritten
            # with the pre-swap action — overwriting would void the
            # ``reset_state_on_swap`` invariant documented in ADR-010.
            swap_reset = self._apply_pending_weight_update()
            if not swap_reset:
                self._prev_action = action.unsqueeze(0) if action.dim() == 1 else action

            ctx.proposed_action = action
            await self._hook_registry.run_phase(HookPhase.PRE_ACTION, ctx)

            await self._execute_action(action)
            ctx.executed_action = action
            await self._hook_registry.run_phase(HookPhase.POST_ACTION, ctx)

            self._log_experience(observation, action)
            await self._voice_observe(observation, safety_ctx)
            await self._update_face(safety_ctx=safety_ctx, action=action)

            self._tick_count += 1
            await self._publish_telemetry(observation, safety_ctx, loop_time_ms)

            if self._task_tracker is not None:
                await self._task_tracker.evaluate_active(ctx)

            await self._hook_registry.run_phase(HookPhase.POST_TICK, ctx)
            # Snapshot AND clear the one-shot ``mission_just_completed``
            # flag atomically (between awaits) BEFORE any observer runs.
            # This avoids three bugs:
            #   * Export running first would clear the flag in its
            #     ``finally`` block, so curiosity reset would silently
            #     skip every mission boundary.
            #   * If the export gate short-circuits (e.g. memory_exporter
            #     is None) the flag would never be cleared and curiosity
            #     would reset on every tick after the first completion.
            #   * Clearing AFTER the export ``await`` would race with a
            #     new mission completing during the I/O window — the
            #     post-await clear would wipe the freshly-latched flag.
            #     By clearing before any await, any completion that lands
            #     during export remains latched for the next tick.
            mission_completed = (
                self._mission_dispatcher is not None
                and self._mission_dispatcher.mission_just_completed
            )
            if mission_completed and self._mission_dispatcher is not None:
                self._mission_dispatcher.clear_mission_completed()
            await self._maybe_export_memory(mission_completed=mission_completed)
            self._maybe_reset_curiosity(mission_completed=mission_completed)

            _log.debug(
                "tick_complete",
                loop_time_ms=loop_time_ms,
                emergency=safety_ctx.is_emergency,
            )
        except Exception as exc:
            ctx.error = exc
            # Hooks observe the error before we re-raise. If a hook itself
            # raises, the HookRegistry's error_policy decides whether to
            # propagate (default: warn-and-continue).
            await self._hook_registry.run_phase(HookPhase.ON_ERROR, ctx)
            raise

    async def _maybe_export_memory(self, *, mission_completed: bool) -> None:
        """Run the OpenClaw MEMORY.md exporter if all three gates pass.

        Gates (any failing gate makes this a no-op):

        1. ``memory_exporter`` was injected (OpenClaw enabled with a
           configured ``shared_memory_path``).
        2. ``memory_tier.episodic`` is non-None (replay buffer exists).
        3. ``mission_completed`` (caller-snapshotted) is ``True`` AND the
           tick count is a multiple of
           ``OpenClawConfig.export_every_n_ticks``.

        The caller is responsible for clearing
        ``mission_just_completed`` exactly once after ALL observers
        (memory exporter, curiosity reset, …) have run; this method no
        longer touches the dispatcher's flag.

        Exceptions are swallowed and logged so a transient filesystem
        failure on the shared path never crashes the control loop.

        Args:
            mission_completed: Snapshot of the dispatcher's
                ``mission_just_completed`` latch taken once per tick.
        """
        if not mission_completed:
            return
        if self._memory_exporter is None or self._memory_tier is None:
            return
        if self._memory_export_every_n <= 0:
            return
        if self._tick_count % self._memory_export_every_n != 0:
            return
        episodic = getattr(self._memory_tier, "episodic", None)
        if episodic is None:
            return
        try:
            _log.info("memory_export_started", path_known=True)
            await self._memory_exporter.export(episodic)
        except Exception as exc:  # pylint: disable=broad-except
            _log.warning(
                "memory_export_hook_failed",
                error=f"{type(exc).__name__}:{exc}",
            )

    def _maybe_reset_curiosity(self, *, mission_completed: bool) -> None:
        """Reset curiosity accumulator at episode boundaries.

        Args:
            mission_completed: Snapshot of the dispatcher's
                ``mission_just_completed`` latch from the tick's
                centralised read so reset fires exactly once per
                mission boundary even when the memory exporter is
                disabled.
        """
        if not mission_completed:
            return
        if self._curiosity_module is None:
            return
        self._curiosity_module.reset_episode()
        _log.info("curiosity_episode_reset", tick=self._tick_count)

    async def process_mission(self, nl_command: str) -> GoalVector:
        """Process a natural language mission command.

        Uses a fallback chain: rule-based parser first (< 1ms), then
        LLM gateway for unknown/ambiguous commands when available.

        Args:
            nl_command: Natural language mission command.

        Returns:
            GoalVector with velocity targets in [-1, 1].
        """
        from mousedroid.llm_gateway.mission_parser import IntentType
        from mousedroid.llm_gateway.protocol import GoalVector

        if not nl_command or not nl_command.strip():
            _log.debug("process_mission_empty_command")
            return GoalVector()

        # Stage 1: Rule-based parser (fast path, < 1ms)
        if self._mission_parser is not None:
            intent = self._mission_parser.parse(nl_command)
            threshold = self._cfg.mission_parser.llm_fallback_confidence
            if intent.confidence >= threshold and intent.intent_type != IntentType.UNKNOWN:
                _log.info(
                    "mission_parsed_rule_based",
                    command=nl_command,
                    intent=intent.intent_type.value,
                    confidence=intent.confidence,
                )
                return intent.goal_vector

        # Stage 2: LLM fallback (slow path, ~100-500ms)
        if self._llm_gateway is not None:
            try:
                goal = await self._llm_gateway.translate_mission(nl_command)
                _log.info(
                    "mission_parsed_llm",
                    command=nl_command,
                    vx=goal.vx_target,
                    vy=goal.vy_target,
                    omega=goal.omega_target,
                )
                return goal
            except Exception:
                _log.warning("mission_llm_fallback_failed", exc_info=True)

        # Stage 3: Fallback to zero (safe default)
        _log.warning("mission_unresolved", command=nl_command)
        return GoalVector()

    def _apply_pending_weight_update(self) -> bool:
        """Atomically swap policy / world-model if any poller has a verified update.

        Runs ONCE per tick, AFTER ``_select_action`` returns. Guarantees the
        current tick saw one consistent weight set for both
        ``_update_world_model`` and ``_select_action``. Reference assignment
        is atomic at the Python interpreter level; we hold no locks because
        the orchestrator's ``tick()`` is single-coroutine on the event loop.

        With Tier C1.2 the orchestrator holds a ``Mapping[str, poller]``
        keyed by ``engine_type``. Each tick this method iterates the mapping
        in the caller-provided insertion order of
        ``self._weight_update_pollers`` and delegates per-poller swap work
        to :meth:`_apply_one_pending_update`. The
        ``build_weight_update_pollers`` factory guarantees ``policy`` before
        ``world_model``; callers constructing ``MouseDroidOrchestrator``
        directly are responsible for the ordering they want. Iteration
        order matters: a world-model swap may zero the recurrent state on
        the same tick, so applying ``policy`` first prevents a stale-policy
        artefact from leaking into a freshly reset world model.

        Method is INTENTIONALLY synchronous: ``tick()`` is the only caller,
        the swap runs entirely in process memory (no I/O after the poller
        downloaded), and keeping it sync avoids scheduling churn between
        select_action and execute_action.

        Returns:
            ``True`` iff at least one world-model swap performed a recurrent-
            state reset (caller MUST skip its own ``_prev_action = action``
            assignment so the zero-state survives into the next tick).
            ``False`` for any other code path (empty mapping, no pendings,
            policy-only swap, loader failure, dead-letter, etc.).
        """
        if not self._weight_update_pollers:
            return False
        any_reset = False
        for poller in self._weight_update_pollers.values():
            update = poller.pending_update
            if update is None:
                continue
            if self._apply_one_pending_update(poller, update):
                any_reset = True
        return any_reset

    def _apply_one_pending_update(
        self,
        poller: WeightUpdatePollerProtocol,
        update: PendingWeightUpdate,
    ) -> bool:
        """Apply one pending update from one poller.

        Extracted from :meth:`_apply_pending_weight_update` so the multi-
        poller loop can delegate per-poller swap work uniformly. The body
        is the unchanged single-poller swap path from Tier C1 — it owns
        the loader invocation, atomic reference swap, engine-type dispatch,
        recurrent-state reset, metric increment, structured-log emission,
        and the final ``acknowledge_swap`` call.

        The new engine is fully materialised via ``self._weight_update_loader``
        BEFORE the reference swap, so a loader failure does NOT corrupt the
        live model — the helper logs the error, leaves the live model
        untouched, and clears the pending slot only on success.

        When ``cfg.cloud.weight_update.reset_state_on_swap`` is ``True`` (the
        default) the latent recurrent state ``(h, z)`` is reset to zeros
        after a world-model swap to avoid one-tick cross-model contamination
        (see ADR-010). The previous-action tensor and latent recovery buffer
        are also cleared in the same pass — they were produced by the OLD
        weights and would seed the new engine with stale context. Device +
        dtype are preserved via ``torch.zeros_like`` so a CUDA-resident
        world-model state survives the swap on its original device.

        Args:
            poller: The poller that surfaced ``update``. Used to invoke
                ``acknowledge_swap`` once the swap (or failure path) lands.
            update: The pending update to apply.

        Returns:
            ``True`` iff this swap zeroed the recurrent state (world-model
            engine only, gated by
            ``cfg.cloud.weight_update.reset_state_on_swap``). ``False`` for
            the no-loader branch, the loader-exception branch, a policy swap,
            and the unknown-engine-type dead-letter branch.
        """
        if self._weight_update_loader is None:
            # Acknowledge-and-warn-once: without ack the same pending update
            # would re-fire ``cloud_weight_update_swap_skipped_no_loader``
            # at 30 Hz forever (one log line per tick). Ack clears the slot
            # so the poller's next download cycle can surface a fresh update,
            # at which point the operator-visible warning fires again — once
            # per revision, not once per tick. (Copilot 3253293630.)
            _log.warning(
                "cloud_weight_update_swap_skipped_no_loader",
                repo_id=update.repo_id,
                revision=update.revision,
                engine_type=update.engine_type,
            )
            poller.acknowledge_swap(update)
            return False

        try:
            new_engine = self._weight_update_loader(update)
        except Exception:  # pylint: disable=broad-except
            _log.error(
                "cloud_weight_update_swap_failed",
                repo_id=update.repo_id,
                revision=update.revision,
                engine_type=update.engine_type,
                exc_info=True,
            )
            # Ack the bad revision so we don't log-spam at 30 Hz. The
            # poller will surface a new PendingWeightUpdate on the next
            # cycle if the upstream artifact changes.
            poller.acknowledge_swap(update)
            return False

        # Atomic reference swap. Single-coroutine guarantee on tick() means
        # no concurrent reader observes a half-swapped state.
        reset_recurrent_state = False
        if update.engine_type == "world_model":
            self._world_model = cast("WorldModelProtocol", new_engine)
            reset_recurrent_state = self._cfg.cloud.weight_update.reset_state_on_swap
        elif update.engine_type == "policy":
            self._vla_policy = cast("VLAPolicyProtocol", new_engine)
        else:
            # Unknown engine type — acknowledge + dead-letter so the same
            # bad pending update doesn't stick around firing this warning
            # at 30 Hz. (Copilot 3253293637.)
            _log.warning(
                "cloud_weight_update_unknown_engine_type",
                engine_type=update.engine_type,
                repo_id=update.repo_id,
                revision=update.revision,
            )
            poller.acknowledge_swap(update)
            return False

        if reset_recurrent_state:
            # Use ``zeros_like`` so device + dtype are preserved. The live
            # world-model may run on CUDA; ``torch.zeros(...)`` with default
            # device would silently move state back to CPU and break the
            # next ``observe_step`` with a device-mismatch error.
            # (Copilot 3253293626 / 3253309982.)
            self._h = torch.zeros_like(self._h)
            self._z = torch.zeros_like(self._z)
            self._prev_action = torch.zeros_like(self._prev_action)
            self._latent_buffer.clear()

        if self._metrics is not None:
            self._metrics.inc_cloud_weight_update_swap(update.engine_type)

        _log.info(
            "cloud_weight_update_swap_applied",
            repo_id=update.repo_id,
            revision=update.revision,
            engine_type=update.engine_type,
            reset_state=reset_recurrent_state,
        )
        poller.acknowledge_swap(update)
        return reset_recurrent_state

    def _update_world_model(self, observation: ObservationProtocol) -> None:
        """Run world model observation step to update latent state.

        Args:
            observation: Current sensor observation bundle.
        """
        with torch.no_grad():
            self._h, self._z, _, _ = self._world_model.observe_step(
                observation,
                self._prev_action,
                self._h,
                self._z,
            )
        self._h, self._z = self._validate_latent(self._h, self._z)

    def _validate_latent(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Check latent state for NaN / saturation; recover from buffer on NaN.

        Performance: three independent scalars (NaN-in-h, NaN-in-z,
        h-norm) are computed on-device, stacked into a single rank-1
        tensor, and read back with one ``.tolist()`` call. This collapses
        three GPU→CPU syncs down to one on the 30 Hz hot path. Addresses
        review comments on GPU synchronization overhead.

        Args:
            h: Hidden state tensor from ``observe_step``.
            z: Latent state tensor from ``observe_step``.

        Returns:
            ``(h, z)`` — possibly replaced by the last known-good values when
            NaN is detected and the recovery buffer is non-empty.
        """
        # Single GPU→CPU sync covering all three diagnostics. ``stack``
        # forces consistent dtype/device so ``.tolist()`` returns Python
        # floats in one round-trip.
        diagnostics = torch.stack(
            [
                torch.isnan(h).any().to(torch.float32),
                torch.isnan(z).any().to(torch.float32),
                torch.linalg.norm(h.float()),
            ]
        )
        nan_h_float, nan_z_float, h_norm = diagnostics.tolist()
        has_nan = bool(nan_h_float) or bool(nan_z_float)

        if has_nan:
            self._failure_recorder.record(
                "world_model",
                "latent_nan",
                level="critical",
                extra={"tick": self._tick_count},
            )
            _log.critical("world_model_latent_nan", tick=self._tick_count)
            if self._latent_buffer:
                h_last, z_last = self._latent_buffer[-1]
                _log.info("world_model_latent_recovered", tick=self._tick_count)
                return h_last.clone(), z_last.clone()
            _log.critical("world_model_latent_unrecoverable", tick=self._tick_count)
            return h, z

        if h_norm > self._cfg.model.latent_norm_threshold:
            self._failure_recorder.record(
                "world_model",
                "latent_saturated",
                level="warning",
                extra={"h_norm": round(h_norm, 3)},
            )
            _log.warning("world_model_latent_saturated", h_norm=h_norm)

        self._latent_buffer.append((h.clone(), z.clone()))
        return h, z

    def _select_action(
        self,
        safety_ctx: SafetyContext,
        observation: ObservationProtocol,
        loop_time_ms: float,
    ) -> torch.Tensor:
        """Select action using cognitive core (primary) or MCTS agent (fallback).

        Args:
            safety_ctx: Current safety context.
            observation: Current sensor observation.
            loop_time_ms: Current loop timing in milliseconds.

        Returns:
            Action tensor.
        """
        if self._cognitive_core is not None:
            action = self._try_cognitive_action(observation, loop_time_ms)
            if action is not None:
                return action

        # VLA branch (Phase 3a). Default policy_selector='nav_agent'
        # short-circuits this and preserves byte-identical pre-Phase-3a
        # behavior. Only active when the orchestrator was wired with a
        # VLA policy AND the config selects it.
        selector = self._cfg.loop.policy_selector
        if selector != "nav_agent" and self._vla_policy is not None:
            vla_action = self._try_vla_action(observation)
            if vla_action is not None:
                return vla_action
            # Fallthrough to nav_agent below in 'auto' mode, or to a
            # zero-action safe stop in strict 'vla' mode.
            if selector == "vla" and not self._cfg.vla.fallback_on_timeout:
                _log.warning("vla_timeout_safe_stop", selector=selector)
                return torch.zeros(int(self._cfg.model.action_dim), dtype=torch.float32)

        return self._agents[0].act(self._h, self._z, safety_ctx)

    def _maybe_project_action(
        self,
        action: torch.Tensor,
        safety_ctx: SafetyContext,
    ) -> torch.Tensor:
        """Apply the optional geometric safety projector to ``action``.

        Wraps the four return sites of :meth:`_select_action` at a single
        seam in :meth:`tick`. The projector is a soft constraint applied
        AFTER the policy returns — it is the complement of the hard E-stop
        short-circuit at the top of :meth:`tick`.

        When ``self._safety_projector is None`` (the default, gated by
        ``cfg.safety.projector.enabled=False``) this method is a pure
        identity pass-through so existing deployments produce
        byte-identical actions to pre-C2.

        Args:
            action: Proposed action from :meth:`_select_action`. Shape is
                ``(action_dim,)`` or ``(1, action_dim)`` depending on the
                upstream policy branch.
            safety_ctx: Frozen safety context for the current tick.

        Returns:
            Either the original ``action`` (when no projector is wired)
            or a clamped copy with the same shape and dtype.
        """
        if self._safety_projector is None:
            return action

        was_unbatched = action.dim() == 1
        flat: torch.Tensor = action if was_unbatched else action.squeeze(0)
        action_np = flat.detach().cpu().numpy().astype(np.float32, copy=False)
        # ``project()`` is typed to return ``NDArray[np.float32]`` and the
        # implementation guarantees that dtype, so no defensive
        # ``np.asarray(..., dtype=np.float32)`` cast is required here.
        projected_np = self._safety_projector.project(action_np, safety_ctx)
        projected = torch.from_numpy(projected_np).to(flat.device)
        if not was_unbatched:
            projected = projected.unsqueeze(0)
        return projected

    def _try_vla_action(
        self,
        observation: ObservationProtocol,
    ) -> torch.Tensor | None:
        """Run a single VLA inference under the per-tick latency budget.

        Returns ``None`` when the policy raises, when inference exceeds
        ``cfg.loop.inference_timeout_s`` (defaulting to
        ``1.0 / cfg.loop.control_hz``), or when the resulting action
        tensor has the wrong shape. The orchestrator decides how to
        respond to ``None`` (fall back to nav_agent or emit a safe
        stop) via ``policy_selector``/``vla.fallback_on_timeout``.

        Args:
            observation: Current sensor observation (unused by MockVLA;
                forwarded for richer policies).

        Returns:
            The VLA action tensor on success, otherwise ``None``.
        """
        del observation  # forwarded via VLAObservation below
        assert self._vla_policy is not None  # narrowed by caller

        from mousedroid.vla.policy import VLAObservation

        budget = self._cfg.loop.inference_timeout_s
        if budget is None:
            budget = 1.0 / float(self._cfg.loop.control_hz)

        start = self._clock.monotonic()
        try:
            with torch.no_grad():
                result = self._vla_policy.predict(VLAObservation(h=self._h, z=self._z))
        except Exception as exc:  # never let VLA crash the loop
            # Surface the exception type so dashboards can distinguish
            # CUDA-OOM from logic errors at a glance (Gemini review).
            self._failure_recorder.record(
                "orchestrator",
                "vla_exception",
                level="warning",
                extra={"error": type(exc).__name__},
            )
            _log.warning("vla_predict_failed", policy=self._vla_policy.name, exc_info=True)
            return None

        elapsed = self._clock.monotonic() - start
        if elapsed > budget:
            self._failure_recorder.record(
                "orchestrator",
                "vla_timeout",
                level="warning",
                extra={"elapsed_s": round(elapsed, 4), "budget_s": round(budget, 4)},
            )
            if self._metrics is not None:
                # Cast is safe: the policy_selector gate at the caller
                # guarantees ``self._vla_policy is not None``, which means
                # ``cfg.vla.backend != "none"``. The runtime value belongs
                # to :data:`VLAActiveBackendLiteral` by construction; mypy
                # can't see the upstream gate so we cast explicitly.
                from mousedroid.config.schema import VLAActiveBackendLiteral

                self._metrics.inc_vla_timeout(cast(VLAActiveBackendLiteral, self._cfg.vla.backend))
            _log.warning(
                "vla_inference_timeout",
                policy=self._vla_policy.name,
                elapsed_s=elapsed,
                budget_s=budget,
                mode=self._cfg.vla.backend,
            )
            return None

        action_dim = int(self._cfg.model.action_dim)
        if result.action.shape != (action_dim,):
            # Record the full tensor shape (stringified for Prometheus
            # label-friendliness) so dashboards distinguish 0-D outputs
            # from rank-2 outputs like ``(1, action_dim)``.
            self._failure_recorder.record(
                "orchestrator",
                "vla_wrong_shape",
                level="warning",
                extra={
                    "expected": str((action_dim,)),
                    "got": str(tuple(result.action.shape)),
                },
            )
            _log.warning(
                "vla_action_shape_mismatch",
                policy=self._vla_policy.name,
                expected=(action_dim,),
                got=tuple(result.action.shape),
            )
            return None

        return result.action

    def _try_cognitive_action(
        self,
        observation: ObservationProtocol,
        loop_time_ms: float,
    ) -> torch.Tensor | None:
        """Attempt action selection via cognitive core.

        Args:
            observation: Current sensor observation.
            loop_time_ms: Current loop timing in milliseconds.

        Returns:
            Action tensor if successful, None on failure.
        """
        try:
            battery_v = (
                float(observation.motor_state[MOTOR_STATE_BATTERY_INDEX])
                if observation.motor_state.size > MOTOR_STATE_BATTERY_INDEX
                else DEFAULT_BATTERY_VOLTAGE
            )
            belief_dim = int(self._cfg.model.belief_dim)
            bdi_state_vec = self._h.numpy().flatten().astype(np.float32, copy=False)
            state_vec: NDArray[np.float32] = bdi_state_vec
            if state_vec.size < belief_dim:
                state_vec = np.pad(state_vec, (0, belief_dim - state_vec.size))
            else:
                state_vec = state_vec[:belief_dim]
            state_vec = state_vec.astype(np.float32, copy=False)

            obs_dict: dict[str, object] = {
                "state": state_vec,
                "bdi_state": bdi_state_vec,
                "battery_v": battery_v,
                "obstacle_dist_m": float(observation.distance_m),
                "mcts_sims": int(self._cfg.mcts.n_simulations_base),
                "loop_time_ms": loop_time_ms,
                "curiosity": self._compute_curiosity_scores(),
            }
            cognitive_core = self._cognitive_core
            assert cognitive_core is not None
            action_np, violations = cognitive_core.tick_fast(obs_dict)
            if violations:
                _log.info(
                    "orchestrator_constitutional_violations_summary",
                    violation_count=len(violations),
                    violations=violations,
                )

            return self._normalize_cognitive_action(action_np)
        except Exception as e:  # pylint: disable=broad-except
            # Surface the exception type so dashboards can distinguish
            # the failure mode (Gemini review).
            self._failure_recorder.record(
                "orchestrator",
                "cognitive_core_exception",
                level="warning",
                extra={"error": type(e).__name__},
            )
            _log.warning(
                "cognitive_core_action_selection_failed",
                error=str(e),
                falling_back_to_mcts=True,
            )
            return None

    def _normalize_cognitive_action(
        self,
        action_np: NDArray[np.float32] | NDArray[np.float64],
    ) -> torch.Tensor:
        """Normalize cognitive core action to match expected action_dim.

        Args:
            action_np: Raw action from cognitive core.

        Returns:
            Normalized 1-D torch tensor with correct dimensions.
        """
        return normalize_action_numpy(action_np, int(self._cfg.model.action_dim))

    async def _execute_action(self, action: torch.Tensor) -> None:
        """Scale and send action to ESP32 motors.

        Args:
            action: Action tensor with values in [-1, 1].
        """
        max_v = self._cfg.esp32.max_velocity_mps
        max_omega = self._cfg.esp32.max_omega_rads
        vx = float(action[0]) * max_v
        vy = float(action[1]) * max_v if action.shape[0] > 1 else 0.0
        omega = float(action[2]) * max_omega if action.shape[0] > 2 else 0.0
        await self._esp32.send_velocity(vx, vy, omega)

    async def _publish_telemetry(
        self,
        observation: ObservationProtocol,
        safety_ctx: SafetyContext,
        loop_time_ms: float,
    ) -> None:
        """Build and publish a telemetry frame from current state.

        Non-blocking: silently drops if queue is full.

        Args:
            observation: Current sensor observation bundle.
            safety_ctx: Current safety context.
            loop_time_ms: Control loop iteration time (ms).
        """
        if self._telemetry_publisher is None and self._cloud_sink is None:
            return

        try:
            frame = build_telemetry_frame(
                observation,
                safety_ctx,
                loop_time_ms,
                self._tick_count,
                liveness_tracker=self._liveness_tracker,
                now_s=self._clock.monotonic(),
            )
            if self._telemetry_publisher is not None:
                await self._telemetry_publisher.publish(frame)
                # PR #4: also publish the latest raw LiDAR scan to the
                # streaming channel when both the publisher and the
                # sensor manager expose them.
                await self._publish_raw_lidar()
            if self._cloud_sink is not None:
                await self._cloud_sink.publish_telemetry(frame.to_dict())
        except Exception:
            _log.debug("telemetry_publish_failed", exc_info=True)

    async def _publish_raw_lidar(self) -> None:
        """Publish the latest raw LiDAR scan to the streaming channel.

        No-op when:

        * The publisher does not expose ``publish_lidar_raw`` (legacy
          publishers without raw-LiDAR support).
        * The sensor manager has no ``last_lidar_scan`` (LiDAR not
          configured or no scan yet).

        Exceptions are swallowed and logged at DEBUG so a publisher
        backpressure event never crashes the 30 Hz control loop.
        """
        publish_raw = getattr(self._telemetry_publisher, "publish_lidar_raw", None)
        if publish_raw is None:
            return
        scan_source = getattr(self._sensor_manager, "last_lidar_scan", None)
        if scan_source is None:
            return
        try:
            from mousedroid.telemetry.protocol import lidar_scan_to_raw

            raw = lidar_scan_to_raw(scan_source)
        except Exception:
            _log.debug("lidar_raw_conversion_failed", exc_info=True)
            return
        try:
            await publish_raw(raw)
        except Exception:
            _log.debug("lidar_raw_publish_failed", exc_info=True)

    async def _voice_lifecycle(self, event: str) -> None:
        """Fire a lifecycle voice event (startup/shutdown) without an observation.

        Args:
            event: Lifecycle event name (e.g. ``"startup"``, ``"shutdown"``).
        """
        if self._voice_engine is None:
            return
        try:
            await self._voice_engine.speak(event, {"valence": 1.0})
        except Exception:
            _log.warning("voice_lifecycle_failed", voice_event=event, exc_info=True)

    async def _voice_event(
        self,
        event: str,
        observation: ObservationProtocol,
        **extra_context: float,
    ) -> None:
        """Fire a voice event if the voice engine is active.

        Enriches context with sensor data (distance, LiDAR min, audio RMS)
        so the voice engine can modulate speech accordingly.

        Non-blocking: delegates to the engine's async queue.

        Args:
            event: Semantic event name.
            observation: Current sensor observation for context.
            **extra_context: Additional key-value context for the voice engine.
        """
        if self._voice_engine is None:
            return
        context = {"distance_m": float(observation.distance_m)}

        # Enrich with LiDAR minimum distance if features are available
        lidar_features = observation.lidar_features
        if lidar_features is not None and lidar_features.size > 0:
            context["lidar_min_dist_m"] = float(np.min(lidar_features))

        # Enrich with audio level RMS if audio chunk is available
        audio_chunk = observation.audio_chunk
        if audio_chunk is not None and audio_chunk.size > 0:
            context["audio_level_rms"] = float(np.sqrt(np.mean(audio_chunk**2)))

        context.update(extra_context)
        try:
            await self._voice_engine.speak(event, context)
        except Exception:
            _log.warning("voice_event_failed", voice_event=event, exc_info=True)

    async def _voice_observe(
        self,
        observation: ObservationProtocol,
        safety_ctx: SafetyContext,
    ) -> None:
        """Derive voice events from the current observation and safety state.

        Checks safety thresholds from config to avoid hardcoded values.

        Args:
            observation: Current sensor observation.
            safety_ctx: Current safety context.
        """
        if self._voice_engine is None:
            return

        if not safety_ctx.forward_clearance_ok:
            await self._voice_event("obstacle_detected", observation)
        elif safety_ctx.battery_voltage < self._cfg.safety.battery_warn_v:
            await self._voice_event(
                "low_battery",
                observation,
                battery_v=safety_ctx.battery_voltage,
            )
        elif safety_ctx.gpu_temp_c >= self._cfg.safety.gpu_warn_temp_c:
            await self._voice_event(
                "error",
                observation,
                gpu_temp_c=safety_ctx.gpu_temp_c,
            )

    async def _update_face(
        self,
        *,
        safety_ctx: SafetyContext,
        action: torch.Tensor | None,
    ) -> None:
        """Drive the face controller from BDI affect + safety state.

        Pulls ``(valence, arousal)`` from
        :meth:`CognitiveCore.get_latest_affect`, defaulting to neutral when
        the slow loop has not produced a result yet. Idle is inferred from
        ``action`` magnitude — emergency-stop callers pass ``action=None``.

        Args:
            safety_ctx: Current safety context (provides ``is_emergency``).
            action: Most recent commanded action, or ``None`` in the
                emergency path.
        """
        if self._face_controller is None:
            return

        if self._cognitive_core is not None:
            valence, arousal = self._cognitive_core.get_latest_affect()
        else:
            valence, arousal = 0.0, 0.0

        # When emergency wins, is_idle is irrelevant — skip the .item() call
        # to avoid an unnecessary GPU↔CPU sync on the hot path.
        if safety_ctx.is_emergency:
            is_idle = False
        elif action is None:
            is_idle = True
        else:
            epsilon = self._cfg.face_display.idle_action_epsilon if self._cfg.face_display else 1e-3
            is_idle = bool(action.abs().max().item() < epsilon)

        try:
            await self._face_controller.update(
                valence=valence,
                arousal=arousal,
                is_emergency=safety_ctx.is_emergency,
                is_idle=is_idle,
            )
        except Exception as exc:  # pylint: disable=broad-except
            _log.warning(
                "face_controller_update_failed",
                exc_type=type(exc).__name__,
                exc_info=True,
            )

    async def _try_sensor_recovery(self, safety_ctx: SafetyContext) -> bool:
        """Attempt sensor recovery if the emergency is due to sensor degradation.

        Only runs when valid_sensor_count is below threshold and the
        configured recovery_attempts > 0.

        Args:
            safety_ctx: Current safety context.

        Returns:
            True if a recovery was attempted, False otherwise.
        """
        max_attempts = self._cfg.safety.sensor_recovery_attempts
        if max_attempts <= 0:
            return False
        if safety_ctx.valid_sensor_count >= self._cfg.safety.min_valid_sensors:
            return False

        _log.warning(
            "sensor_recovery_starting",
            valid_sensors=safety_ctx.valid_sensor_count,
            required=self._cfg.safety.min_valid_sensors,
            max_attempts=max_attempts,
        )

        for attempt in range(max_attempts):
            recovered = await self._sensor_manager.recovery_attempt()
            if recovered > 0:
                _log.info(
                    "sensor_recovery_success",
                    attempt=attempt + 1,
                    recovered=recovered,
                )
                return True
            if attempt < max_attempts - 1:
                await self._clock.sleep(self._cfg.safety.sensor_recovery_delay_s)

        _log.error("sensor_recovery_exhausted", attempts=max_attempts)
        return False

    def _log_experience(
        self,
        observation: ObservationProtocol,
        action: torch.Tensor,
    ) -> None:
        """Log experience to memory tier and LMDB.

        Builds a ``MouseDroidExperienceRecord`` from the current observation
        and action, then pushes it to episodic replay, working memory, and
        the persistent experience logger.

        Args:
            observation: Current sensor observation.
            action: Action tensor just executed.
        """
        if self._memory_tier is None and self._experience_logger is None:
            return

        from mousedroid.experience.record import MouseDroidExperienceRecord

        action_np = action.detach().cpu().numpy().flatten().astype(np.float32)
        record = MouseDroidExperienceRecord(
            vision_features=observation.vision_features,
            distance_m=float(observation.distance_m),
            motor_state=observation.motor_state,
            action=action_np,
        )

        # Compute intrinsic reward for surprise-based prioritization
        surprise = 0.0
        if self._curiosity_module is not None:
            with torch.no_grad():
                s = self._h.flatten().unsqueeze(0)
                a = action.unsqueeze(0) if action.dim() == 1 else action
                s_next = self._z.flatten().unsqueeze(0)
                intrinsic = self._curiosity_module.intrinsic_reward(s, a, s_next)
                surprise = float(intrinsic.item())
        record.surprise = surprise

        if self._memory_tier is not None:
            min_priority = self._cfg.memory.min_episodic_priority
            self._memory_tier.episodic.push(record, priority=max(surprise, min_priority))
            latent = self._h.detach().clone()
            self._memory_tier.working.push(latent)

        if self._experience_logger is not None:
            record.reward = surprise
            self._experience_logger.log(record)

        if self._cloud_sink is not None:
            spawn_tracked(
                self._cloud_publish_tasks,
                self._cloud_sink.publish_experience(record),
                name="cloud_publish_experience",
            )

    def _compute_curiosity_scores(self) -> dict[str, float]:
        """Compute curiosity channel scores for cognitive core obs_dict.

        Returns:
            Dictionary with 'intrinsic' and 'epistemic' curiosity channels.
        """
        scores: dict[str, float] = {"intrinsic": 0.0, "epistemic": 0.0}

        if self._curiosity_module is not None:
            with torch.no_grad():
                s = self._h.flatten().unsqueeze(0)
                a = self._prev_action
                s_next = self._z.flatten().unsqueeze(0)
                intrinsic = self._curiosity_module.intrinsic_reward(s, a, s_next)
                scores["intrinsic"] = float(intrinsic.item())

        if self._memory_tier is not None and self._memory_tier.semantic.size > 0:
            query = self._h.detach().cpu().numpy().flatten().astype(np.float32)
            k = self._cfg.memory.semantic_retrieve_k
            results = self._memory_tier.semantic.retrieve(query, k=k)
            if results:
                _, distance = results[0]
                scores["epistemic"] = float(distance)

        return scores

    async def _consolidation_loop(self) -> None:
        """Background loop that consolidates episodic memory into semantic index.

        Runs at the interval specified by ``cfg.memory.consolidation_interval_s``.
        Automatically cancelled by ``stop()``.
        """
        interval = self._cfg.memory.consolidation_interval_s
        _log.info("consolidation_loop_started", interval_s=interval)
        while True:
            await self._clock.sleep(interval)
            if self._memory_tier is None:
                break
            try:
                count = await asyncio.to_thread(self._memory_tier.consolidation.consolidate)
                if count > 0:
                    _log.debug(
                        "consolidation_cycle_complete",
                        records_consolidated=count,
                        semantic_size=self._memory_tier.semantic.size,
                    )
            except Exception:
                _log.warning("consolidation_cycle_failed", exc_info=True)

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
