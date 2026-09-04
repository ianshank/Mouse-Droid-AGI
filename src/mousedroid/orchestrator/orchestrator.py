"""MouseDroid orchestrator — main sense-plan-act loop.

Platform-agnostic via dependency injection. All components injected
through constructor, wired by factory functions.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, Any

import torch

from mousedroid.cloud.protocol import ENGINE_TYPE_POLICY
from mousedroid.common.time.protocol import ClockProtocol, RealClock
from mousedroid.comms.command_set import command_set_supports_lateral
from mousedroid.harness.protocol import HookPhase, TickContext
from mousedroid.logging.setup import get_logger
from mousedroid.orchestrator._action_mixin import _ActionMixin
from mousedroid.orchestrator._background_cadence_mixin import _BackgroundCadenceMixin
from mousedroid.orchestrator._lifecycle_mixin import _LifecycleMixin
from mousedroid.orchestrator._mission_mixin import _MissionMixin
from mousedroid.orchestrator._telemetry_experience_mixin import _TelemetryExperienceMixin
from mousedroid.orchestrator._voice_face_mixin import _VoiceFaceMixin
from mousedroid.orchestrator._world_model_state_mixin import _WorldModelStateMixin

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mousedroid.agents.base import AgentProtocol
    from mousedroid.cloud.protocol import (
        CloudExperienceExporterProtocol,
        CloudFirestoreSyncProtocol,
        CloudMetricsExporterProtocol,
        CloudTelemetrySinkProtocol,
        PendingWeightUpdate,
        WeightUpdatePollerProtocol,
    )
    from mousedroid.cognitive.cognitive_core import CognitiveCore
    from mousedroid.comms.protocol import ESP32CommProtocol
    from mousedroid.config.schema import Settings
    from mousedroid.curiosity.protocol import CuriosityProtocol
    from mousedroid.experience.logger import ExperienceLogger
    from mousedroid.growth.coordinator import GrowthDistillationCoordinator
    from mousedroid.hardware.accelerator.hailo_runtime import HailoRuntimeProtocol
    from mousedroid.harness.journal.protocol import JournalProtocol
    from mousedroid.harness.protocol import (
        HookRegistryProtocol,
        TaskTrackerProtocol,
    )
    from mousedroid.health.watchdog import WatchdogProtocol
    from mousedroid.learning.on_device.replay_trigger import ReplayTriggerCoordinator
    from mousedroid.llm_gateway.mission_parser import MissionParserProtocol
    from mousedroid.llm_gateway.protocol import LLMGatewayProtocol
    from mousedroid.mcp.protocol import MCPServerProtocol
    from mousedroid.memory.exporter import MemoryExporterProtocol
    from mousedroid.memory.tier import MemoryTier
    from mousedroid.orchestrator.face_controller import FaceController
    from mousedroid.orchestrator.mission_dispatcher import MissionDispatcherProtocol
    from mousedroid.orchestrator.mission_lifecycle import MissionLifecycle
    from mousedroid.safety.projector_protocol import SafetyActionProjectorProtocol
    from mousedroid.safety.protocol import SafetyMonitorProtocol
    from mousedroid.sensing.manager import SensorManager
    from mousedroid.skills.delegator import SkillDelegator
    from mousedroid.telemetry.failure_recorder import FailureRecorder
    from mousedroid.telemetry.metrics import MetricsRegistry
    from mousedroid.telemetry.protocol import TelemetryPublisherProtocol, TelemetryServerProtocol
    from mousedroid.vla.policy import VLAPolicyProtocol
    from mousedroid.voice.greeting import GreeterProtocol
    from mousedroid.voice.protocol import VoiceEngineProtocol
    from mousedroid.world_model.protocol import LatentContextProtocol, WorldModelProtocol

_log = get_logger(__name__)


class MouseDroidOrchestrator(
    _LifecycleMixin,
    _MissionMixin,
    _WorldModelStateMixin,
    _ActionMixin,
    _TelemetryExperienceMixin,
    _VoiceFaceMixin,
    _BackgroundCadenceMixin,
):
    """Main orchestrator — runs the sense-plan-act loop.

    All components are injected via constructor. No hardcoded types.
    Composed via mixins for lifecycle, mission, world-model state,
    action selection, telemetry/experience, voice/face, and background
    cadence operations.
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
        cloud_metrics_exporter: CloudMetricsExporterProtocol | None = None,
        cloud_firestore_sync: CloudFirestoreSyncProtocol | None = None,
        *,
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
        mission_lifecycle: MissionLifecycle | None = None,
        on_device_coordinator: ReplayTriggerCoordinator | None = None,
        growth_coordinator: GrowthDistillationCoordinator | None = None,
        latent_context: LatentContextProtocol | None = None,
        greeter: GreeterProtocol | None = None,
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
            cloud_metrics_exporter: Optional Cloud Monitoring metrics exporter.
            cloud_firestore_sync: Optional Firestore episodic-memory sync.
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
                ``engine_type`` property (falling back to the legacy private
                ``_engine_type`` attribute, then to
                :data:`mousedroid.cloud.protocol.ENGINE_TYPE_POLICY`) so
                internal handling is uniform between the two shapes. Prefer
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
            mission_lifecycle: Optional :class:`MissionLifecycle` (Tier
                C2 / C2.2). When supplied, the orchestrator drives a
                single ``lifecycle.tick(obs_t, obs_tminus1)`` call at the
                POST_TICK seam each tick, after telemetry and just before
                the POST_TICK hook fires. The helper caches the previous
                tick's ``observation.vision_features`` so the lifecycle's
                two-frame contract is honoured. ``None`` (default, gated
                by ``cfg.mission.replan_enabled=false``) makes the seam a
                no-op so existing deployments are byte-identical.
            on_device_coordinator: Optional Phase-6 WS3
                :class:`ReplayTriggerCoordinator`. When supplied AND
                ``cfg.on_device_learning.enabled`` is ``True``, the
                orchestrator spawns a slow-cadence background task in
                ``start()`` that checks the fresh-record count and produces +
                persists a SHA-256-stamped candidate weight slot (the torch
                update is offloaded via ``asyncio.to_thread``; the 30 Hz hot
                loop is never touched). ``None`` (default) — or
                ``cfg.on_device_learning`` absent/disabled — makes the
                orchestrator byte-identical to pre-WS3 (no extra task spawned).
                WS3 only PRODUCES the candidate; promotion into the live
                policy is WS4's safety-gated decision.
            growth_coordinator: Optional growth-pillar
                :class:`GrowthDistillationCoordinator`. When supplied AND
                ``cfg.growth.enabled`` is ``True``, the orchestrator spawns a
                slow-cadence background task in ``start()`` that distils the VLA
                teacher into a compact student and persists a SHA-256-stamped
                slot (all torch work is offloaded via ``asyncio.to_thread``; the
                30 Hz hot loop is never touched). ``None`` (default) — or
                ``cfg.growth`` absent/disabled, or no VLA teacher wired — makes
                the orchestrator byte-identical to pre-feature (no extra task).
                Distillation only PRODUCES a student slot; deploying it is a
                separate soak-gated operator decision.
            latent_context: Optional :class:`LatentContextProtocol` (F-023,
                ADR-015). When supplied (factory gates on
                ``cfg.world_model_memory.enabled``), the tick's
                ``_update_world_model`` stores the RAW validated ``(h, z)``
                and blends the attention-retrieved context into the carried
                state — a pure deterministic ``no_grad`` tensor op. Unhealthy
                (non-finite) ticks skip both calls so today's NaN
                self-healing is preserved. ``None`` (default) keeps the tick
                path byte-identical to pre-feature. The memory resets with
                the OTA weight-swap seam and re-arms its sink at mission
                boundaries (``recapture_on_mission``).
            greeter: Optional :class:`GreeterProtocol` (Issue #109). When
                supplied AND ``cfg.greeting`` is enabled with
                ``fire_on_startup=True``, the orchestrator fires the
                greeting ONCE during ``start()`` — before entering the
                30 Hz loop, after the voice engine it shares is started.
                The call is wrapped in try/except so a greeting failure is
                logged (``greeting_startup_failed``) and swallowed; it
                never blocks startup. ``None`` (default) — or the flag
                left ``False`` — keeps the hot loop byte-identical (the
                startup greeting is a one-shot OUTSIDE the loop).
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
        # F-025: resolved once (the tick path runs at 30 Hz). Owned by the
        # codec so the "which axes exist" answer lives in exactly one place.
        self._supports_lateral: bool = command_set_supports_lateral(cfg.esp32)
        self._cloud_sink = cloud_sink
        self._cloud_experience_exporter = cloud_experience_exporter
        self._cloud_metrics_exporter = cloud_metrics_exporter
        self._cloud_firestore_sync = cloud_firestore_sync
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
        # ``engine_type`` property so the rest of the orchestrator only ever
        # sees the mapping shape. Precedence is keyed on whether the
        # mapping kwarg was *provided* (not whether it is non-empty) so an
        # explicit ``weight_update_pollers={}`` cleanly disables OTA
        # without being silently overridden by a legacy single-poller arg.
        self._weight_update_pollers: dict[str, WeightUpdatePollerProtocol] = dict(
            weight_update_pollers or {}
        )
        if weight_update_poller is not None and weight_update_pollers is None:
            # Fall back to the typed ``engine_type`` property; the legacy
            # ``_engine_type`` private attribute fallback is retained for one
            # release only as a safety net for external pollers that
            # implement the protocol structurally but predate the property
            # addition.
            engine_type = getattr(
                weight_update_poller,
                "engine_type",
                getattr(weight_update_poller, "_engine_type", ENGINE_TYPE_POLICY),
            )
            self._weight_update_pollers[engine_type] = weight_update_poller
        elif weight_update_poller is not None and weight_update_pollers is not None:
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
        # Tier C2 / C2.2 — optional mission lifecycle state machine. ``None``
        # (gated by ``cfg.mission.replan_enabled=False``) makes
        # ``_maybe_tick_mission_lifecycle`` a no-op. The previous tick's
        # vision-feature tensor is cached between ticks so the lifecycle
        # receives the (obs_t, obs_tminus1) pair it expects; the first
        # tick after wiring populates the cache and skips the lifecycle.
        self._mission_lifecycle = mission_lifecycle
        # Phase 6 WS3 — replay-triggered on-device update coordinator. ``None``
        # (default) AND/OR ``cfg.on_device_learning`` absent/disabled keeps the
        # orchestrator byte-identical to pre-WS3: no slow task is spawned in
        # ``start()`` and ``stop()`` has nothing extra to cancel. When wired AND
        # enabled, the coordinator runs on its own slow-cadence background task
        # OUTSIDE the 30 Hz hot loop (torch work offloaded via asyncio.to_thread).
        self._on_device_coordinator = on_device_coordinator
        self._on_device_task: asyncio.Task[Any] | None = None
        self._on_device_tasks: set[asyncio.Task[Any]] = set()
        # Growth pillar — slow-cadence VLA distillation background task (None when
        # cfg.growth is absent/disabled OR no VLA teacher wired).
        self._growth_coordinator = growth_coordinator
        self._growth_task: asyncio.Task[Any] | None = None
        self._growth_tasks: set[asyncio.Task[Any]] = set()
        # Issue #109 — one-shot startup greeting. ``None`` (default) or
        # ``cfg.greeting.fire_on_startup=False`` keeps ``start()``
        # byte-identical; the greeting never touches the 30 Hz loop.
        self._greeter: GreeterProtocol | None = greeter
        self._prev_obs_for_vlm: torch.Tensor | None = None
        # Per-mission monotonic counter used to build collision-free
        # mission IDs in ``process_mission``. Decoupled from ``_tick_count``
        # so back-to-back process_mission calls (between control-loop ticks,
        # or before the loop starts) cannot generate duplicate IDs that
        # silently overwrite ``MissionLifecycle._mission`` state.
        self._mission_seq: int = 0
        self._running = False
        self._tick_count: int = 0
        self._consolidation_task: asyncio.Task[Any] | None = None
        self._consolidation_tasks: set[asyncio.Task[Any]] = set()
        # Strong-reference set for fire-and-forget cloud publishes. Keeping
        # the reference prevents premature GC of the asyncio.Task; entries
        # are evicted by spawn_tracked's done-callback as tasks resolve.
        self._cloud_publish_tasks: set[asyncio.Task[Any]] = set()

        # Latent state (combined_dim = hidden_dim + cfc_hidden_dim for dual-stream)
        _combined_hidden_dim = cfg.model.hidden_dim + cfg.model.cfc_hidden_dim
        self._h = torch.zeros(1, _combined_hidden_dim)
        # Previous tick's measured total duration, fed to this tick's safety
        # evaluation and metrics. 0.0 on tick 0 so the first evaluation cannot
        # trip -- there is no previous tick to have overrun.
        self._last_tick_ms: float = 0.0
        # Resolved once: `_mark_phase` runs 8x per tick at 30 Hz, so this
        # avoids a config attribute walk 240 times a second.
        self._tick_phase_timing_enabled: bool = cfg.metrics.track_tick_phases
        self._z = torch.zeros(1, cfg.model.latent_dim)
        self._prev_action = torch.zeros(1, cfg.model.action_dim)
        # Rolling buffer of (h, z) tuples for latent NaN recovery.
        self._latent_buffer: deque[tuple[torch.Tensor, torch.Tensor]] = deque(
            maxlen=cfg.model.latent_recovery_buffer_size
        )
        # F-023 bounded-context latent memory. ``None`` (default) keeps the
        # tick path byte-identical to pre-feature; the recovery buffer above
        # keeps holding RAW (pre-blend) states so NaN recovery restores the
        # unblended state.
        self._latent_context = latent_context

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
        # The safety interlock and the Prometheus gauge are fed the PREVIOUS
        # tick's *total* duration. A tick cannot know its own total until it
        # ends, and the value this code used to pass -- the sensor-read segment
        # measured before planning, actuation and telemetry ran -- made the
        # loop-overrun e-stop and `mousedroid_loop_time_ms` blind to the phases
        # most likely to blow the budget. Tick 0 sees 0.0 and is structurally
        # incapable of tripping, which is correct: there is no previous tick.
        prev_tick_ms = self._last_tick_ms
        ok = False
        try:
            observation = await self._sensor_manager.read_all()
            mark = self._mark_phase("sense", loop_start)

            safety_ctx = self._safety_monitor.evaluate(
                observation, prev_tick_ms, tick_index=self._tick_count
            )
            mark = self._mark_phase("safety", mark)

            self._update_world_model(observation)
            mark = self._mark_phase("world_model", mark)

            ctx.observation = observation
            ctx.safety_ctx = safety_ctx
            ctx.loop_time_ms = prev_tick_ms
            if self._task_tracker is not None:
                ctx.active_tasks = tuple(s.id for s in self._task_tracker.active())
            await self._hook_registry.run_phase(HookPhase.PRE_TICK, ctx)

            if safety_ctx.is_emergency:
                # Attempt sensor recovery before emergency stop if sensors degraded
                if await self._try_sensor_recovery(safety_ctx):
                    # Re-read after recovery — sensors may have come back.
                    # Reuse prev_tick_ms: re-reading sensors inside THIS tick
                    # cannot change how long the PREVIOUS tick took, and the
                    # same tick_index keeps the overrun streak from counting
                    # this tick twice.
                    observation = await self._sensor_manager.read_all()
                    safety_ctx = self._safety_monitor.evaluate(
                        observation, prev_tick_ms, tick_index=self._tick_count
                    )
                    ctx.observation = observation
                    ctx.safety_ctx = safety_ctx
                    ctx.loop_time_ms = prev_tick_ms

                if safety_ctx.is_emergency:
                    await self._esp32.emergency_stop()
                    await self._voice_event("emergency_stop", observation)
                    await self._update_face(safety_ctx=safety_ctx, action=None)
                    _log.warning("emergency_stop_triggered")
                    self._tick_count += 1
                    await self._publish_telemetry(observation, safety_ctx, prev_tick_ms)
                    # Tier C2 / C2.2 (Copilot MED follow-up): drive the
                    # mission lifecycle on the emergency-stop branch too
                    # so active missions keep accumulating progress/stall
                    # state during emergency ticks. Without this, a
                    # stuck-emergency condition would freeze the
                    # lifecycle's stall counter and silently extend any
                    # in-flight mission past its stall window. The helper
                    # is a no-op when no lifecycle is wired, so pre-C2.2
                    # deployments are byte-identical.
                    await self._maybe_tick_mission_lifecycle(observation)
                    await self._hook_registry.run_phase(HookPhase.POST_TICK, ctx)
                    # An emergency tick RAN TO COMPLETION — it just took a
                    # shorter route. Its duration is a real control-loop
                    # latency sample and must reach Prometheus, or the loop
                    # timing goes dark for exactly as long as the emergency
                    # condition persists, which is when it is most needed.
                    # ``ok`` answers "did this tick raise", not "was the rover
                    # healthy"; the error and cancellation paths below are the
                    # ones telemetry invariant 5 excludes.
                    #
                    # ``post`` closes the last open bracket so the phases tile
                    # here as they do on the full path. On THIS path it spans a
                    # wider slice — PRE_TICK hooks, any sensor recovery and
                    # re-evaluation, the e-stop write, voice, face, telemetry,
                    # lifecycle, POST_TICK hooks. ``act`` stays deliberately
                    # unrecorded: it means "execute the selected action", and
                    # an emergency stop is the override, not a selection.
                    self._mark_phase("post", mark)
                    ok = True
                    return

            action = self._select_action(safety_ctx, observation, prev_tick_ms)
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
            # Restrict to axes the chassis can actually execute BEFORE the
            # action is recorded as ``_prev_action``, sent, and logged, so the
            # world model, curiosity, ``executed_action`` and the experience
            # log all describe the motion that really happened. Projecting
            # only at the dispatch seam would still feed the unprojected
            # lateral to ``observe_step`` on the next tick.
            executable = self._project_action_to_executable_axes(action)
            swap_reset = self._apply_pending_weight_update()
            if not swap_reset:
                self._prev_action = executable.unsqueeze(0) if executable.dim() == 1 else executable

            # PRE_ACTION hooks see the raw proposal — a safety hook inspecting
            # what the policy WANTED must not be shown a value the projection
            # already zeroed.
            ctx.proposed_action = action
            await self._hook_registry.run_phase(HookPhase.PRE_ACTION, ctx)
            mark = self._mark_phase("plan", mark)

            action = executable
            await self._execute_action(action)
            ctx.executed_action = action
            await self._hook_registry.run_phase(HookPhase.POST_ACTION, ctx)
            mark = self._mark_phase("act", mark)

            self._log_experience(observation, action)
            await self._voice_observe(observation, safety_ctx)
            await self._update_face(safety_ctx=safety_ctx, action=action)
            mark = self._mark_phase("learn", mark)

            self._tick_count += 1
            await self._publish_telemetry(observation, safety_ctx, prev_tick_ms)
            mark = self._mark_phase("telemetry", mark)

            if self._task_tracker is not None:
                await self._task_tracker.evaluate_active(ctx)

            # Tier C2 / C2.2 — drive the optional mission lifecycle once per
            # tick. No-op when no lifecycle is wired or the previous tick's
            # vision-feature cache is unpopulated; failures are logged and
            # swallowed so the control loop never crashes on lifecycle bugs.
            # Runs AFTER task_tracker.evaluate_active so the tracker observes
            # active tasks BEFORE the lifecycle potentially transitions them
            # to terminal (SUCCEEDED/FAILED) states — preventing double-count
            # or stale timeout enforcement on just-completed tasks.
            await self._maybe_tick_mission_lifecycle(observation)

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
            self._maybe_rearm_latent_sink(mission_completed=mission_completed)
            self._mark_phase("post", mark)
            ok = True

            _log.debug(
                "tick_complete",
                loop_time_ms=prev_tick_ms,
                emergency=safety_ctx.is_emergency,
            )
        except Exception as exc:
            ctx.error = exc
            # Hooks observe the error before we re-raise. If a hook itself
            # raises, the HookRegistry's error_policy decides whether to
            # propagate (default: warn-and-continue).
            await self._hook_registry.run_phase(HookPhase.ON_ERROR, ctx)
            raise
        finally:
            # Latch the true duration on EVERY exit path — success, the
            # emergency branch's early return, and the error path. A tick that
            # is chronically slow because it keeps failing must not stay
            # invisible to the next tick's interlock just because it never
            # reached the end of the method.
            self._finish_tick_timing(loop_start, ok=ok)
