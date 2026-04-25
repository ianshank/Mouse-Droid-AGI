"""MouseDroid orchestrator — main sense-plan-act loop.

Platform-agnostic via dependency injection. All components injected
through constructor, wired by factory functions.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray

from mousedroid.common.actions import normalize_action_numpy
from mousedroid.common.async_utils import cancel_and_drain, spawn_tracked
from mousedroid.constants import (
    DEFAULT_BATTERY_VOLTAGE,
    MILLISECONDS_PER_SECOND,
    MOTOR_STATE_BATTERY_INDEX,
)
from mousedroid.logging.setup import get_logger
from mousedroid.telemetry.frame_builder import build_telemetry_frame

if TYPE_CHECKING:
    from mousedroid.agents.base import AgentProtocol
    from mousedroid.cloud.protocol import (
        CloudExperienceExporterProtocol,
        CloudTelemetrySinkProtocol,
    )
    from mousedroid.cognitive.cognitive_core import CognitiveCore
    from mousedroid.comms.protocol import ESP32CommProtocol
    from mousedroid.config.schema import Settings
    from mousedroid.curiosity.protocol import CuriosityProtocol
    from mousedroid.experience.logger import ExperienceLogger
    from mousedroid.hardware.accelerator.hailo_runtime import HailoRuntimeProtocol
    from mousedroid.health.watchdog import WatchdogProtocol
    from mousedroid.llm_gateway.mission_parser import MissionParserProtocol
    from mousedroid.llm_gateway.protocol import GoalVector, LLMGatewayProtocol
    from mousedroid.memory.tier import MemoryTier
    from mousedroid.safety.context import SafetyContext
    from mousedroid.safety.protocol import SafetyMonitorProtocol
    from mousedroid.sensing.manager import SensorManager
    from mousedroid.sensing.protocol import ObservationProtocol
    from mousedroid.telemetry.protocol import TelemetryPublisherProtocol, TelemetryServerProtocol
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
        self._cfg = cfg
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
            await self._telemetry_server.start()
        if self._llm_gateway is not None:
            try:
                await self._llm_gateway.start()
            except RuntimeError:
                _log.warning("llm_gateway_start_failed", exc_info=True)
        if self._voice_engine is not None:
            await self._voice_engine.start()
            await self._voice_lifecycle("startup")
        if self._experience_logger is not None:
            self._experience_logger.open()
        if self._cloud_sink is not None:
            await self._cloud_sink.start()
        if self._cloud_experience_exporter is not None:
            await self._cloud_experience_exporter.start()
        if self._memory_tier is not None:
            self._consolidation_task = spawn_tracked(
                self._consolidation_tasks,
                self._consolidation_loop(),
                name=self._consolidation_loop.__name__,
            )
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
        if self._cloud_experience_exporter is not None:
            await self._cloud_experience_exporter.close()
        if self._cloud_sink is not None:
            await self._cloud_sink.flush()
            await self._cloud_sink.close()
        if self._experience_logger is not None:
            self._experience_logger.close()
        if self._voice_engine is not None:
            await self._voice_lifecycle("shutdown")
            await self._voice_engine.stop()
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
        _log.info("orchestrator_stopped")

    async def tick(self) -> None:
        """Execute one sense-plan-act cycle."""
        loop_start = time.monotonic()

        observation = await self._sensor_manager.read_all()
        loop_time_ms = (time.monotonic() - loop_start) * MILLISECONDS_PER_SECOND

        safety_ctx = self._safety_monitor.evaluate(observation, loop_time_ms)

        self._update_world_model(observation)

        if safety_ctx.is_emergency:
            # Attempt sensor recovery before emergency stop if sensors degraded
            if await self._try_sensor_recovery(safety_ctx):
                # Re-read after recovery — sensors may have come back
                observation = await self._sensor_manager.read_all()
                loop_time_ms = (time.monotonic() - loop_start) * MILLISECONDS_PER_SECOND
                safety_ctx = self._safety_monitor.evaluate(observation, loop_time_ms)

            if safety_ctx.is_emergency:
                await self._esp32.emergency_stop()
                await self._voice_event("emergency_stop", observation)
                _log.warning("emergency_stop_triggered")
                self._tick_count += 1
                await self._publish_telemetry(observation, safety_ctx, loop_time_ms)
                return

        action = self._select_action(safety_ctx, observation, loop_time_ms)
        self._prev_action = action.unsqueeze(0) if action.dim() == 1 else action

        await self._execute_action(action)
        self._log_experience(observation, action)
        await self._voice_observe(observation, safety_ctx)

        self._tick_count += 1
        await self._publish_telemetry(observation, safety_ctx, loop_time_ms)

        _log.debug(
            "tick_complete",
            loop_time_ms=loop_time_ms,
            emergency=safety_ctx.is_emergency,
        )

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

        return self._agents[0].act(self._h, self._z, safety_ctx)

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
            )
            if self._telemetry_publisher is not None:
                await self._telemetry_publisher.publish(frame)
            if self._cloud_sink is not None:
                spawn_tracked(
                    self._cloud_publish_tasks,
                    self._cloud_sink.publish_telemetry(frame.to_dict()),
                    name="cloud_publish_telemetry",
                )
        except Exception:
            _log.debug("telemetry_publish_failed", exc_info=True)

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
                await asyncio.sleep(self._cfg.safety.sensor_recovery_delay_s)

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
            await asyncio.sleep(interval)
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
            tick_start = time.monotonic()
            try:
                await asyncio.wait_for(self.tick(), timeout=tick_timeout)
            except asyncio.TimeoutError:
                _log.critical(
                    "tick_timeout",
                    timeout_s=tick_timeout,
                    elapsed_s=time.monotonic() - tick_start,
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

            elapsed = time.monotonic() - tick_start
            sleep_time = max(0.0, control_period - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

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
