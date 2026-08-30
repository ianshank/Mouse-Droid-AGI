"""Factory builder — composes the production MouseDroidOrchestrator from every other domain builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from mousedroid.logging.setup import get_logger
from pathlib import Path
from typing import cast
import numpy as np
from mousedroid.hardware.protocols import (
    DistanceSensorProtocol,
    LidarProtocol,
    VisionProtocol,
)
from mousedroid.cloud.protocol import ENGINE_TYPE_WORLD_MODEL
from mousedroid.llm_gateway.protocol import LLMGatewayProtocol

from mousedroid.factory.telemetry import (
    build_mock_telemetry_source,
    _compose_weight_update_loader,
)
from mousedroid.factory.llm_gateway import build_injection_filter
from mousedroid.factory.voice import _build_orchestrator_greeter
from mousedroid.factory.mission import build_vlm_progress

if TYPE_CHECKING:
    import torch
    from torch import Tensor

    from mousedroid.agents.base import AgentProtocol
    from mousedroid.arm.protocols import (
        ArmControllerProtocol,
        ArmDriverProtocol,
        ArmEnvironmentProtocol,
        ArmPerceptionProtocol,
        ArmPlannerProtocol,
        SymbolicPlannerBackend,
    )
    from mousedroid.cloud.protocol import (
        CloudExperienceExporterProtocol,
        CloudFirestoreSyncProtocol,
        CloudLoggingSinkProtocol,
        CloudMetricsExporterProtocol,
        CloudTelemetrySinkProtocol,
        PendingWeightUpdate,
        WeightUpdatePollerProtocol,
    )
    from mousedroid.cognitive.bdi_model import NeuralBDI
    from mousedroid.cognitive.cognitive_core import CognitiveCore
    from mousedroid.common.time.protocol import ClockProtocol
    from mousedroid.common.tools.registry import ToolRegistry
    from mousedroid.config.schema import (
        ESP32Config,
        LLMConfig,
        ModelConfig,
        Settings,
        UltrasonicConfig,
    )
    from mousedroid.curiosity.protocol import CuriosityProtocol
    from mousedroid.efficiency.tensorrt import TensorRTCompilerProtocol
    from mousedroid.experience.logger import ExperienceLogger
    from mousedroid.experience.record import MouseDroidExperienceRecord
    from mousedroid.growth.coordinator import GrowthDistillationCoordinator
    from mousedroid.hardware.accelerator.hailo_runtime import HailoRuntimeProtocol
    from mousedroid.hardware.camera.feature_extractor import FeatureExtractorProtocol
    from mousedroid.harness.approval.protocol import ApprovalGateProtocol
    from mousedroid.harness.protocol import TaskTrackerProtocol
    from mousedroid.health.monitor import HealthMonitor
    from mousedroid.learning.on_device.hot_swap import OnDeviceWeightUpdateSource
    from mousedroid.learning.on_device.replay_trigger import ReplayTriggerCoordinator
    from mousedroid.learning.on_device.slot_store import CandidateSlot, OnDeviceSlotStore
    from mousedroid.llm_gateway.mission_parser import MissionParserProtocol
    from mousedroid.mcp.protocol import MCPServerProtocol
    from mousedroid.memory.episodic import EpisodicReplay
    from mousedroid.memory.tier import MemoryTier
    from mousedroid.orchestrator.face_controller import FaceController
    from mousedroid.orchestrator.mission_dispatcher import MissionDispatcherProtocol
    from mousedroid.orchestrator.mission_lifecycle import (
        MissionLifecycle,
        MissionReplannerProtocol,
    )
    from mousedroid.reward.protocol import RewardModelProtocol
    from mousedroid.reward.vlm_progress import VLMProgressHead
    from mousedroid.sensing.manager import SensorManager
    from mousedroid.sim.protocols import RoverEnvProtocol
    from mousedroid.telemetry.failure_recorder import FailureRecorder
    from mousedroid.telemetry.log_buffer import LogRingBuffer
    from mousedroid.telemetry.metrics import MetricsRegistry
    from mousedroid.telemetry.protocol import TelemetryPublisherProtocol, TelemetryServerProtocol
    from mousedroid.training.observability import ExperimentLoggerProtocol
    from mousedroid.training.replay import ReplayReaderProtocol
    from mousedroid.training.replay.lmdb_reader import LMDBReplayReader
    from mousedroid.vla.policy import VLAPolicyProtocol
    from mousedroid.voice.greeting import Greeter, GreeterProtocol
    from mousedroid.voice.mock_tts import MockTTS
    from mousedroid.voice.tts import PiperTTS
    from mousedroid.world_model.encoder import MultimodalEncoder
    from mousedroid.world_model.protocol import LatentContextProtocol, WorldModelProtocol
    from mousedroid.world_model.rssm import RSSM

from mousedroid.factory.hardware import (
    build_audio_feature_extractor,
    build_camera,
    build_distance_sensor,
    build_esp32_driver,
    build_face_display,
    build_hailo_runtime,
    build_lidar,
    build_lidar_feature_extractor,
    build_microphone,
    build_sensor_manager,
    build_tensorrt_compiler,
)
from mousedroid.factory.voice import (
    build_face_controller,
    build_greeter,
    build_speaker,
    build_voice_engine,
)
from mousedroid.factory.world_model import (
    build_latent_context,
    build_rover_env,
    build_vision_feature_extractor,
    build_world_model,
)
from mousedroid.factory.llm_gateway import build_llm_gateway
from mousedroid.factory.learning import build_replay_reader, build_reward_model, build_vla_policy
from mousedroid.factory.mission import (
    build_mission_lifecycle,
    build_mission_parser,
    build_mission_replanner,
)
from mousedroid.factory.telemetry import (
    build_experiment_logger,
    build_failure_recorder,
    build_metrics_registry,
    build_telemetry_publisher,
    build_telemetry_server,
    build_weight_update_loader,
    build_weight_update_pollers,
)
from mousedroid.factory.safety import build_safety_monitor, build_safety_projector
from mousedroid.factory.health import build_health_monitor, build_watchdog
from mousedroid.factory.cognitive import build_agent, build_cognitive_core
from mousedroid.factory.memory_curiosity import (
    build_curiosity_module,
    build_experience_logger,
    build_memory_tier,
)
from mousedroid.factory.cloud import (
    build_cloud_experience_exporter,
    build_cloud_firestore_sync,
    build_cloud_logging_sink,
    build_cloud_metrics_exporter,
    build_cloud_telemetry_sink,
)
from mousedroid.factory.arm import (
    build_arm_controller,
    build_arm_driver,
    build_arm_environment,
    build_arm_perception,
    build_arm_planner,
    build_symbolic_planner_backend,
)
from mousedroid.factory.on_device_learning import (
    build_on_device_coordinator,
    build_on_device_hot_swap_source,
)
from mousedroid.factory.growth import build_growth_coordinator
from mousedroid.factory.mcp_harness import (
    build_approval_gate,
    build_builtin_skills,
    build_hook_registry,
    build_journal,
    build_mcp_server,
    build_memory_exporter,
    build_skill_delegator,
    build_skill_loaders,
    build_skill_registry,
    build_task_tracker,
)

_log = get_logger(__name__)


def build_orchestrator(cfg: Settings) -> object:
    """Build fully-wired orchestrator.

    Args:
        cfg: Root settings.

    Returns:
        Fully configured ``MouseDroidOrchestrator``.
    """
    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    # Build Hailo-8 runtime early — shared by camera and arm perception
    hailo_runtime = build_hailo_runtime(cfg)

    wm = build_world_model(cfg)
    agent = build_agent(cfg, wm)
    monitor = build_safety_monitor(cfg)
    esp32 = build_esp32_driver(cfg)

    camera: VisionProtocol | None = None
    try:
        camera = build_camera(cfg, hailo_runtime=hailo_runtime)
    except Exception as exc:
        _log.warning("camera_init_failed_degrading", error=str(exc))

    distance: DistanceSensorProtocol | None = None
    try:
        distance = build_distance_sensor(cfg)
    except Exception as exc:
        _log.warning("distance_sensor_init_failed_degrading", error=str(exc))

    microphone = build_microphone(cfg)

    lidar_driver: LidarProtocol | None = None
    try:
        lidar_driver = build_lidar(cfg)
    except Exception as exc:
        _log.warning("lidar_init_failed_degrading", error=str(exc))

    sensor_manager = build_sensor_manager(
        cfg,
        vision=camera,
        distance=distance,
        esp32=esp32,
        microphone=microphone,
        lidar=lidar_driver,
    )

    cognitive_core: CognitiveCore | None = None
    if cfg.cognitive.enabled:
        try:
            cognitive_core = build_cognitive_core(cfg)
        except Exception as e:
            if cfg.cognitive.fallback_to_mcts:
                _log.warning(
                    "cognitive_core_init_failed_falling_back_to_mcts",
                    error=str(e),
                )
            else:
                raise

    # Telemetry (optional — disabled by default)
    telemetry_publisher = build_telemetry_publisher(cfg)
    health_monitor = build_health_monitor(cfg)

    # PR #4: per-sensor liveness tracker — declared once at build time
    # so the orchestrator can attach a four-state liveness map
    # (disabled / awaiting / live / stale) to every telemetry frame.
    # Disabled when telemetry itself is off (no consumers).
    liveness_tracker = None
    if cfg.telemetry.enabled:
        from mousedroid.telemetry.sensor_liveness import SensorLivenessTracker

        liveness_tracker = SensorLivenessTracker(
            stale_s=cfg.telemetry.sensor_liveness_stale_s,
        )
        liveness_tracker.register("lidar", enabled=lidar_driver is not None)
        liveness_tracker.register("vision", enabled=camera is not None)
        liveness_tracker.register("audio", enabled=microphone is not None)
        liveness_tracker.register("motor", enabled=True)
        _log.info(
            "sensor_liveness_tracker_built",
            stale_s=cfg.telemetry.sensor_liveness_stale_s,
            lidar_enabled=lidar_driver is not None,
            vision_enabled=camera is not None,
            audio_enabled=microphone is not None,
        )

    # PR #4: synthetic telemetry source for mock_hardware mode (None in
    # production). Lifecycle is owned by the orchestrator.
    mock_telemetry_source = build_mock_telemetry_source(cfg, telemetry_publisher)

    # Optional log ring buffer for telemetry log streaming
    from mousedroid.telemetry.log_buffer import LogRingBuffer as _LogRingBuffer

    log_buffer: _LogRingBuffer | None = None
    buffer_size = cfg.telemetry.log_stream_buffer
    if buffer_size:
        log_buffer = _LogRingBuffer(buffer_size)

    metrics_registry = build_metrics_registry(cfg)
    failure_recorder = build_failure_recorder(cfg, metrics_registry)

    # Shared prompt-injection filter — reused by the LLM gateway and the
    # OpenClaw mission dispatcher so REST + MCP + LLM ingress share one
    # rejection envelope. Built before the telemetry server because the
    # dispatcher (which the server registers POST /api/v1/mission against)
    # consumes it.
    injection_filter = build_injection_filter(cfg)

    # OpenClaw mission dispatcher (None when openclaw is disabled). The
    # deferred orchestrator ref is bound just before returning the
    # orchestrator at the end of this function.
    from mousedroid.factory import build_approval_gate
    from mousedroid.orchestrator.mission_dispatcher import build_mission_dispatcher

    approval_gate = build_approval_gate(cfg)

    mission_dispatcher, deferred_orchestrator_ref = build_mission_dispatcher(
        cfg.openclaw,
        approval_gate=approval_gate,
        injection_filter=injection_filter,
    )

    telemetry_server = build_telemetry_server(
        cfg,
        telemetry_publisher,
        health_monitor,
        log_buffer=log_buffer,
        metrics_registry=metrics_registry,
        camera=camera,
        mission_dispatcher=mission_dispatcher,
    )

    # LLM gateway + mission parser (optional — gated by llm.enabled)
    llm_gateway: LLMGatewayProtocol | None = None
    if cfg.llm.enabled:
        llm_gateway = build_llm_gateway(
            cfg, injection_filter=injection_filter, metrics=metrics_registry
        )
    mission_parser: MissionParserProtocol | None = build_mission_parser(cfg)

    # VLA policy (optional — gated by vla.backend, default 'none')
    vla_policy: VLAPolicyProtocol | None = build_vla_policy(cfg)

    from mousedroid.common.tools.registry import create_default_registry

    _tool_registry = create_default_registry(
        llm_gateway=llm_gateway,
        metrics_registry=metrics_registry,
        gcp_cfg=cfg.gcp,
    )

    # Motor tools — only registered for the rover platform; the arm
    # platform has its own actuation surface and shouldn't expose
    # rover-specific velocity controls. Import is local so static
    # analysers don't drag the runtime symbol into the TYPE_CHECKING
    # block above.
    from mousedroid.config.schema import PlatformType as _Platform

    if cfg.platform == _Platform.MOUSE_DROID:
        from mousedroid.common.tools.motor_tools import MotorToolDeps, register_motor_tools

        register_motor_tools(
            _tool_registry,
            MotorToolDeps(esp32=esp32, cfg=cfg),
        )

    # MCP server (optional — disabled by default). Built after the tool
    # registry so it can bridge real tools, and after telemetry/log
    # buffers so it can expose them as resources. Memory tier is wired
    # below; when present and enabled, the server gets a non-None
    # ``memory_tier``.
    speaker = build_speaker(cfg, metrics=metrics_registry)
    voice_engine = build_voice_engine(
        cfg, speaker=speaker, failure_recorder=failure_recorder, metrics=metrics_registry
    )

    # Issue #109 — one-shot startup greeter (None unless cfg.greeting is
    # enabled). Reuses the orchestrator's own voice engine so a single
    # engine serves both the greeting and normal voice output.
    startup_greeter = _build_orchestrator_greeter(cfg, voice_engine)

    # Face display (optional — disabled by default)
    face_display = build_face_display(cfg)
    face_controller = build_face_controller(cfg, face_display)

    # Memory tier + experience logger + curiosity module (optional)
    memory_tier = build_memory_tier(cfg)
    experience_logger = build_experience_logger(cfg)
    curiosity_module = build_curiosity_module(cfg)

    # OpenClaw MEMORY.md exporter (None unless cfg.openclaw.shared_memory_path set)
    memory_exporter = build_memory_exporter(cfg)

    mcp_server = build_mcp_server(
        cfg,
        tool_registry=_tool_registry,
        safety_monitor=monitor,
        publisher=telemetry_publisher,
        log_buffer=log_buffer,
        metrics_registry=metrics_registry,
        memory_tier=memory_tier,
    )

    # Watchdog notifier (optional — disabled by default)
    watchdog = build_watchdog(cfg)

    # GCP Digital Twin (optional — disabled when gcp=None)
    cloud_sink = build_cloud_telemetry_sink(cfg, metrics_registry=metrics_registry)
    cloud_experience_exporter = build_cloud_experience_exporter(cfg)
    cloud_metrics_exporter = build_cloud_metrics_exporter(cfg, metrics_registry=metrics_registry)
    cloud_firestore_sync = build_cloud_firestore_sync(
        cfg,
        episodic=None if memory_tier is None else memory_tier.episodic,
    )

    # Agent harness (all opt-in; passing None preserves byte-identical legacy behaviour)
    task_tracker = build_task_tracker(cfg)
    journal = build_journal(cfg)
    approval_gate = build_approval_gate(cfg)
    skill_loaders = build_skill_loaders(cfg)
    skill_registry = build_skill_registry(cfg, skill_loaders)
    skill_delegator = build_skill_delegator(
        cfg,
        skill_registry,
        approval_gate,
        journal,
        task_tracker,
        llm_gateway=llm_gateway,
    )
    # Wire the approval gate onto the tool registry so dispatched tools
    # flagged with ``requires_approval=True`` are gated through it.
    if hasattr(_tool_registry, "set_approval_gate"):
        _tool_registry.set_approval_gate(approval_gate)
    hook_registry = build_hook_registry(cfg, journal)

    # Tier C1 / C1.2 — wire the optional OTA weight-update pollers. Default
    # ``cfg.cloud.weight_update.poll_interval_s = 0.0`` keeps the mapping
    # empty so the orchestrator's swap helper short-circuits and existing
    # deployments remain byte-identical. When polling is enabled the
    # ``policy`` poller is always present; the ``world_model`` poller is
    # added iff ``cfg.cloud.weight_update.world_model_enabled is True``.
    weight_update_pollers = build_weight_update_pollers(cfg, metrics=metrics_registry)
    weight_update_loader = build_weight_update_loader(cfg)
    # Tier C2 / C2.1 — soft-constraint safety projector. Returns ``None``
    # when ``cfg.safety.projector.enabled`` is ``False`` (the default),
    # which makes the orchestrator's projection seam a no-op so pre-C2
    # deployments produce byte-identical actions.
    safety_projector = build_safety_projector(cfg, metrics=metrics_registry)
    # Tier C2.3 — build VLM head + LLM replanner so the lifecycle is no
    # longer the defensive None from PR #98. Both build_* helpers short-
    # circuit on their own flags so this remains byte-identical to
    # pre-Tier-C2.3 behaviour when the new ``cfg.mission.vlm_progress_enabled``
    # and ``cfg.mission.llm_replanner_enabled`` flags stay at False.
    vlm_progress = build_vlm_progress(cfg)
    mission_replanner = build_mission_replanner(
        cfg,
        llm_gateway=llm_gateway,
        metrics=metrics_registry,
    )
    # Tier C2 / C2.2 — mission lifecycle state machine. Returns ``None``
    # in four scenarios so the orchestrator's POST_TICK seam stays a no-op
    # and pre-Tier-C2.3 deployments are byte-identical:
    #   1. ``cfg.mission.replan_enabled`` is ``False`` (the default).
    #   2. ``cfg.mission.vlm_progress_enabled`` is ``False`` (vlm_progress=None).
    #   3. ``cfg.mission.llm_replanner_enabled`` is ``False`` OR
    #      ``cfg.llm.enabled`` is ``False`` (mission_replanner=None).
    #   4. Either of the above missing — defensive guard inside
    #      :func:`build_mission_lifecycle` (PR #98 Copilot HIGH fix).
    mission_lifecycle = build_mission_lifecycle(
        cfg,
        task_tracker=task_tracker,
        vlm_progress=vlm_progress,
        replanner=mission_replanner,
        metrics=metrics_registry,
    )

    # Phase 6 WS3/WS4 — replay-trigger on-device update coordinator. ``None``
    # when ``cfg.on_device_learning`` is absent/disabled (orchestrator
    # byte-identical to pre-WS3); otherwise a slow-cadence background task
    # producing stamped candidate slots and (WS4) running the safety-regression
    # gate. The shared metrics registry is threaded so the WS4 revert counter
    # surfaces on ``/metrics``; the already-built live world model ``wm`` is
    # threaded so the gate scores against the REAL model architecture (Phase-6
    # ENABLEMENT) instead of a wrong-arch ``RSSM(cfg.model)`` literal.
    on_device_coordinator = build_on_device_coordinator(
        cfg, metrics=metrics_registry, world_model=wm
    )

    # Growth pillar — slow-cadence VLA knowledge distillation. ``None`` when
    # ``cfg.growth`` is absent/disabled OR no VLA teacher is wired (orchestrator
    # byte-identical to pre-feature); otherwise a background task distilling the
    # live ``vla_policy`` teacher into a compact student over real on-policy
    # latents rolled from the live world model ``wm``. The shared metrics registry
    # is threaded so the distillation counter surfaces on ``/metrics``. The
    # distilled student is persisted to a SHA-256 slot, never hot-swapped.
    growth_coordinator = build_growth_coordinator(
        cfg, metrics=metrics_registry, vla_policy=vla_policy, world_model=wm
    )

    # Phase 6 WS-E4 — off-loop hot-swap of a PROMOTED on-device slot. ``None``
    # (default) when ``cfg.on_device_learning.enable_hot_swap`` is ``False`` — NO
    # swap path is wired AT ALL, so the orchestrator is byte-identical to #134
    # (promotion via ``mark_active`` stays SEPARATE from activation). When
    # enabled, the source watches the active slot on its OWN slow-cadence task,
    # materialises the device-correct engine OFF the hot loop, and surfaces it as
    # a ``world_model`` ``PendingWeightUpdate`` so the existing C1 atomic-swap
    # seam (``_apply_pending_weight_update``) applies it as a PURE reference
    # assignment. The live ``wm`` is threaded so the swap engine lands on the
    # SAME device + architecture (device-parity contract).
    on_device_hot_swap_source = build_on_device_hot_swap_source(
        cfg, world_model=wm, metrics=metrics_registry
    )
    if on_device_hot_swap_source is not None:
        merged_pollers: dict[str, WeightUpdatePollerProtocol] = dict(weight_update_pollers)
        if ENGINE_TYPE_WORLD_MODEL in merged_pollers:
            # A cloud OTA world-model poller and the on-device hot-swap source
            # both target the ``world_model`` engine slot; they are mutually
            # exclusive activation paths. The on-device source takes the slot
            # (the rover production overlay disables cloud OTA, so this only fires
            # in a misconfigured both-on deployment — logged loudly, never silent).
            _log.warning(
                "on_device_hot_swap_supersedes_cloud_world_model_poller",
                reason="enable_hot_swap=true claims the world_model swap slot",
            )
        merged_pollers[ENGINE_TYPE_WORLD_MODEL] = on_device_hot_swap_source
        weight_update_pollers = merged_pollers
        # Compose the loader so the hot-loop swap returns the source's PRE-
        # materialised engine for on-device updates (pure reference return, no
        # tick I/O) and delegates any cloud update to the cloud loader.
        weight_update_loader = _compose_weight_update_loader(
            weight_update_loader, on_device_hot_swap_source
        )

    orchestrator = MouseDroidOrchestrator(
        world_model=wm,
        agents=[agent],
        safety_monitor=monitor,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cognitive_core=cognitive_core,
        cfg=cfg,
        telemetry_publisher=telemetry_publisher,
        telemetry_server=telemetry_server,
        voice_engine=voice_engine,
        hailo_runtime=hailo_runtime,
        memory_tier=memory_tier,
        experience_logger=experience_logger,
        curiosity_module=curiosity_module,
        llm_gateway=llm_gateway,
        mission_parser=mission_parser,
        watchdog=watchdog,
        cloud_sink=cloud_sink,
        cloud_experience_exporter=cloud_experience_exporter,
        cloud_metrics_exporter=cloud_metrics_exporter,
        cloud_firestore_sync=cloud_firestore_sync,
        tool_registry=_tool_registry,
        face_controller=face_controller,
        mcp_server=mcp_server,
        vla_policy=vla_policy,
        hook_registry=hook_registry,
        task_tracker=task_tracker,
        journal=journal,
        skill_delegator=skill_delegator,
        memory_exporter=memory_exporter,
        mission_dispatcher=mission_dispatcher,
        failure_recorder=failure_recorder,
        liveness_tracker=liveness_tracker,
        mock_telemetry_source=mock_telemetry_source,
        metrics=metrics_registry,
        weight_update_pollers=weight_update_pollers,
        weight_update_loader=weight_update_loader,
        safety_projector=safety_projector,
        mission_lifecycle=mission_lifecycle,
        on_device_coordinator=cast("ReplayTriggerCoordinator | None", on_device_coordinator),
        growth_coordinator=cast("GrowthDistillationCoordinator | None", growth_coordinator),
        latent_context=build_latent_context(cfg),
        greeter=startup_greeter,
    )
    # Bind the deferred orchestrator reference so the OpenClaw mission
    # dispatcher (built before the orchestrator above) can route through
    # ``orchestrator.process_mission`` from this point onwards.
    if deferred_orchestrator_ref is not None:
        deferred_orchestrator_ref.bind(orchestrator)
    return orchestrator


# ---------------------------------------------------------------------------
# Robot Arm platform factory functions
# ---------------------------------------------------------------------------
