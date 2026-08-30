"""Platform factory functions — build all components via dependency injection.

Factory functions eliminate platform branching. Each ``build_*()`` function
returns the correct implementation based on ``Settings``.

This package is the single factory-first DI composition root (Invariant #1 from
CLAUDE.md). Concrete types are imported inside package modules only; application
code is typed against @runtime_checkable Protocol interfaces.
"""

from __future__ import annotations

# Runtime imports (visible in dir())
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

from mousedroid.cloud.protocol import (
    ENGINE_TYPE_POLICY,
    ENGINE_TYPE_WORLD_MODEL,
)
from mousedroid.common.imports import module_available, module_importable
from mousedroid.comms.protocol import ESP32CommProtocol
from mousedroid.constants import (
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_LIDAR_BUFFER_SIZE,
    DEFAULT_LIDAR_MAX_RANGE_M,
    DEFAULT_MOTOR_BAUDRATE,
)
from mousedroid.hardware.protocols import (
    AudioProtocol,
    DistanceSensorProtocol,
    FaceDisplayProtocol,
    LidarProtocol,
    SpeakerProtocol,
    VisionProtocol,
)
from mousedroid.health.watchdog import WatchdogProtocol
from mousedroid.interfaces.protocols import (
    CameraProtocol,
    LiDARProtocol,
    MetricsRegistryProtocol,
    MotorControllerProtocol,
)
from mousedroid.llm_gateway.protocol import LLMGatewayProtocol
from mousedroid.logging.redaction import redact_uri_credentials, redact_uris_in_text
from mousedroid.logging.setup import get_logger
from mousedroid.safety.projector_protocol import SafetyActionProjectorProtocol
from mousedroid.safety.protocol import SafetyMonitorProtocol
from mousedroid.security.injection_filter import (
    PromptInjectionFilterProtocol,
    RegexInjectionFilter,
)
from mousedroid.voice.protocol import VoiceEngineProtocol

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

# Module-level constants and logger
_log = get_logger(__name__)

_CoroResult = TypeVar("_CoroResult")

_SLOT_PT_SUFFIX: str = ".pt"

_MAX_REPLAY_COUNT_CHUNK: int = 1000

# Import all factory builders from submodules
from mousedroid.factory.hardware import (
    build_esp32_driver,
    _resolve_esp32_serial_via_usbc_discovery,
    build_camera,
    build_distance_sensor,
    build_microphone,
    build_face_display,
    build_sensor_manager,
    build_audio_feature_extractor,
    build_lidar,
    build_lidar_feature_extractor,
    build_tensorrt_compiler,
    build_hailo_runtime,
)
from mousedroid.factory.voice import (
    build_face_controller,
    build_speaker,
    build_greeter,
    _build_orchestrator_greeter,
    build_voice_engine,
)
from mousedroid.factory.world_model import (
    build_world_model,
    build_latent_context,
    build_rssm_trainable,
    build_rssm_vision_finetune,
    build_vision_feature_extractor,
    _build_onnx_world_model,
    _resolve_world_model_onnx_path,
    build_rover_env,
)
from mousedroid.factory.llm_gateway import (
    build_injection_filter,
    _build_single_llm_gateway,
    build_llm_gateway,
)
from mousedroid.factory.learning import (
    build_vla_policy,
    _build_distilled_onnx_vla,
    build_reward_model,
    build_replay_reader,
)
from mousedroid.factory.mission import (
    build_mission_parser,
    build_vlm_progress,
    build_mission_replanner,
    build_mission_lifecycle,
)
from mousedroid.factory.telemetry import (
    _PINNED_URI_SCHEMES,
    _IN_MEMORY_SQLITE_PATHS,
    build_metrics_registry,
    build_experiment_logger,
    _resolve_tracking_uri,
    build_weight_update_pollers,
    build_weight_update_loader,
    _compose_weight_update_loader,
    build_failure_recorder,
    build_telemetry_publisher,
    build_telemetry_server,
    build_mock_telemetry_source,
)
from mousedroid.factory.safety import (
    build_safety_monitor,
    build_safety_projector,
)
from mousedroid.factory.health import (
    build_health_monitor,
    build_watchdog,
)
from mousedroid.factory.cognitive import (
    build_agent,
    _resolve_bdi_weights,
    build_cognitive_core,
)
from mousedroid.factory.memory_curiosity import (
    build_memory_tier,
    build_experience_logger,
    build_curiosity_module,
)
from mousedroid.factory.cloud import (
    build_cloud_telemetry_sink,
    build_cloud_experience_exporter,
    build_cloud_logging_sink,
    build_cloud_metrics_exporter,
    build_cloud_firestore_sync,
)
from mousedroid.factory.arm import (
    build_arm_driver,
    build_symbolic_planner_backend,
    build_arm_planner,
    build_arm_environment,
    build_arm_controller,
    build_arm_perception,
    build_llm_replanner,
)
from mousedroid.factory.autonomous import (
    build_autonomous_metrics_registry,
    build_motor_controller,
    build_autonomous_camera,
    build_autonomous_lidar,
    build_autonomous_orchestrator,
)
from mousedroid.factory._replay_batch_helpers import (
    _make_consumed_offset_advancer,
    _build_shared_replay_reader,
    _run_coro_blocking,
    _count_replay_records,
    _count_new_replay_records,
    _load_replay_batch,
    _load_replay_sequence_batch,
    _build_held_out_sequence_batch,
    _MAX_REPLAY_COUNT_CHUNK as _MAX_REPLAY_COUNT_CHUNK,
)
from mousedroid.factory.on_device_learning import (
    build_on_device_coordinator,
    build_on_device_hot_swap_source,
    _build_on_device_gate_runner,
)
from mousedroid.factory.growth import (
    _make_growth_latent_sampler,
    build_growth_coordinator,
)
from mousedroid.factory.mcp_harness import (
    build_mcp_server,
    build_task_tracker,
    build_journal,
    _resolve_approval_callback,
    build_approval_gate,
    build_skill_loaders,
    build_memory_exporter,
    build_builtin_skills,
    build_skill_registry,
    _build_sub_agent_factory,
    build_skill_delegator,
    build_hook_registry,
)
from mousedroid.factory.orchestrator import (
    build_orchestrator,
)

# Import submodules explicitly so they can be deleted from the namespace
from mousedroid.factory import (
    arm,
    autonomous,
    cloud,
    cognitive,
    growth,
    hardware,
    health,
    learning,
    llm_gateway,
    mcp_harness,
    memory_curiosity,
    mission,
    on_device_learning,
    orchestrator,
    safety,
    telemetry,
    voice,
    world_model,
    _replay_batch_helpers,
)

# Hide submodule names from dir() to match pre-split flat module's dir() output.
# Python's import mechanism exposes submodules automatically; we delete them here
# to maintain Invariant #1's pre/post decomposition dir() equivalence.
del arm, autonomous, cloud, cognitive, growth, hardware, health, learning
del llm_gateway, mcp_harness, memory_curiosity, mission, on_device_learning
del orchestrator, safety, telemetry, voice, world_model, _replay_batch_helpers

__all__ = [
    "Any",
    "Awaitable",
    "Callable",
    "Coroutine",
    "DEFAULT_CAMERA_HEIGHT",
    "DEFAULT_CAMERA_WIDTH",
    "DEFAULT_LIDAR_BUFFER_SIZE",
    "DEFAULT_LIDAR_MAX_RANGE_M",
    "DEFAULT_MOTOR_BAUDRATE",
    "ENGINE_TYPE_POLICY",
    "ENGINE_TYPE_WORLD_MODEL",
    "ESP32CommProtocol",
    "LLMGatewayProtocol",
    "LiDARProtocol",
    "LidarProtocol",
    "Mapping",
    "Path",
    "PromptInjectionFilterProtocol",
    "RegexInjectionFilter",
    "SafetyActionProjectorProtocol",
    "SafetyMonitorProtocol",
    "TypeVar",
    "TYPE_CHECKING",
    "VisionProtocol",
    "WatchdogProtocol",
    "_log",
    "_CoroResult",
    "_SLOT_PT_SUFFIX",
    "_MAX_REPLAY_COUNT_CHUNK",
    "build_agent",
    "build_approval_gate",
    "build_arm_controller",
    "build_arm_driver",
    "build_arm_environment",
    "build_arm_perception",
    "build_arm_planner",
    "build_audio_feature_extractor",
    "build_autonomous_camera",
    "build_autonomous_lidar",
    "build_autonomous_metrics_registry",
    "build_autonomous_orchestrator",
    "build_builtin_skills",
    "build_camera",
    "build_cloud_experience_exporter",
    "build_cloud_firestore_sync",
    "build_cloud_logging_sink",
    "build_cloud_metrics_exporter",
    "build_cloud_telemetry_sink",
    "build_cognitive_core",
    "build_curiosity_module",
    "build_distance_sensor",
    "build_esp32_driver",
    "build_experience_logger",
    "build_experiment_logger",
    "build_face_controller",
    "build_face_display",
    "build_failure_recorder",
    "build_greeter",
    "build_growth_coordinator",
    "build_hailo_runtime",
    "build_health_monitor",
    "build_hook_registry",
    "build_injection_filter",
    "build_journal",
    "build_latent_context",
    "build_lidar",
    "build_lidar_feature_extractor",
    "build_llm_gateway",
    "build_llm_replanner",
    "build_mcp_server",
    "build_memory_exporter",
    "build_memory_tier",
    "build_metrics_registry",
    "build_microphone",
    "build_mission_lifecycle",
    "build_mission_parser",
    "build_mission_replanner",
    "build_mock_telemetry_source",
    "build_motor_controller",
    "build_on_device_coordinator",
    "build_on_device_hot_swap_source",
    "build_orchestrator",
    "build_replay_reader",
    "build_reward_model",
    "build_rover_env",
    "build_rssm_trainable",
    "build_rssm_vision_finetune",
    "build_safety_monitor",
    "build_safety_projector",
    "build_sensor_manager",
    "build_skill_delegator",
    "build_skill_loaders",
    "build_skill_registry",
    "build_speaker",
    "build_symbolic_planner_backend",
    "build_task_tracker",
    "build_telemetry_publisher",
    "build_telemetry_server",
    "build_tensorrt_compiler",
    "build_vision_feature_extractor",
    "build_vla_policy",
    "build_vlm_progress",
    "build_voice_engine",
    "build_watchdog",
    "build_weight_update_loader",
    "build_weight_update_pollers",
    "build_world_model",
    "cast",
    "get_logger",
    "module_available",
    "module_importable",
    "redact_uri_credentials",
    "redact_uris_in_text",
    "_PINNED_URI_SCHEMES",
    "_IN_MEMORY_SQLITE_PATHS",
    "AudioProtocol",
    "CameraProtocol",
    "DistanceSensorProtocol",
    "FaceDisplayProtocol",
    "MetricsRegistryProtocol",
    "MotorControllerProtocol",
    "SpeakerProtocol",
    "VoiceEngineProtocol",
    "_build_distilled_onnx_vla",
    "_build_held_out_sequence_batch",
    "_build_on_device_gate_runner",
    "_build_onnx_world_model",
    "_build_orchestrator_greeter",
    "_build_shared_replay_reader",
    "_build_single_llm_gateway",
    "_build_sub_agent_factory",
    "_compose_weight_update_loader",
    "_count_new_replay_records",
    "_count_replay_records",
    "_load_replay_batch",
    "_load_replay_sequence_batch",
    "_make_consumed_offset_advancer",
    "_make_growth_latent_sampler",
    "_resolve_approval_callback",
    "_resolve_bdi_weights",
    "_resolve_esp32_serial_via_usbc_discovery",
    "_resolve_tracking_uri",
    "_resolve_world_model_onnx_path",
    "_run_coro_blocking",
]
