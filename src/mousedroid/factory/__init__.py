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
from typing import Any, cast

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

# Import submodules explicitly so they can be deleted from the namespace
from mousedroid.factory import (
    _replay_batch_helpers,
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
)
from mousedroid.factory._replay_batch_helpers import (
    _MAX_REPLAY_COUNT_CHUNK as _MAX_REPLAY_COUNT_CHUNK,
)
from mousedroid.factory._replay_batch_helpers import (
    _build_held_out_sequence_batch as _build_held_out_sequence_batch,
)
from mousedroid.factory._replay_batch_helpers import (
    _build_shared_replay_reader as _build_shared_replay_reader,
)
from mousedroid.factory._replay_batch_helpers import (
    _CoroResult as _CoroResult,
)
from mousedroid.factory._replay_batch_helpers import (
    _count_new_replay_records as _count_new_replay_records,
)
from mousedroid.factory._replay_batch_helpers import (
    _count_replay_records as _count_replay_records,
)
from mousedroid.factory._replay_batch_helpers import (
    _load_replay_batch as _load_replay_batch,
)
from mousedroid.factory._replay_batch_helpers import (
    _load_replay_sequence_batch as _load_replay_sequence_batch,
)
from mousedroid.factory._replay_batch_helpers import (
    _make_consumed_offset_advancer as _make_consumed_offset_advancer,
)
from mousedroid.factory._replay_batch_helpers import (
    _run_coro_blocking as _run_coro_blocking,
)
from mousedroid.factory.arm import (
    build_arm_controller,
    build_arm_driver,
    build_arm_environment,
    build_arm_perception,
    build_arm_planner,
    build_llm_replanner,
    build_symbolic_planner_backend,
)
from mousedroid.factory.autonomous import (
    build_autonomous_camera,
    build_autonomous_lidar,
    build_autonomous_metrics_registry,
    build_autonomous_orchestrator,
    build_motor_controller,
)
from mousedroid.factory.cloud import (
    build_cloud_experience_exporter,
    build_cloud_firestore_sync,
    build_cloud_logging_sink,
    build_cloud_metrics_exporter,
    build_cloud_telemetry_sink,
)
from mousedroid.factory.cognitive import (
    _resolve_bdi_weights as _resolve_bdi_weights,
)
from mousedroid.factory.cognitive import (
    build_agent,
    build_cognitive_core,
)
from mousedroid.factory.growth import (
    _make_growth_latent_sampler as _make_growth_latent_sampler,
)
from mousedroid.factory.growth import (
    build_growth_coordinator,
)
from mousedroid.factory.hardware import (
    _resolve_esp32_serial_via_usbc_discovery as _resolve_esp32_serial_via_usbc_discovery,
)

# No TYPE_CHECKING block: this facade re-exports runtime symbols only. The
# old flat factory.py needed one because build_orchestrator (and friends)
# lived here and used these types in their own signatures; every one of
# those functions has since moved to its own submodule (factory/orchestrator.py
# etc.), which imports the types it actually annotates with locally. None of
# this __init__.py's own code (imports, __all__, a docstring) uses any of
# them, so re-declaring the same TYPE_CHECKING block here was dead weight
# left over from the split -- confirmed by both mypy --strict (passes
# without it) and a grep for real usage (none).
# Import all factory builders from submodules
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
from mousedroid.factory.health import (
    build_health_monitor,
    build_watchdog,
)
from mousedroid.factory.learning import (
    _build_distilled_onnx_vla as _build_distilled_onnx_vla,
)
from mousedroid.factory.learning import (
    build_replay_reader,
    build_reward_model,
    build_vla_policy,
)
from mousedroid.factory.llm_gateway import (
    _build_single_llm_gateway as _build_single_llm_gateway,
)
from mousedroid.factory.llm_gateway import (
    build_injection_filter,
    build_llm_gateway,
)
from mousedroid.factory.mcp_harness import (
    _build_sub_agent_factory as _build_sub_agent_factory,
)
from mousedroid.factory.mcp_harness import (
    _resolve_approval_callback as _resolve_approval_callback,
)
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
from mousedroid.factory.memory_curiosity import (
    build_curiosity_module,
    build_experience_logger,
    build_memory_tier,
)
from mousedroid.factory.mission import (
    build_mission_lifecycle,
    build_mission_parser,
    build_mission_replanner,
    build_vlm_progress,
)
from mousedroid.factory.on_device_learning import (
    _SLOT_PT_SUFFIX as _SLOT_PT_SUFFIX,
)
from mousedroid.factory.on_device_learning import (
    _build_on_device_gate_runner as _build_on_device_gate_runner,
)
from mousedroid.factory.on_device_learning import (
    build_on_device_coordinator,
    build_on_device_hot_swap_source,
)
from mousedroid.factory.orchestrator import (
    build_orchestrator,
)
from mousedroid.factory.safety import (
    build_safety_monitor,
    build_safety_projector,
)
from mousedroid.factory.telemetry import (
    _IN_MEMORY_SQLITE_PATHS as _IN_MEMORY_SQLITE_PATHS,
)
from mousedroid.factory.telemetry import (
    _PINNED_URI_SCHEMES as _PINNED_URI_SCHEMES,
)
from mousedroid.factory.telemetry import (
    _compose_weight_update_loader as _compose_weight_update_loader,
)
from mousedroid.factory.telemetry import (
    _resolve_tracking_uri as _resolve_tracking_uri,
)
from mousedroid.factory.telemetry import (
    build_experiment_logger,
    build_failure_recorder,
    build_metrics_registry,
    build_mock_telemetry_source,
    build_telemetry_publisher,
    build_telemetry_server,
    build_weight_update_loader,
    build_weight_update_pollers,
)
from mousedroid.factory.voice import (
    _build_orchestrator_greeter as _build_orchestrator_greeter,
)
from mousedroid.factory.voice import (
    build_face_controller,
    build_greeter,
    build_speaker,
    build_voice_engine,
)
from mousedroid.factory.world_model import (
    _build_onnx_world_model as _build_onnx_world_model,
)
from mousedroid.factory.world_model import (
    _resolve_world_model_onnx_path as _resolve_world_model_onnx_path,
)
from mousedroid.factory.world_model import (
    build_latent_context,
    build_rover_env,
    build_rssm_trainable,
    build_rssm_vision_finetune,
    build_vision_feature_extractor,
    build_world_model,
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

# Module-level logger, defined only after every builder import above so no
# E402 ("module level import not at top of file") fires on them.
_log = get_logger(__name__)

# Hide submodule names from dir() to match pre-split flat module's dir() output.
# Python's import mechanism exposes submodules automatically; we delete them here
# to maintain Invariant #1's pre/post decomposition dir() equivalence.
del arm, autonomous, cloud, cognitive, growth, hardware, health, learning
del llm_gateway, mcp_harness, memory_curiosity, mission, on_device_learning
del orchestrator, safety, telemetry, voice, world_model, _replay_batch_helpers

__all__ = [
    "DEFAULT_CAMERA_HEIGHT",
    "DEFAULT_CAMERA_WIDTH",
    "DEFAULT_LIDAR_BUFFER_SIZE",
    "DEFAULT_LIDAR_MAX_RANGE_M",
    "DEFAULT_MOTOR_BAUDRATE",
    "ENGINE_TYPE_POLICY",
    "ENGINE_TYPE_WORLD_MODEL",
    "Any",
    "AudioProtocol",
    "Awaitable",
    "Callable",
    "CameraProtocol",
    "Coroutine",
    "DistanceSensorProtocol",
    "ESP32CommProtocol",
    "FaceDisplayProtocol",
    "LLMGatewayProtocol",
    "LiDARProtocol",
    "LidarProtocol",
    "Mapping",
    "MetricsRegistryProtocol",
    "MotorControllerProtocol",
    "Path",
    "PromptInjectionFilterProtocol",
    "RegexInjectionFilter",
    "SafetyActionProjectorProtocol",
    "SafetyMonitorProtocol",
    "SpeakerProtocol",
    "VisionProtocol",
    "VoiceEngineProtocol",
    "WatchdogProtocol",
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
]

# Private names re-exported above (visible in dir(), importable directly)
# but deliberately excluded from __all__ — same convention as
# config/schema/__init__.py's _WORLD_MODEL_DEFAULT_REPO_ID: present for a
# real test or internal cross-module call site to import by name, but not
# advertised as part of the wildcard-import / public API surface. See
# ADR-017 for the full list of which private names real tests depend on.
