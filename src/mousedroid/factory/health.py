"""Factory builders — health monitor and watchdog."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from mousedroid.logging.setup import get_logger
from mousedroid.health.watchdog import WatchdogProtocol


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

_log = get_logger(__name__)


def build_health_monitor(cfg: Settings) -> HealthMonitor:
    """Build health monitor for GPU/thermal monitoring.

    Args:
        cfg: Root settings.

    Returns:
        Configured ``HealthMonitor``.
    """
    from mousedroid.health.monitor import HealthMonitor

    _log.info("health_monitor_built")
    return HealthMonitor(cfg.health, cfg.jetson)


def build_watchdog(cfg: Settings) -> WatchdogProtocol:
    """Build watchdog notifier based on config.

    Returns :class:`SystemdNotifier` when the ``NOTIFY_SOCKET`` env var is
    present (set automatically by systemd for ``Type=notify`` services),
    :class:`FileHeartbeatNotifier` for Docker/custom monitoring, or
    :class:`NullNotifier` when watchdog is disabled.

    Args:
        cfg: Root settings.

    Returns:
        Watchdog notifier satisfying :class:`WatchdogProtocol`.
    """
    import os
    from pathlib import Path

    from mousedroid.health.watchdog import (
        FileHeartbeatNotifier,
        NullNotifier,
        SystemdNotifier,
    )

    if not cfg.loop.watchdog_enabled:
        return NullNotifier()

    mode = cfg.loop.watchdog_mode
    if mode == "none":
        return NullNotifier()
    if mode == "systemd":
        return SystemdNotifier()
    if mode == "file":
        return FileHeartbeatNotifier(Path(cfg.loop.watchdog_heartbeat_path))
    if mode == "auto":
        if os.environ.get("NOTIFY_SOCKET"):
            return SystemdNotifier()
        return FileHeartbeatNotifier(Path(cfg.loop.watchdog_heartbeat_path))

    _log.warning("unknown_watchdog_mode_falling_back_to_null", mode=mode)
    return NullNotifier()
