"""Factory builders — safety monitor and action projector."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from mousedroid.logging.setup import get_logger
from mousedroid.safety.protocol import SafetyMonitorProtocol
from mousedroid.safety.projector_protocol import SafetyActionProjectorProtocol


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


def build_safety_monitor(cfg: Settings) -> SafetyMonitorProtocol:
    """Build safety monitor for configured platform.

    Args:
        cfg: Root settings.

    Returns:
        Safety monitor conforming to ``SafetyMonitorProtocol``.
    """
    from mousedroid.safety.monitor import MouseDroidSafetyMonitor

    return MouseDroidSafetyMonitor(cfg.safety)


def build_safety_projector(
    cfg: Settings,
    *,
    metrics: MetricsRegistry | None = None,
) -> SafetyActionProjectorProtocol | None:
    """Build the optional geometric safety action projector (Tier C2 / C2.1).

    Returns ``None`` when ``cfg.safety.projector.enabled`` is ``False`` —
    the orchestrator skips the projection seam entirely in that case, so
    pre-C2 deployments produce byte-identical actions.

    Args:
        cfg: Root settings.
        metrics: Optional shared metrics registry. When supplied, the
            projector increments ``mousedroid_safety_action_clamps_total``
            with one of ``forward_velocity`` / ``human_proximity`` /
            ``tight_quarters`` on every materially different clamp.

    Returns:
        :class:`SafetyActionProjectorProtocol` implementation when enabled,
        ``None`` otherwise.
    """
    if not cfg.safety.projector.enabled:
        _log.debug("safety_projector_disabled")
        return None

    from mousedroid.safety.projector import GeometricSafetyProjector

    _log.info("safety_projector_built", backend="geometric")
    return GeometricSafetyProjector(cfg.safety.projector, metrics=metrics)
