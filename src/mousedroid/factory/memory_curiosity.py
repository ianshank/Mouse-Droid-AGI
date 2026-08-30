"""Factory builders — layered memory tier, experience logger, curiosity module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from mousedroid.logging.setup import get_logger


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


def build_memory_tier(cfg: Settings) -> MemoryTier | None:
    """Build all four memory subsystems if memory is enabled.

    Args:
        cfg: Root settings.

    Returns:
        ``MemoryTier`` dataclass or ``None`` if ``cfg.memory.enabled`` is False.
    """
    if not cfg.memory.enabled:
        return None

    from mousedroid.memory.consolidation import MemoryConsolidation
    from mousedroid.memory.episodic import EpisodicReplay
    from mousedroid.memory.semantic import SemanticIndex
    from mousedroid.memory.tier import MemoryTier
    from mousedroid.memory.working import WorkingMemory

    episodic = EpisodicReplay(cfg.memory, seed=cfg.memory.replay_seed)
    semantic = SemanticIndex(cfg.memory)
    working = WorkingMemory(cfg.memory, embed_dim=cfg.memory.semantic_dim)
    consolidation = MemoryConsolidation(cfg.memory, episodic, semantic)

    _log.info(
        "memory_tier_built",
        episodic_capacity=cfg.memory.episodic_capacity,
        semantic_dim=cfg.memory.semantic_dim,
        working_context=cfg.memory.working_context_size,
    )
    return MemoryTier(
        episodic=episodic,
        semantic=semantic,
        working=working,
        consolidation=consolidation,
    )


def build_experience_logger(cfg: Settings) -> ExperienceLogger | None:
    """Build LMDB experience logger if memory is enabled and experience config is present.

    Args:
        cfg: Root settings.

    Returns:
        ``ExperienceLogger`` or ``None`` if memory is disabled or experience config is absent.
    """
    if not cfg.memory.enabled:
        return None

    experience_cfg = getattr(cfg, "experience", None)
    if experience_cfg is None:
        return None

    from mousedroid.experience.logger import ExperienceLogger

    logger = ExperienceLogger(experience_cfg)
    _log.info("experience_logger_built", path=experience_cfg.path)
    return logger


def build_curiosity_module(cfg: Settings) -> CuriosityProtocol | None:
    """Build intrinsic curiosity module if memory is enabled.

    Args:
        cfg: Root settings.

    Returns:
        ``IntrinsicCuriosityModule`` or ``None`` if memory is disabled.
    """
    if not cfg.memory.enabled:
        return None

    from mousedroid.curiosity.icm import IntrinsicCuriosityModule

    try:
        module = IntrinsicCuriosityModule(cfg.model, cfg.curiosity)
        _log.info("curiosity_module_built", scale=cfg.curiosity.intrinsic_reward_scale)
        return module
    except Exception:
        _log.warning("curiosity_module_build_failed", exc_info=True)
        return None
