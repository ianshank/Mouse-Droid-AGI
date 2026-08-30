"""Factory builders — BDI agent and metacognitive core."""

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


def build_agent(cfg: Settings, world_model: WorldModelProtocol) -> AgentProtocol:
    """Build navigation agent for configured platform.

    Args:
        cfg: Root settings.
        world_model: World model for planning.

    Returns:
        Agent conforming to ``AgentProtocol``.
    """
    from mousedroid.agents.navigation import MouseDroidNavigationAgent
    from mousedroid.world_model.mcts import MCTSPlanner

    planner = MCTSPlanner(cfg.mcts, world_model, action_dim=cfg.model.action_dim)
    return MouseDroidNavigationAgent(planner, cfg)


def _resolve_bdi_weights(cfg: Settings) -> tuple[NeuralBDI, str]:
    """Resolve BDI model weights: local, HuggingFace, or random.

    Args:
        cfg: Root settings.

    Returns:
        Tuple of ``(NeuralBDI instance, weights_source_label)``.
    """
    from pathlib import Path

    from mousedroid.cognitive.bdi_model import NeuralBDI
    from mousedroid.utils import (
        download_weights_from_huggingface,
        weights_exist_locally,
    )

    weights_dir = Path(cfg.cognitive.weights_dir)
    bdi_filenames = ["belief.npz", "desire.npz", "intention.npz", "affect.npz"]

    if weights_exist_locally(weights_dir, bdi_filenames):
        _log.info("cognitive_core_loading_local_weights", weights_dir=str(weights_dir))
        return NeuralBDI(weights_dir=weights_dir), "local"

    if cfg.cognitive.auto_download:
        success = download_weights_from_huggingface(
            repo_id=cfg.cognitive.huggingface_repo,
            filenames=bdi_filenames,
            cache_dir=weights_dir,
            subfolder=cfg.cognitive.huggingface_subfolder,
            local_dir=weights_dir.parent,
            max_retries=cfg.cognitive.download_max_retries,
            backoff_base=cfg.cognitive.download_backoff_base,
        )
        if success:
            _log.info(
                "cognitive_core_loaded_from_huggingface",
                repo_id=cfg.cognitive.huggingface_repo,
                weights_dir=str(weights_dir),
            )
            return NeuralBDI(weights_dir=weights_dir), "huggingface"

    _log.warning(
        "weights_not_found_using_random_initialization",
        weights_dir=str(weights_dir),
        auto_download=cfg.cognitive.auto_download,
    )
    return NeuralBDI(), "random"


def build_cognitive_core(cfg: Settings) -> CognitiveCore:
    """Build cognitive core with optional weight loading from HuggingFace.

    Args:
        cfg: Root settings.

    Returns:
        Fully configured ``CognitiveCore``.
    """
    from mousedroid.cognitive.cognitive_core import CognitiveCore
    from mousedroid.cognitive.constitutional_rl import (
        ConstitutionalChecker,
        ConstitutionalRLConfig,
        PolicyMLP,
    )
    from mousedroid.cognitive.metacognitive import MetacognitiveModel

    _log.info(
        "cognitive_core_init_starting",
        weights_dir=str(cfg.cognitive.weights_dir),
        auto_download=cfg.cognitive.auto_download,
    )

    bdi, weights_source = _resolve_bdi_weights(cfg)

    policy = PolicyMLP(
        action_dim=cfg.model.action_dim,
        input_dim=cfg.model.belief_dim,
    )
    core = CognitiveCore(
        bdi=bdi,
        metacog=MetacognitiveModel(
            n_capabilities=cfg.metacognitive.n_capabilities,
            loop_score_scale=cfg.metacognitive.loop_score_scale,
        ),
        checker=ConstitutionalChecker(
            config=ConstitutionalRLConfig(
                speed_ceiling_mps=cfg.safety.max_velocity_mps,
                battery_min_v=cfg.safety.battery_critical_v,
            ),
        ),
        policy=policy,
    )
    _log.info(
        "cognitive_core_initialized",
        weights_source=weights_source,
        belief_dim=cfg.model.belief_dim,
        desire_dim=cfg.model.desire_dim,
        intention_classes=cfg.model.intention_classes,
    )
    return core
