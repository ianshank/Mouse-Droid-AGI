"""Factory builders — VLA policy, reward model, and replay reader."""

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


def build_vla_policy(
    cfg: Settings,
    *,
    metrics: MetricsRegistry | None = None,
) -> VLAPolicyProtocol | None:
    """Build the VLA policy if configured.

    Returns ``None`` when ``cfg.vla.backend == "none"`` so callers (the
    orchestrator) can skip the VLA branch without inspecting backend
    strings.

    Backends:
        - ``"none"`` (default): VLA path disabled; returns ``None``.
        - ``"mock"`` (Phase 3a): in-tree zero-dependency :class:`MockVLA`.
        - ``"distilled_onnx"`` (Phase 3b): :class:`DistilledVLAOnnx`
          backed by ONNX Runtime. Optionally pulls weights from
          HuggingFace via
          :func:`mousedroid.utils.weights_manager.download_weights_from_huggingface`.
          Requires the ``[vla]`` extra (``onnxruntime`` /
          ``onnxruntime-gpu``); the import is deferred to
          :meth:`DistilledVLAOnnx.warmup`.

    Args:
        cfg: Root settings.
        metrics: Optional :class:`MetricsRegistry` forwarded to the chosen
            backend so ``predict()`` calls populate
            ``mousedroid_vla_inference_seconds``. ``None`` (default)
            preserves byte-identical pre-PR-A2.1 behavior.

    Returns:
        A :class:`VLAPolicyProtocol` instance, or ``None`` when disabled.

    Raises:
        ValueError: When ``vla.canned_action`` length disagrees with
            ``model.action_dim``, or the configured ``distilled_onnx``
            cache directory cannot be located and no ``model_repo_id``
            is configured for download.
    """
    backend = cfg.vla.backend
    if backend == "none":
        _log.info("vla_policy_disabled")
        return None

    import torch as _torch  # runtime import; torch is a project dependency

    action_dim = cfg.model.action_dim
    canned: _torch.Tensor | None = None
    if cfg.vla.canned_action is not None:
        if len(cfg.vla.canned_action) != action_dim:
            msg = (
                f"vla.canned_action length {len(cfg.vla.canned_action)} "
                f"!= model.action_dim ({action_dim})"
            )
            raise ValueError(msg)
        canned = _torch.tensor(cfg.vla.canned_action, dtype=_torch.float32)

    if backend == "mock":
        from mousedroid.vla.policy import MockVLA

        _log.info("vla_policy_built", backend="mock", action_dim=action_dim)
        return MockVLA(
            action_dim=action_dim,
            canned_action=canned,
            confidence=cfg.vla.confidence,
            metrics=metrics,
        )

    if backend == "distilled_onnx":
        return _build_distilled_onnx_vla(cfg, action_dim, metrics=metrics)

    msg = f"Unknown VLA backend {backend!r}"
    raise ValueError(msg)


def _build_distilled_onnx_vla(
    cfg: Settings,
    action_dim: int,
    *,
    metrics: MetricsRegistry | None = None,
) -> VLAPolicyProtocol:
    """Resolve the ONNX model and instantiate :class:`DistilledVLAOnnx`.

    Args:
        cfg: Root settings.
        action_dim: Configured action dimensionality.
        metrics: Optional :class:`MetricsRegistry` forwarded to
            :class:`DistilledVLAOnnx` so each inference call populates
            ``mousedroid_vla_inference_seconds``.

    Returns:
        An un-warmed :class:`DistilledVLAOnnx`. Warmup happens on first
        :meth:`predict` call (or eagerly if ``warmup_iterations > 0``
        and the file exists at construction time).

    Raises:
        ValueError: When neither a downloadable ``model_repo_id`` nor a
            local ``cache_dir`` containing ``model_filename`` is
            available.
    """
    from pathlib import Path as _Path

    from mousedroid.vla.policy import DistilledVLAOnnx

    # ``VLAConfig.cache_dir`` is required (Pydantic-defaulted to
    # ``"weights/vla"``); guard against an explicit ``None`` override that
    # may slip in via legacy YAML.
    if cfg.vla.cache_dir is None:
        msg = "vla.cache_dir must be set (default is 'weights/vla')"
        raise ValueError(msg)
    cache_dir = _Path(cfg.vla.cache_dir)
    model_path = cache_dir / cfg.vla.model_filename

    if not model_path.is_file():
        if cfg.vla.model_repo_id is None:
            msg = (
                f"distilled_onnx model not found at {model_path} and "
                f"vla.model_repo_id is unset; either provide a local "
                f"file or configure a HuggingFace repo for download"
            )
            raise ValueError(msg)
        from mousedroid.utils.weights_manager import (
            download_weights_from_huggingface,
        )

        cache_dir.mkdir(parents=True, exist_ok=True)
        success = download_weights_from_huggingface(
            repo_id=cfg.vla.model_repo_id,
            filenames=[cfg.vla.model_filename],
            cache_dir=cache_dir,
            # Force flat layout (cache_dir/model_filename) so the
            # ``model_path.is_file()`` check below sees the file. Without
            # ``local_dir``, hf_hub_download uses its blob/snapshot cache
            # layout and the file would not be at the expected path.
            local_dir=cache_dir,
        )
        if not success or not model_path.is_file():
            msg = (
                f"failed to download distilled_onnx weights "
                f"({cfg.vla.model_repo_id}/{cfg.vla.model_filename}) "
                f"into {cache_dir}"
            )
            raise ValueError(msg)

    policy = DistilledVLAOnnx(
        model_path=model_path,
        action_dim=action_dim,
        providers=list(cfg.vla.providers) if cfg.vla.providers is not None else None,
        h_input_name=cfg.vla.h_input_name,
        z_input_name=cfg.vla.z_input_name,
        action_output_name=cfg.vla.action_output_name,
        warmup_iterations=cfg.vla.warmup_iterations,
        confidence=cfg.vla.confidence,
        metrics=metrics,
    )
    _log.info(
        "vla_policy_built",
        backend="distilled_onnx",
        action_dim=action_dim,
        model_path=str(model_path),
    )
    return policy


def build_reward_model(
    cfg: Settings,
    *,
    metrics: MetricsRegistry | None = None,
) -> RewardModelProtocol:
    """Build the multi-objective reward model with optional VLM progress head.

    The Three Laws head is constructed inside
    :class:`MultiObjectiveRewardModel` whenever ``cfg.three_laws.enabled``.
    The Phase 4 VLM progress head is attached only when both
    ``cfg.reward.vlm_progress.enabled`` and
    ``cfg.reward.weight_vlm_progress > 0`` so that, by default, behaviour is
    byte-identical to the pre-Phase 4 reward path.

    Args:
        cfg: Root settings.
        metrics: Optional :class:`MetricsRegistry` forwarded to
            :class:`VLMProgressHead` so cache hit/miss decisions populate
            ``mousedroid_vlm_progress_cache_hits_total`` / ``..._misses_total``.
            ``None`` (default) preserves byte-identical pre-PR-A2.1 behavior.

    Returns:
        Configured reward model.
    """
    from mousedroid.reward.model import MultiObjectiveRewardModel
    from mousedroid.reward.vlm_progress import VLMProgressHead

    vlm_head: VLMProgressHead | None = None
    if cfg.reward.vlm_progress.enabled and cfg.reward.weight_vlm_progress > 0.0:
        vlm_head = VLMProgressHead(cfg.reward.vlm_progress, metrics=metrics)

    model = MultiObjectiveRewardModel(
        cfg.model,
        cfg.reward,
        law_cfg=cfg.three_laws,
        vlm_head=vlm_head,
    )
    _log.info(
        "reward_model_built",
        vlm_progress_enabled=vlm_head is not None,
        three_laws_enabled=cfg.three_laws.enabled,
    )
    return model


def build_replay_reader(
    cfg: Settings,
    *,
    metrics: MetricsRegistry | None = None,
) -> ReplayReaderProtocol:
    """Build the Phase 2 LMDB replay reader.

    Args:
        cfg: Root settings. Reads ``cfg.experience`` (LMDB path + map size)
            and respects ``cfg.training.replay.source_path`` as a path
            override when set.
        metrics: Optional :class:`MetricsRegistry` forwarded to the reader
            so each decoded record / schema-mismatch drop populates
            ``mousedroid_replay_records_total{outcome=...}``. ``None``
            (default) preserves byte-identical pre-PR-A2.1 behavior.

    Returns:
        Reader conforming to :class:`ReplayReaderProtocol`. The concrete
        type is hidden behind the protocol so callers cannot couple to
        LMDB internals (CLAUDE.md invariants 1+2).
    """
    from mousedroid.training.replay import LMDBReplayReader

    reader = LMDBReplayReader(
        cfg.experience,
        path_override=cfg.training.replay.source_path,
        debug_log_every_n=cfg.training.replay_mixer.debug_log_every_n,
        metrics=metrics,
    )
    _log.info(
        "replay_reader_built",
        path=str(reader.path),
        debug_log_every_n=cfg.training.replay_mixer.debug_log_every_n,
        metrics_enabled=metrics is not None,
    )
    return reader
