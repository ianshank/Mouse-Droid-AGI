"""Factory builders — RSSM/latent-dynamics engine, vision feature extraction, and the rover simulation environment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from mousedroid.logging.setup import get_logger
from pathlib import Path

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


def build_world_model(cfg: Settings) -> WorldModelProtocol:
    """Build world model for configured platform.

    Dispatch order:

    1. ``cfg.world_model.engine == "onnx_trt"`` — construct
       :class:`~mousedroid.world_model.dual_stream_rssm_onnx.DualStreamRSSMOnnx`
       backed by the exported ``.onnx`` at ``cfg.world_model.onnx_path``.
       The runtime is constructed cheaply (no ORT import at this point)
       and warms up on first ``observe_step()`` call. Requires
       ``cfc_hidden_dim > 0`` because the ONNX export is built from
       :class:`DualStreamRSSM`.
    2. ``cfg.world_model.engine == "torch"`` (default) AND
       ``cfc_hidden_dim > 0`` — construct :class:`DualStreamRSSM`.
    3. Fallback — construct the classic :class:`~mousedroid.world_model.rssm.RSSM`.

    Default behavior (``engine="torch"``) is byte-identical to pre-B2:
    existing ``config/*.yaml`` files that omit the ``world_model:`` block
    load unchanged.

    Args:
        cfg: Root settings.

    Returns:
        World model conforming to ``WorldModelProtocol``.
    """
    engine = cfg.world_model.engine
    if engine == "onnx_trt":
        return _build_onnx_world_model(cfg)
    if engine != "torch":
        # Pydantic Literal["torch", "onnx_trt"] should catch this earlier,
        # but defend in depth so dynamic instantiation doesn't drift past
        # the dispatcher.
        msg = f"Unknown world_model.engine {engine!r}"
        raise ValueError(msg)

    if cfg.model.cfc_hidden_dim > 0:
        try:
            from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM
        except ImportError:
            _log.warning(
                "dual_stream_unavailable_falling_back_to_rssm",
                reason="ncps package not installed (pip install ncps)",
                requested_cfc_dim=cfg.model.cfc_hidden_dim,
            )
        else:
            _log.info(
                "world_model_engine_selected",
                engine="torch",
                gru_dim=cfg.model.hidden_dim,
                cfc_dim=cfg.model.cfc_hidden_dim,
            )
            return DualStreamRSSM(cfg.model)

    from mousedroid.world_model.rssm import RSSM

    return RSSM(cfg.model)


def build_latent_context(cfg: Settings) -> LatentContextProtocol | None:
    """Build the bounded-context latent memory (F-023), or ``None`` when off.

    Returns ``None`` when the ``world_model_memory`` block is absent OR
    ``enabled=False`` — the orchestrator tick path stays byte-identical to
    pre-feature. The memory is engine-agnostic: it operates on the
    orchestrator-carried ``(h, z)`` tensors, so ``h_dim`` is the combined
    ``hidden_dim + cfc_hidden_dim`` (matching the orchestrator's carried
    state for the dual-stream engine).

    Args:
        cfg: Root settings.

    Returns:
        A :class:`LatentContextProtocol` implementation, or ``None``.
    """
    memory_cfg = cfg.world_model_memory
    if memory_cfg is None or not memory_cfg.enabled:
        return None
    from mousedroid.world_model.bounded_context import BoundedContextMemory

    h_dim = cfg.model.hidden_dim + cfg.model.cfc_hidden_dim
    context = BoundedContextMemory(memory_cfg, h_dim=h_dim, z_dim=cfg.model.latent_dim)
    _log.info(
        "latent_context_enabled",
        h_dim=h_dim,
        z_dim=cfg.model.latent_dim,
        recent_size=memory_cfg.recent_size,
        blend_weight=memory_cfg.blend_weight,
    )
    return context


def build_rssm_trainable(cfg: Settings) -> RSSM:
    """Build the concrete trainable RSSM for MuJoCo dynamics pretraining.

    Unlike :func:`build_world_model` (which returns a ``WorldModelProtocol``
    wrapper for deployment), this returns the concrete ``nn.Module`` so the
    pretrainer can call ``train_sequence`` + backprop. Vision is disabled
    (``vision_dim=0`` paired with ``vision_proj_dim=0`` per the schema
    validator) — the sim has no camera; the dynamics core is what gets
    pretrained. Operator pretrain knobs from :class:`TrainingConfig` are copied
    onto the model config so they live in one place (``training:``).

    Args:
        cfg: Root settings.

    Returns:
        A concrete :class:`~mousedroid.world_model.rssm.RSSM` with vision off.
    """
    from mousedroid.world_model.rssm import RSSM

    update: dict[str, object] = {
        "vision_dim": 0,
        "vision_proj_dim": 0,
        "kl_beta": cfg.training.kl_beta,
        "kl_free_nats": cfg.training.rssm_free_nats,
        "kl_balance_alpha": cfg.training.rssm_kl_balance_alpha,
    }
    # Use the rover's full lidar signal when a MuJoCo rover is configured: size the
    # model's lidar modality to the sim's sector count so train_sequence actually
    # reconstructs lidar (otherwise it is silently dropped — leaving only motor +
    # a single min-range scalar). Falls back to the model default when no rover.
    rover = cfg.rover
    if rover is not None and rover.sim.backend == "mujoco":
        update["lidar_dim"] = rover.sim.mujoco.lidar_num_sectors
        update["lidar_proj_dim"] = cfg.model.lidar_proj_dim
    model_cfg = cfg.model.model_copy(update=update)
    return RSSM(model_cfg)


def build_rssm_vision_finetune(cfg: Settings, checkpoint: Path) -> RSSM:
    """Load a vision-OFF pretrained RSSM and migrate it to a vision-ON model.

    Uses :func:`~mousedroid.world_model.checkpoint_migration.load_rssm_with_migration`
    to transfer the dynamics core (gru/posterior/prior/decoder/reward) verbatim,
    copy retained-modality fusion columns, and Kaiming-init the new vision
    columns + ``vision_proj``. Vision dim = ``cfg.camera.feature_dim`` so the
    model matches the sim ``MeanPoolExtractor`` output; lidar mirrors the rover.

    Args:
        cfg: Root settings.
        checkpoint: Path to the vision-OFF pretrained RSSM checkpoint.

    Returns:
        A vision-ON :class:`~mousedroid.world_model.rssm.RSSM` ready to fine-tune.
    """
    import torch

    from mousedroid.world_model.checkpoint_migration import load_rssm_with_migration

    update: dict[str, object] = {
        "vision_dim": cfg.camera.feature_dim,
        # Use the configured projection dim directly; the ModelConfig validator
        # rejects a zero proj_dim paired with a nonzero modality dim, so a
        # misconfig surfaces explicitly instead of being patched to a literal.
        "vision_proj_dim": cfg.model.vision_proj_dim,
        "kl_beta": cfg.training.kl_beta,
        "kl_free_nats": cfg.training.rssm_free_nats,
        "kl_balance_alpha": cfg.training.rssm_kl_balance_alpha,
    }
    rover = cfg.rover
    if rover is not None and rover.sim.backend == "mujoco":
        update["lidar_dim"] = rover.sim.mujoco.lidar_num_sectors
        update["lidar_proj_dim"] = cfg.model.lidar_proj_dim
    model_cfg = cfg.model.model_copy(update=update)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_rssm_with_migration(checkpoint, model_cfg, device)


def build_vision_feature_extractor(cfg: Settings) -> FeatureExtractorProtocol:
    """Build the sim vision feature extractor for RSSM vision-on fine-tuning.

    Returns the same non-learned :class:`MeanPoolExtractor` the deployed
    ``mean_pool`` camera path uses (mean-pool → L2), so rendered-sim and real
    ``vision_features`` share a distribution by construction — no CNN to train.
    Dims come from :class:`CameraConfig` (invariant #3).

    Args:
        cfg: Root settings.

    Returns:
        A ``FeatureExtractorProtocol`` producing ``cfg.camera.feature_dim`` features.
    """
    from mousedroid.hardware.camera.feature_extractor import MeanPoolExtractor

    return MeanPoolExtractor(cfg.camera.feature_dim, l2_normalize=cfg.camera.l2_normalize)


def _build_onnx_world_model(cfg: Settings) -> WorldModelProtocol:
    """Construct the ONNX runtime world model.

    Resolves ``cfg.world_model.onnx_path`` (filesystem-first, HF Hub
    fallback) and hands it to :class:`DualStreamRSSMOnnx`. The runtime
    is constructed lazily — ``onnxruntime`` is not imported here.
    """
    if cfg.model.cfc_hidden_dim <= 0:
        msg = (
            "world_model.engine='onnx_trt' requires model.cfc_hidden_dim > 0 "
            "(the ONNX export is built from DualStreamRSSM, which requires "
            "the CfC stream). Set cfc_hidden_dim in your config YAML or "
            "switch to engine='torch'."
        )
        raise ValueError(msg)

    from mousedroid.world_model.composite import CompositeWorldModel
    from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM
    from mousedroid.world_model.dual_stream_rssm_onnx import DualStreamRSSMOnnx

    model_path = _resolve_world_model_onnx_path(cfg)

    # observe_step path: ONNX-accelerated (the hot 30Hz tick benefit).
    observe_engine = DualStreamRSSMOnnx(
        model_path=model_path,
        cfg=cfg.model,
        warmup_iterations=cfg.world_model.onnx_warmup_iterations,
    )
    # imagine_step + get_safety_trace path: PyTorch DualStreamRSSM. The
    # ONNX export (B2 Story 1) is scoped to observe_step only, so MCTS
    # rollouts and the safety monitor's CfC inspection need the PyTorch
    # graph. Both engines share the same ModelConfig so dimensions stay
    # consistent across the composition boundary.
    imagine_engine = DualStreamRSSM(cfg.model)
    imagine_engine.train(False)

    _log.info(
        "world_model_engine_selected",
        engine="onnx_trt",
        model_path=str(model_path),
        cfc_dim=cfg.model.cfc_hidden_dim,
        composite=True,
        observe_engine=type(observe_engine).__name__,
        imagine_engine=type(imagine_engine).__name__,
    )
    return CompositeWorldModel(
        observe_engine=observe_engine,
        imagine_engine=imagine_engine,
    )


def _resolve_world_model_onnx_path(cfg: Settings) -> Path:
    """Resolve the .onnx artifact path for the ONNX runtime engine.

    Resolution order:

    1. ``cfg.world_model.onnx_path`` when set — use it directly. If the
       file is missing, the runtime's :meth:`warmup` will raise
       ``FileNotFoundError`` so operators get a clear error from the
       runtime, not a confusing ``hf_hub_download`` traceback.
    2. HF Hub download via
       ``cfg.world_model.onnx_repo_id``/``cfg.world_model.onnx_filename``.
       Mirrors the [vla] pattern at ``_build_distilled_onnx_vla``.
       Cached under ``weights/dual_stream_rssm/`` so the same file is
       reused across runs without re-downloading.
    """
    explicit = cfg.world_model.onnx_path
    if explicit is not None:
        return Path(explicit)

    # HF Hub auto-download fallback. Reuses the same
    # ``download_weights_from_huggingface`` helper the VLA path uses, so
    # retries / auth tokens / progress bars work identically.
    from mousedroid.utils.weights_manager import (
        download_weights_from_huggingface,
    )

    # Operator-tunable per deployment via ``cfg.world_model.onnx_cache_dir``
    # (default ``weights/dual_stream_rssm``). Mirrors the VLA pattern at
    # ``_build_distilled_onnx_vla`` so Jetson deployments can repoint both
    # caches under ``/opt/mousedroid/weights/...`` in one place.
    cache_dir = Path(cfg.world_model.onnx_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / cfg.world_model.onnx_filename

    if model_path.is_file():
        _log.info(
            "world_model_onnx_cache_hit",
            cache_path=str(model_path),
            repo_id=cfg.world_model.onnx_repo_id,
        )
        return model_path

    _log.info(
        "world_model_onnx_download_start",
        repo_id=cfg.world_model.onnx_repo_id,
        filename=cfg.world_model.onnx_filename,
        cache_dir=str(cache_dir),
    )
    success = download_weights_from_huggingface(
        repo_id=cfg.world_model.onnx_repo_id,
        filenames=[cfg.world_model.onnx_filename],
        cache_dir=cache_dir,
        # Force flat layout so model_path.is_file() check succeeds.
        # Without local_dir, hf_hub_download uses its blob/snapshot
        # cache layout and the file would not be at the expected path.
        local_dir=cache_dir,
    )
    if not success or not model_path.is_file():
        msg = (
            f"failed to download world-model ONNX artifact "
            f"({cfg.world_model.onnx_repo_id}/{cfg.world_model.onnx_filename}) "
            f"into {cache_dir}. Set world_model.onnx_path to a local path "
            f"or run scripts/export_dual_stream_rssm_onnx.py --push-to-hf "
            f"to publish a fresh artifact first."
        )
        raise FileNotFoundError(msg)
    _log.info(
        "world_model_onnx_downloaded",
        path=str(model_path),
        repo_id=cfg.world_model.onnx_repo_id,
    )
    return model_path


def build_rover_env(cfg: Settings) -> RoverEnvProtocol:
    """Build the rover simulation environment selected by ``cfg.rover.sim.backend``.

    Backends:
        - ``"mock"`` (default): NumPy-only kinematic integrator. Has no
          physics or GPU dependency and is the only backend used in CI.
        - ``"isaac_lab"``: Isaac Lab env stub; requires
          ``pip install -e ".[isaac]"`` on a workstation with NVIDIA
          Isaac Lab prerequisites. Phase B fills in the actual
          articulation / sensor wiring.
        - ``"mujoco"``: reserved for a future PR — raises
          :class:`NotImplementedError`.

    Args:
        cfg: Root settings. ``cfg.rover`` must be populated.

    Returns:
        Environment conforming to :class:`RoverEnvProtocol`.

    Raises:
        ValueError: If ``cfg.rover`` is ``None``.
        NotImplementedError: For backends not yet wired in this phase.
    """
    if cfg.rover is None:
        msg = (
            "rover config required for build_rover_env; set the top-level "
            "'rover:' block in your YAML or pass RoverConfig() directly."
        )
        raise ValueError(msg)

    backend = cfg.rover.sim.backend
    if backend == "mock":
        from mousedroid.sim.mock_rover_env import MockRoverEnv

        _log.info(
            "rover_env_mock_built",
            mode=cfg.rover.action.mode,
            obs_keys=list(cfg.rover.observation.enabled_keys()),
        )
        return MockRoverEnv(
            cfg.rover,
            wheel_radius_m=cfg.robot.wheel_radius_m,
            track_width_m=cfg.robot.track_width_m,
        )

    if backend == "isaac_lab":
        from mousedroid.sim.isaaclab.rover_env import RoverIsaacLabEnv

        env = RoverIsaacLabEnv(
            cfg.rover,
            wheel_radius_m=cfg.robot.wheel_radius_m,
            track_width_m=cfg.robot.track_width_m,
            domain_randomization=cfg.domain_randomization,
        )
        _log.info(
            "rover_env_isaaclab_built",
            num_envs=cfg.rover.sim.num_envs,
            dr_enabled=cfg.domain_randomization.enabled,
        )
        return env

    if backend == "mujoco":
        from mousedroid.sim.mujoco_rover_env import RoverMuJoCoEnv

        mj_env = RoverMuJoCoEnv(
            cfg.rover,
            wheel_radius_m=cfg.robot.wheel_radius_m,
            track_width_m=cfg.robot.track_width_m,
        )
        _log.info(
            "rover_env_mujoco_built",
            lidar_sectors=cfg.rover.sim.mujoco.lidar_num_sectors,
            dr_enabled=cfg.domain_randomization.enabled,
        )
        return mj_env

    msg = f"unknown rover sim backend: {backend!r}"
    raise ValueError(msg)
