"""Factory builders — growth-pillar VLA knowledge distillation coordinator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable
from mousedroid.logging.setup import get_logger

from mousedroid.factory.world_model import build_rssm_trainable, build_world_model
from mousedroid.factory._replay_batch_helpers import (
    _build_shared_replay_reader,
    _count_new_replay_records,
    _make_consumed_offset_advancer,
)


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


def _make_growth_latent_sampler(
    world_model: WorldModelProtocol,
    vla_policy: VLAPolicyProtocol,
    *,
    h_dim: int,
    z_dim: int,
    batch_size: int,
    device: torch.device,
) -> Callable[[], tuple[Tensor, Tensor] | None]:
    """Real on-policy latent sampler: roll the live world model under the teacher.

    Starts from a zero latent and, for ``batch_size`` steps, queries the teacher
    VLA for an action and advances the world model one imagined step, collecting
    each resulting ``(h, z)``. Uses the REAL world model + teacher — no config-
    sized stand-in — so the student is distilled over the teacher's own on-policy
    latent distribution. All work runs under ``torch.no_grad`` (data generation
    only; the student's gradient step happens later in the distiller).
    """
    import torch

    from mousedroid.vla.policy import VLAObservation

    def _sample() -> tuple[Tensor, Tensor] | None:
        h = torch.zeros(1, h_dim, device=device)
        z = torch.zeros(1, z_dim, device=device)
        hs: list[Tensor] = []
        zs: list[Tensor] = []
        with torch.no_grad():
            for _ in range(batch_size):
                obs = VLAObservation(h=h[0], z=z[0])
                action = vla_policy.predict(obs).action.reshape(1, -1).to(device)
                h, z, _reward = world_model.imagine_step(action, h, z)
                hs.append(h.reshape(-1))
                zs.append(z.reshape(-1))
        return torch.stack(hs, dim=0), torch.stack(zs, dim=0)

    return _sample


def build_growth_coordinator(
    cfg: Settings,
    *,
    metrics: MetricsRegistry | None = None,
    vla_policy: VLAPolicyProtocol | None = None,
    world_model: WorldModelProtocol | None = None,
) -> object | None:
    """Build the growth-pillar VLA knowledge-distillation coordinator.

    Returns ``None`` (so the orchestrator stays byte-identical to pre-feature)
    whenever ``cfg.growth`` is absent/disabled OR no VLA teacher policy is wired
    (there is nothing to distil). When enabled, wires a compact
    :class:`~mousedroid.growth.student.StudentVLAPolicy`, a paramless
    :class:`~mousedroid.growth.student.VLATeacherModule` around the live VLA
    policy, a regression-objective
    :class:`~mousedroid.growth.distillation.KnowledgeDistiller`, the SHA-256
    :class:`~mousedroid.growth.slot_store.GrowthSlotStore` resolved under
    ``cfg.experience.path``, and a latent sampler that rolls the live world model
    under the teacher for real on-policy ``(h, z)`` batches. The new-record trigger
    reuses the LMDB replay signal (like on-device learning), so distillation arms
    only as fresh experience accumulates.

    Args:
        cfg: Root settings.
        metrics: Optional shared metrics registry (keyword-only) so the growth
            distillation counter surfaces on ``/metrics``.
        vla_policy: The live VLA teacher policy (the orchestrator's already-built
            ``build_vla_policy(cfg)`` result). ``None`` disables growth — there is
            no teacher to distil from.
        world_model: The live world model (keyword-only) rolled to sample latents.
            ``None`` resolves it via ``build_world_model(cfg)``.

    Returns:
        A ``GrowthDistillationCoordinator`` when enabled, else ``None``.
    """
    growth_cfg = cfg.growth
    if growth_cfg is None or not growth_cfg.enabled:
        return None
    if vla_policy is None:
        _log.info(
            "growth_disabled_no_vla_teacher",
            hint="growth distillation requires an enabled vla.backend teacher",
        )
        return None

    import torch

    from mousedroid.growth.coordinator import GrowthDistillationCoordinator
    from mousedroid.growth.distillation import KnowledgeDistiller
    from mousedroid.growth.slot_store import GrowthSlotStore
    from mousedroid.growth.student import StudentVLAPolicy, VLATeacherModule

    effective_wm = world_model if world_model is not None else build_world_model(cfg)

    h_dim = cfg.model.hidden_dim
    z_dim = cfg.model.latent_dim
    action_dim = cfg.model.action_dim

    # Resolve the model device so the student trains where the teacher + world
    # model live — never a hardcoded CPU. The RSSM variants are all ``nn.Module``s;
    # the protocol does not declare ``parameters()`` so narrow before reading it.
    device = torch.device("cpu")
    if isinstance(effective_wm, torch.nn.Module):
        first_param = next(effective_wm.parameters(), None)
        if first_param is not None:
            device = first_param.device

    student = StudentVLAPolicy(
        h_dim=h_dim,
        z_dim=z_dim,
        hidden_dim=growth_cfg.student_hidden_dim,
        action_dim=action_dim,
    ).to(device)
    teacher = VLATeacherModule(vla_policy, h_dim=h_dim, z_dim=z_dim)
    distiller = KnowledgeDistiller(
        teacher,
        student,
        temperature=growth_cfg.temperature,
        alpha=growth_cfg.alpha,
        lr=growth_cfg.learning_rate,
        objective="regression",
    )
    slot_store = GrowthSlotStore(experience_cfg=cfg.experience, growth_cfg=growth_cfg)

    # Reuse the main replay path's reader so the growth trigger honours any
    # ``source_path`` override and shares the debug-log cadence.
    reader = _build_shared_replay_reader(cfg)
    cap = growth_cfg.trigger_min_new_records
    # In-memory consumed offset (see build_on_device_coordinator): the trigger
    # arms on records BEYOND the last fired baseline so it disarms until fresh
    # experience accumulates instead of re-distilling stale data every cadence.
    consumed_offset = [0]

    def _count_new_records() -> int:
        return _count_new_replay_records(reader, consumed=consumed_offset[0], cap=cap)

    _advance_consumed = _make_consumed_offset_advancer(
        consumed_offset, log_event="growth_consumed_offset_advanced"
    )

    sample_batch = _make_growth_latent_sampler(
        effective_wm,
        vla_policy,
        h_dim=h_dim,
        z_dim=z_dim,
        batch_size=growth_cfg.batch_size,
        device=device,
    )

    return GrowthDistillationCoordinator(
        cfg=growth_cfg,
        distiller=distiller,
        student=student,
        sample_batch=sample_batch,
        slot_store=slot_store,
        count_new_records=_count_new_records,
        on_consumed=_advance_consumed,
        metrics=metrics,
    )
