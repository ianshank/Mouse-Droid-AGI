"""Factory builders — GCP telemetry/experience/logging/metrics/Firestore sinks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from mousedroid.logging.setup import get_logger
from mousedroid.common.imports import module_available
from mousedroid.cloud.protocol import (
    ENGINE_TYPE_POLICY,
    ENGINE_TYPE_WORLD_MODEL,
    PendingWeightUpdate,
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


def build_cloud_telemetry_sink(
    cfg: Settings,
    *,
    metrics_registry: MetricsRegistry | None = None,
) -> CloudTelemetrySinkProtocol | None:
    """Build GCP Pub/Sub telemetry sink if GCP is configured.

    Returns ``None`` when ``cfg.gcp`` is ``None`` (offline mode) or when
    the ``google-cloud-pubsub`` package is not installed.

    Args:
        cfg: Root settings.
        metrics_registry: Optional metrics registry. When provided the
            sink forwards publish outcomes, publish latency, and circuit
            breaker state transitions to the registry.

    Returns:
        Cloud telemetry sink or None.
    """
    if cfg.gcp is None:
        return None
    if not cfg.gcp.pubsub.enabled:
        # Explicit log: a silent None here is indistinguishable from "gcp not
        # configured at all", which is the first thing an operator checks when
        # telemetry stops arriving.
        _log.info("cloud_telemetry_sink_disabled", reason="gcp.pubsub.enabled=false")
        return None
    if not module_available("google.cloud.pubsub_v1"):
        # google-cloud-pubsub is only in the optional [gcp] extra; the SDK
        # import itself is deferred into CloudTelemetrySink.start(), so this
        # spec-only probe is what actually detects a missing install (the
        # except ImportError below never fires for that case on its own).
        _log.warning(
            "cloud_pubsub_not_available",
            hint="Install via: pip install 'mousedroid[gcp]'",
        )
        return None
    try:
        from mousedroid.cloud.pubsub_sink import CloudTelemetrySink
    except ImportError:
        _log.warning(
            "cloud_pubsub_not_available",
            hint="Install via: pip install 'mousedroid[gcp]'",
        )
        return None

    sink = CloudTelemetrySink(cfg.gcp, metrics=metrics_registry)
    _log.info("cloud_telemetry_sink_built", metrics_wired=metrics_registry is not None)
    return sink


def build_cloud_experience_exporter(
    cfg: Settings,
) -> CloudExperienceExporterProtocol | None:
    """Build GCS experience exporter if GCP is configured.

    Returns ``None`` when ``cfg.gcp`` is ``None`` or when the
    ``google-cloud-storage`` package is not installed.

    Args:
        cfg: Root settings.

    Returns:
        Cloud experience exporter or None.
    """
    if cfg.gcp is None:
        return None
    if not cfg.gcp.storage.enabled:
        _log.info("cloud_experience_exporter_disabled", reason="gcp.storage.enabled=false")
        return None
    if not module_available("google.cloud.storage"):
        # google-cloud-storage is only in the optional [gcp] extra; the SDK
        # import itself is deferred into CloudExperienceExporter.start(), so
        # this spec-only probe is what actually detects a missing install.
        _log.warning(
            "cloud_storage_not_available",
            hint="Install via: pip install 'mousedroid[gcp]'",
        )
        return None
    try:
        from mousedroid.cloud.experience_exporter import CloudExperienceExporter
    except ImportError:
        _log.warning(
            "cloud_storage_not_available",
            hint="Install via: pip install 'mousedroid[gcp]'",
        )
        return None

    exporter = CloudExperienceExporter(cfg.gcp, cfg.experience)
    _log.info("cloud_experience_exporter_built")
    return exporter


def build_cloud_logging_sink(cfg: Settings) -> CloudLoggingSinkProtocol | None:
    """Build the Cloud Logging structlog processor if GCP is configured.

    Returns ``None`` when ``cfg.gcp`` is ``None`` (offline mode), when
    ``cfg.gcp.logging.enabled`` is ``False``, or when the
    ``google-cloud-logging`` package is not installed.

    Unlike the other cloud builders, this instance's lifecycle is NOT owned
    by the orchestrator — ``configure_logging()`` runs synchronously in
    ``main.py::cli_entry()`` before ``build_orchestrator()`` is ever called,
    so ``main.py`` threads this instance directly into ``configure_logging()``
    and its own ``_run``/``_health_check`` functions.

    Args:
        cfg: Root settings.

    Returns:
        Cloud logging sink or None.
    """
    if cfg.gcp is None:
        return None
    if not cfg.gcp.logging.enabled:
        _log.info("cloud_logging_sink_disabled", reason="gcp.logging.enabled=false")
        return None
    if not module_available("google.cloud.logging"):
        # google-cloud-logging is only in the optional [gcp] extra; the SDK
        # import itself is deferred into CloudLoggingSink.start(), so this
        # spec-only probe is what actually detects a missing install.
        _log.warning(
            "cloud_logging_not_available",
            hint="Install via: pip install 'mousedroid[gcp]'",
        )
        return None
    try:
        from mousedroid.cloud.logging_sink import CloudLoggingSink
    except ImportError:
        _log.warning(
            "cloud_logging_not_available",
            hint="Install via: pip install 'mousedroid[gcp]'",
        )
        return None

    sink = CloudLoggingSink(cfg.gcp)
    _log.info("cloud_logging_sink_built")
    return sink


def build_cloud_metrics_exporter(
    cfg: Settings,
    *,
    metrics_registry: MetricsRegistry | None = None,
) -> CloudMetricsExporterProtocol | None:
    """Build the Cloud Monitoring metrics exporter if GCP is configured.

    Returns ``None`` when ``cfg.gcp`` is ``None``, when
    ``cfg.gcp.monitoring.enabled`` is ``False``, when ``metrics_registry`` is
    ``None`` (``build_metrics_registry`` returns ``None`` whenever
    ``cfg.metrics.enabled`` is ``False``, independent of the GCP toggle), or
    when the ``google-cloud-monitoring`` package is not installed.

    Args:
        cfg: Root settings.
        metrics_registry: Metrics registry to export from. Required by
            ``CloudMetricsExporter`` — a ``None`` registry disables this
            builder rather than crashing the concrete class's constructor.

    Returns:
        Cloud metrics exporter or None.
    """
    if cfg.gcp is None:
        return None
    if not cfg.gcp.monitoring.enabled:
        _log.info("cloud_metrics_exporter_disabled", reason="gcp.monitoring.enabled=false")
        return None
    if metrics_registry is None:
        _log.info("cloud_metrics_exporter_disabled", reason="metrics_registry_not_available")
        return None
    if not module_available("google.cloud.monitoring_v3"):
        # google-cloud-monitoring is only in the optional [gcp] extra; the SDK
        # import itself is deferred into CloudMetricsExporter.start(), so this
        # spec-only probe is what actually detects a missing install.
        _log.warning(
            "cloud_monitoring_not_available",
            hint="Install via: pip install 'mousedroid[gcp]'",
        )
        return None
    try:
        from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter
    except ImportError:
        _log.warning(
            "cloud_monitoring_not_available",
            hint="Install via: pip install 'mousedroid[gcp]'",
        )
        return None

    exporter = CloudMetricsExporter(cfg.gcp, metrics_registry)
    _log.info("cloud_metrics_exporter_built")
    return exporter


def build_cloud_firestore_sync(
    cfg: Settings,
    *,
    episodic: EpisodicReplay | None = None,
) -> CloudFirestoreSyncProtocol | None:
    """Build the Cloud Firestore episodic-memory sync if GCP is configured.

    Returns ``None`` when ``cfg.gcp`` is ``None``, when
    ``cfg.gcp.firestore.enabled`` is ``False``, when ``episodic`` is ``None``
    (``build_memory_tier`` returns ``None`` whenever ``cfg.memory.enabled`` is
    ``False`` — the default — independent of the GCP toggle), or when the
    ``google-cloud-firestore`` package is not installed.

    Args:
        cfg: Root settings.
        episodic: Episodic replay buffer to sync from. Required by
            ``CloudFirestoreSync`` — a ``None`` buffer disables this builder
            rather than crashing the concrete class's constructor.

    Returns:
        Cloud Firestore sync or None.
    """
    if cfg.gcp is None:
        return None
    if not cfg.gcp.firestore.enabled:
        _log.info("cloud_firestore_sync_disabled", reason="gcp.firestore.enabled=false")
        return None
    if episodic is None:
        _log.info("cloud_firestore_sync_disabled", reason="episodic_memory_not_available")
        return None
    if not module_available("google.cloud.firestore"):
        # google-cloud-firestore is only in the optional [gcp] extra; the SDK
        # import itself is deferred into CloudFirestoreSync.start(), so this
        # spec-only probe is what actually detects a missing install.
        _log.warning(
            "cloud_firestore_not_available",
            hint="Install via: pip install 'mousedroid[gcp]'",
        )
        return None
    try:
        from mousedroid.cloud.firestore_sync import CloudFirestoreSync
    except ImportError:
        _log.warning(
            "cloud_firestore_not_available",
            hint="Install via: pip install 'mousedroid[gcp]'",
        )
        return None

    sync = CloudFirestoreSync(cfg.gcp, episodic)
    _log.info("cloud_firestore_sync_built")
    return sync
