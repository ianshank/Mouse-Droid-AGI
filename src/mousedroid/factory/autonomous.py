"""Factory builders — the parked AutonomousOrchestrator stack. Zero production callers (see docs/architecture/ADR-016-autonomous-orchestrator-disposition.md)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from mousedroid.logging.setup import get_logger
from mousedroid.constants import (
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_LIDAR_BUFFER_SIZE,
    DEFAULT_LIDAR_MAX_RANGE_M,
    DEFAULT_MOTOR_BAUDRATE,
)
from mousedroid.interfaces.protocols import (
    CameraProtocol,
    LiDARProtocol,
    MetricsRegistryProtocol,
    MotorControllerProtocol,
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


def build_autonomous_metrics_registry(cfg: Settings) -> MetricsRegistryProtocol:
    """Build the Prometheus telemetry registry.

    Args:
        cfg: Master system configuration.

    Returns:
        Instance conforming to MetricsRegistryProtocol.
    """
    from mousedroid.telemetry.metrics_registry import PrometheusMetricsRegistry

    return PrometheusMetricsRegistry(cfg=cfg.telemetry)


def build_motor_controller(
    cfg: Settings,
    resolver: Any = None,
    metrics: MetricsRegistryProtocol | None = None,
) -> MotorControllerProtocol:
    """Build motor controller driver with dynamic USB-C port resolution.

    Args:
        cfg: Master system configuration.
        resolver: Optional USB-C endpoint resolver.
        metrics: Optional metrics registry for command recording.

    Returns:
        Driver conforming to MotorControllerProtocol.
    """
    from mousedroid.hardware.motor_controller import MockMotorController, MotorController

    if cfg.mock_hardware or not getattr(cfg.motor, "enabled", True):
        _log.info("building_mock_motor_controller")
        return MockMotorController(metrics=metrics)

    port = getattr(cfg.motor, "serial_port", "/dev/ttyUSB0")
    if resolver and cfg.usbc_discovery and cfg.usbc_discovery.enabled:
        resolved = getattr(resolver, "resolve_endpoint", lambda name: None)("esp32")
        if resolved:
            port = resolved

    baud = getattr(cfg.motor, "baudrate", DEFAULT_MOTOR_BAUDRATE)
    _log.info("building_real_motor_controller", port=port, baud=baud)
    return MotorController(cfg=cfg.motor, port=port, metrics=metrics)


def build_autonomous_camera(cfg: Settings) -> CameraProtocol:
    """Build CSI/USB camera perception driver for autonomous mission control.

    Args:
        cfg: Master system configuration.

    Returns:
        Driver conforming to CameraProtocol.
    """
    from mousedroid.hardware.camera_csi import CSICamera, MockCamera

    if cfg.mock_hardware:
        _log.info("building_mock_camera")
        width = getattr(cfg.camera, "resolution_width", DEFAULT_CAMERA_WIDTH)
        height = getattr(cfg.camera, "resolution_height", DEFAULT_CAMERA_HEIGHT)
        return MockCamera(width=width, height=height)
    return CSICamera(cfg=cfg.camera)


def build_autonomous_lidar(
    cfg: Settings,
    resolver: Any = None,
) -> LiDARProtocol:
    """Build LiDAR range scanner driver for autonomous navigation.

    Args:
        cfg: Master system configuration.
        resolver: Optional USB-C endpoint resolver.

    Returns:
        Driver conforming to LiDARProtocol.
    """
    from mousedroid.config.schema.hardware import LidarConfig
    from mousedroid.hardware.lidar_driver import LiDARScanner, MockLiDAR

    if cfg.mock_hardware:
        _log.info("building_mock_lidar")
        return MockLiDAR(default_distance=DEFAULT_LIDAR_MAX_RANGE_M)

    port = getattr(cfg.lidar, "device_path", "/dev/ttyUSB1") if cfg.lidar else "/dev/ttyUSB1"
    if resolver and cfg.usbc_discovery and cfg.usbc_discovery.enabled:
        resolved = getattr(resolver, "resolve_endpoint", lambda name: None)("lidar")
        if resolved:
            port = resolved

    return LiDARScanner(
        cfg=cfg.lidar if cfg.lidar else LidarConfig(),
        port=port,
        buffer_size=DEFAULT_LIDAR_BUFFER_SIZE,
    )


def build_autonomous_orchestrator(
    cfg: Settings,
    resolver: Any = None,
    metrics: MetricsRegistryProtocol | None = None,
) -> Any:
    """Build master autonomous orchestrator.

    Args:
        cfg: Master system configuration.
        resolver: Optional USB-C endpoint resolver.
        metrics: Optional metrics registry.

    Returns:
        Configured AutonomousOrchestrator instance.
    """
    from mousedroid.llm_gateway.composite_gateway import CompositeLLMGateway
    from mousedroid.orchestrator.autonomous import AutonomousOrchestrator

    metrics_reg = metrics or build_autonomous_metrics_registry(cfg)
    motor = build_motor_controller(cfg, resolver=resolver, metrics=metrics_reg)
    cam = build_autonomous_camera(cfg)
    lidar = build_autonomous_lidar(cfg, resolver=resolver)
    llm = CompositeLLMGateway(cfg=cfg.llm, mock_mode=cfg.mock_hardware, metrics=metrics_reg)

    return AutonomousOrchestrator(
        cfg=cfg,
        motor=motor,
        camera=cam,
        lidar=lidar,
        llm=llm,
        metrics=metrics_reg,
    )
