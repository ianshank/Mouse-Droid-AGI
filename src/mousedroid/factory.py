"""Platform factory functions — build all components via dependency injection.

Factory functions eliminate platform branching. Each ``build_*()`` function
returns the correct implementation based on ``Settings``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.agents.base import AgentProtocol
from mousedroid.comms.protocol import ESP32CommProtocol
from mousedroid.efficiency.tensorrt import TensorRTCompilerProtocol
from mousedroid.hardware.protocols import AudioProtocol, DistanceSensorProtocol, VisionProtocol
from mousedroid.llm_gateway.protocol import LLMGatewayProtocol
from mousedroid.logging.setup import get_logger
from mousedroid.safety.protocol import SafetyMonitorProtocol
from mousedroid.telemetry.log_buffer import LogRingBuffer
from mousedroid.world_model.protocol import WorldModelProtocol

if TYPE_CHECKING:
    from mousedroid.arm.protocols import (
        ArmControllerProtocol,
        ArmDriverProtocol,
        ArmEnvironmentProtocol,
        ArmPerceptionProtocol,
        ArmPlannerProtocol,
    )
    from mousedroid.cognitive.bdi_model import NeuralBDI
    from mousedroid.cognitive.cognitive_core import CognitiveCore
    from mousedroid.config.schema import Settings, UltrasonicConfig
    from mousedroid.hardware.accelerator.hailo_runtime import HailoRuntimeProtocol
    from mousedroid.health.monitor import HealthMonitor
    from mousedroid.sensing.manager import SensorManager
    from mousedroid.telemetry.protocol import TelemetryPublisherProtocol, TelemetryServerProtocol

_log = get_logger(__name__)


def build_esp32_driver(cfg: Settings) -> ESP32CommProtocol:
    """Build ESP32 communication driver based on config.

    Wraps the underlying driver with circuit breaker + retry for
    fault tolerance.  The wrapper implements ``ESP32CommProtocol``
    so the orchestrator doesn't need to know about it.

    Args:
        cfg: Root settings.

    Returns:
        ESP32 driver conforming to ``ESP32CommProtocol``.
    """
    inner: ESP32CommProtocol

    if cfg.mock_hardware:
        from mousedroid.comms.mock_driver import MockESP32Driver

        inner = MockESP32Driver(cfg.esp32)
    elif cfg.esp32.protocol == "serial":
        from mousedroid.comms.serial_driver import SerialESP32Driver

        inner = SerialESP32Driver(cfg.esp32)
    else:
        from mousedroid.comms.wifi_driver import WiFiESP32Driver

        inner = WiFiESP32Driver(cfg.esp32)

    from mousedroid.resilience.resilient_driver import ResilientESP32Driver

    return ResilientESP32Driver(inner, cfg.retry, cfg.circuit_breaker)


def build_camera(
    cfg: Settings,
    hailo_runtime: HailoRuntimeProtocol | None = None,
) -> VisionProtocol:
    """Build camera driver based on config.

    When a Hailo-8 runtime is provided and the camera feature extractor
    is set to ``"hailo"`` or ``"auto"``, the camera's internal feature
    extractor is replaced with a :class:`HailoFeatureExtractor` that
    offloads inference to the accelerator.

    Args:
        cfg: Root settings.
        hailo_runtime: Optional Hailo-8 runtime for accelerated feature extraction.

    Returns:
        Camera driver conforming to ``VisionProtocol``.
    """
    from mousedroid.hardware.camera.feature_extractor import build_feature_extractor

    hailo_extractor = None
    if hailo_runtime is not None and cfg.camera.feature_extractor in ("hailo", "auto"):
        hailo_extractor = build_feature_extractor(cfg.camera, hailo_runtime=hailo_runtime)

    if cfg.mock_hardware:
        from mousedroid.hardware.camera.mock_camera import MockCamera

        camera: VisionProtocol = MockCamera(cfg.camera)
        if hailo_extractor is not None:
            camera._extractor = hailo_extractor  # type: ignore[attr-defined]
        return camera

    if cfg.camera.backend == "jetson_csi":
        from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

        camera = JetsonCSICamera(cfg.camera)
        if hailo_extractor is not None:
            camera._extractor = hailo_extractor  # type: ignore[attr-defined]
        return camera

    if cfg.camera.backend == "picamera2":
        from mousedroid.hardware.camera.imx500 import IMX500Camera

        camera = IMX500Camera(cfg.camera)
        if hailo_extractor is not None:
            camera._extractor = hailo_extractor  # type: ignore[attr-defined]
        return camera

    # auto: try picamera2 first, fall back to jetson_csi
    try:
        from picamera2 import Picamera2  # noqa: F401

        from mousedroid.hardware.camera.imx500 import IMX500Camera

        camera = IMX500Camera(cfg.camera)
    except ImportError:
        from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

        camera = JetsonCSICamera(cfg.camera)

    if hailo_extractor is not None:
        camera._extractor = hailo_extractor  # type: ignore[attr-defined]
    return camera


def build_distance_sensor(cfg: Settings) -> DistanceSensorProtocol:
    """Build distance sensor driver based on config.

    Args:
        cfg: Root settings.

    Returns:
        Distance sensor conforming to ``DistanceSensorProtocol``.
    """
    if cfg.mock_hardware:
        from mousedroid.config.schema import UltrasonicConfig as UltraCfg
        from mousedroid.hardware.sensors.mock_ultrasonic import MockUltrasonic

        ultrasonic_cfg: UltrasonicConfig = cfg.ultrasonic or UltraCfg(  # type: ignore[call-arg]
            trigger_pin=0,
            echo_pin=0,
        )
        return MockUltrasonic(ultrasonic_cfg)

    if cfg.ultrasonic is None:
        msg = "ultrasonic config required for real hardware"
        raise ValueError(msg)

    from mousedroid.hardware.sensors.ultrasonic import HcSr04

    return HcSr04(cfg.ultrasonic)


def build_microphone(cfg: Settings) -> AudioProtocol | None:
    """Build USB microphone driver based on config.

    Args:
        cfg: Root settings.

    Returns:
        Microphone driver conforming to ``AudioProtocol``, or None if disabled.
    """
    if cfg.microphone is None or not cfg.microphone.enabled:
        return None

    if cfg.mock_hardware:
        from mousedroid.hardware.audio.mock_microphone import MockMicrophone

        return MockMicrophone(cfg.microphone)

    from mousedroid.hardware.audio.usb_microphone import UsbMicrophone

    return UsbMicrophone(cfg.microphone)


def build_world_model(cfg: Settings) -> WorldModelProtocol:
    """Build world model for configured platform.

    Args:
        cfg: Root settings.

    Returns:
        World model conforming to ``WorldModelProtocol``.
    """
    from mousedroid.world_model.rssm import RSSM

    return RSSM(cfg.model)


def build_llm_gateway(cfg: Settings) -> LLMGatewayProtocol:
    """Build LLM gateway for NL command translation.

    Constructs a ``GatewayConfig`` from the root settings' ``llm`` section
    and returns an ``LLMGateway`` conforming to ``LLMGatewayProtocol``.

    Args:
        cfg: Root settings.

    Returns:
        LLM gateway conforming to ``LLMGatewayProtocol``.
    """
    from mousedroid.llm_gateway.config import GatewayConfig
    from mousedroid.llm_gateway.gateway import LLMGateway

    gateway_cfg = GatewayConfig(  # type: ignore[call-arg]
        enabled=cfg.llm.enabled,
        model_path=cfg.llm.model_path,
        model_url=cfg.llm.model_url,
        model_checksum=cfg.llm.model_checksum,
        context_length=cfg.llm.context_length,
        n_threads=cfg.llm.n_threads,
        n_gpu_layers=cfg.llm.n_gpu_layers,
        max_tokens=cfg.llm.max_tokens,
        temperature=cfg.llm.temperature,
        latency_target_ms=cfg.llm.latency_target_ms,
        stop_tokens=cfg.llm.stop_tokens,
        max_command_len=cfg.llm.max_command_len,
    )
    _log.info("llm_gateway_built", enabled=cfg.llm.enabled)
    return LLMGateway(gateway_cfg)


def build_safety_monitor(cfg: Settings) -> SafetyMonitorProtocol:
    """Build safety monitor for configured platform.

    Args:
        cfg: Root settings.

    Returns:
        Safety monitor conforming to ``SafetyMonitorProtocol``.
    """
    from mousedroid.safety.monitor import MouseDroidSafetyMonitor

    return MouseDroidSafetyMonitor(cfg.safety)


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
    from mousedroid.cognitive.constitutional_rl import ConstitutionalChecker, PolicyMLP
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
        metacog=MetacognitiveModel(),
        checker=ConstitutionalChecker(),
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


def build_telemetry_publisher(cfg: Settings) -> TelemetryPublisherProtocol | None:
    """Build telemetry publisher if telemetry is enabled.

    Args:
        cfg: Root settings.

    Returns:
        ``TelemetryPublisher`` or ``None`` if telemetry disabled.
    """
    if not cfg.telemetry.enabled:
        return None
    from mousedroid.telemetry.publisher import TelemetryPublisher

    _log.info("telemetry_publisher_built", publish_hz=cfg.telemetry.publish_hz)
    return TelemetryPublisher(cfg.telemetry)


def build_telemetry_server(
    cfg: Settings,
    publisher: TelemetryPublisherProtocol | None,
    health_monitor: HealthMonitor,
    log_buffer: LogRingBuffer | None = None,
) -> TelemetryServerProtocol | None:
    """Build telemetry server if telemetry is enabled.

    Args:
        cfg: Root settings.
        publisher: Telemetry publisher to consume frames from.
        health_monitor: Health monitor for health endpoint.
        log_buffer: Optional log ring buffer for log streaming.

    Returns:
        ``TelemetryServer`` or ``None`` if telemetry disabled.
    """
    if not cfg.telemetry.enabled or publisher is None:
        return None

    if cfg.mock_hardware:
        from mousedroid.telemetry.mock_server import MockTelemetryServer

        _log.info("telemetry_mock_server_built")
        return MockTelemetryServer()

    metrics_registry = None
    metrics_path = cfg.metrics.path
    telemetry_metrics_path_default = type(cfg.telemetry).model_fields["metrics_path"].default
    metrics_path_default = type(cfg.metrics).model_fields["path"].default
    if (
        metrics_path == metrics_path_default
        and cfg.telemetry.metrics_path != telemetry_metrics_path_default
    ):
        metrics_path = cfg.telemetry.metrics_path

    if cfg.metrics.enabled:
        from mousedroid.telemetry.metrics import MetricsRegistry

        metrics_registry = MetricsRegistry(cfg.metrics)

    from mousedroid.telemetry.server import TelemetryServer

    _log.info(
        "telemetry_server_built",
        host=cfg.telemetry.host,
        port=cfg.telemetry.port,
    )
    return TelemetryServer(
        cfg=cfg.telemetry,
        telemetry_queue=publisher.get_queue(),
        health_monitor=health_monitor,
        log_buffer=log_buffer,
        metrics_registry=metrics_registry,
        metrics_path=metrics_path,
        publisher=publisher,
    )


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


def build_sensor_manager(
    cfg: Settings,
    vision: VisionProtocol,
    distance: DistanceSensorProtocol,
    esp32: ESP32CommProtocol,
    microphone: AudioProtocol | None = None,
) -> SensorManager:
    """Build sensor manager for aggregated sensor reads.

    Args:
        cfg: Root settings.
        vision: Camera/vision protocol.
        distance: Distance sensor protocol.
        esp32: ESP32 communication protocol.
        microphone: Optional audio protocol.

    Returns:
        Configured ``SensorManager``.
    """
    from mousedroid.hardware.audio.feature_extractor import AudioFeatureExtractor
    from mousedroid.sensing.manager import SensorManager

    audio_extractor = build_audio_feature_extractor(cfg)
    typed_extractor: AudioFeatureExtractor | None = (
        audio_extractor if isinstance(audio_extractor, AudioFeatureExtractor) else None
    )

    _log.info(
        "sensor_manager_built",
        audio_features_enabled=typed_extractor is not None,
    )
    return SensorManager(
        vision=vision,
        distance=distance,
        esp32=esp32,
        cfg=cfg,
        microphone=microphone,
        audio_feature_extractor=typed_extractor,
    )


def build_audio_feature_extractor(cfg: Settings) -> object | None:
    """Build audio feature extractor if microphone is configured.

    Args:
        cfg: Root settings.

    Returns:
        ``AudioFeatureExtractor`` or ``None`` if microphone is disabled.
    """
    if cfg.microphone is None or not cfg.microphone.enabled:
        return None

    from mousedroid.hardware.audio.feature_extractor import AudioFeatureExtractor

    extractor = AudioFeatureExtractor(cfg.microphone)
    _log.info("audio_feature_extractor_built", feature_dim=extractor.feature_dim)
    return extractor


def build_tensorrt_compiler(cfg: Settings) -> TensorRTCompilerProtocol:
    """Build TensorRT compiler based on config and hardware availability.

    Returns a real ``JetsonTensorRTCompiler`` when TensorRT is enabled and
    the ``torch2trt`` package is available. Falls back to
    ``MockTensorRTCompiler`` otherwise.

    Args:
        cfg: Root settings.

    Returns:
        Compiler conforming to ``TensorRTCompilerProtocol``.
    """
    if cfg.jetson.tensorrt_enabled:
        from mousedroid.efficiency.tensorrt import JetsonTensorRTCompiler

        _log.info(
            "tensorrt_compiler_built",
            precision=cfg.jetson.precision,
            cache_dir=str(cfg.jetson.tensorrt_cache_dir),
        )
        return JetsonTensorRTCompiler(cfg.jetson)

    from mousedroid.efficiency.tensorrt import MockTensorRTCompiler

    _log.info("tensorrt_compiler_mock_built")
    return MockTensorRTCompiler()


def build_hailo_runtime(cfg: Settings) -> HailoRuntimeProtocol | None:
    """Build Hailo-8 accelerator runtime if configured and available.

    Returns ``None`` when Hailo is disabled, the ``hailo_platform``
    package is missing, or the device cannot be found.  This ensures
    graceful degradation — the rest of the pipeline falls back to
    GPU-based inference automatically.

    Args:
        cfg: Root settings.

    Returns:
        Hailo runtime or ``None`` if unavailable.
    """
    if cfg.hailo is None or not cfg.hailo.enabled:
        return None

    if cfg.mock_hardware:
        from mousedroid.hardware.accelerator.hailo_runtime import MockHailoRuntime

        _log.info("hailo_runtime_mock_built")
        return MockHailoRuntime(cfg.hailo)

    try:
        from mousedroid.hardware.accelerator.hailo_runtime import HailoRuntime

        runtime = HailoRuntime(cfg.hailo)
        _log.info("hailo_runtime_built", device_path=cfg.hailo.device_path)
        return runtime
    except Exception:
        _log.warning("hailo_runtime_build_failed_falling_back_to_gpu", exc_info=True)
        return None


def build_orchestrator(cfg: Settings) -> object:
    """Build fully-wired orchestrator.

    Args:
        cfg: Root settings.

    Returns:
        Fully configured ``MouseDroidOrchestrator``.
    """
    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    # Build Hailo-8 runtime early — shared by camera and arm perception
    hailo_runtime = build_hailo_runtime(cfg)

    wm = build_world_model(cfg)
    agent = build_agent(cfg, wm)
    monitor = build_safety_monitor(cfg)
    esp32 = build_esp32_driver(cfg)
    camera = build_camera(cfg, hailo_runtime=hailo_runtime)
    distance = build_distance_sensor(cfg)
    microphone = build_microphone(cfg)

    sensor_manager = build_sensor_manager(
        cfg,
        vision=camera,
        distance=distance,
        esp32=esp32,
        microphone=microphone,
    )

    cognitive_core: CognitiveCore | None = None
    if cfg.cognitive.enabled:
        try:
            cognitive_core = build_cognitive_core(cfg)
        except Exception as e:  # pylint: disable=broad-except
            if cfg.cognitive.fallback_to_mcts:
                _log.warning(
                    "cognitive_core_init_failed_falling_back_to_mcts",
                    error=str(e),
                )
            else:
                raise

    # Telemetry (optional — disabled by default)
    telemetry_publisher = build_telemetry_publisher(cfg)
    health_monitor = build_health_monitor(cfg)

    # Optional log ring buffer for telemetry log streaming
    log_buffer: LogRingBuffer | None = None
    telemetry_cfg = getattr(cfg, "telemetry", None)
    if telemetry_cfg is not None:
        buffer_size = getattr(telemetry_cfg, "log_stream_buffer", 0)
        if buffer_size:
            log_buffer = LogRingBuffer(buffer_size)

    telemetry_server = build_telemetry_server(
        cfg,
        telemetry_publisher,
        health_monitor,
        log_buffer=log_buffer,
    )

    return MouseDroidOrchestrator(
        world_model=wm,
        agents=[agent],
        safety_monitor=monitor,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cognitive_core=cognitive_core,
        cfg=cfg,
        telemetry_publisher=telemetry_publisher,
        telemetry_server=telemetry_server,
    )


# ---------------------------------------------------------------------------
# Robot Arm platform factory functions
# ---------------------------------------------------------------------------


def build_arm_driver(cfg: Settings) -> ArmDriverProtocol:
    """Build robot arm hardware driver based on config.

    Args:
        cfg: Root settings (must have arm config populated).

    Returns:
        Arm driver conforming to ``ArmDriverProtocol``.

    Raises:
        ValueError: If arm config is not populated.
    """
    if cfg.arm is None:
        msg = "arm config required for robot arm platform"
        raise ValueError(msg)

    if cfg.mock_hardware:
        from mousedroid.arm.hardware.mock_arm_driver import MockArmDriver

        _log.info("arm_driver_mock_built")
        return MockArmDriver(cfg.arm)

    from mousedroid.arm.hardware.so_arm100_driver import SoArm100Driver

    _log.info("arm_driver_real_built", port=cfg.arm.serial_port)
    return SoArm100Driver(cfg.arm)


def build_arm_planner(cfg: Settings) -> ArmPlannerProtocol:
    """Build symbolic planner for arm manipulation tasks.

    Args:
        cfg: Root settings (must have arm_planning and arm_task populated).

    Returns:
        Planner conforming to ``ArmPlannerProtocol``.

    Raises:
        ValueError: If arm planning or task config is not populated.
    """
    if cfg.arm_planning is None or cfg.arm_task is None:
        msg = "arm_planning and arm_task configs required for arm planner"
        raise ValueError(msg)

    from mousedroid.arm.planning.symbolic_planner import SymbolicPlanner

    _log.info("arm_planner_built", backend=cfg.arm_planning.planner_backend)
    return SymbolicPlanner(cfg.arm_planning, cfg.arm_task)


def build_arm_environment(cfg: Settings) -> ArmEnvironmentProtocol:
    """Build Gymnasium environment for arm training.

    Args:
        cfg: Root settings (must have arm_task and arm_training populated).

    Returns:
        Environment conforming to ``ArmEnvironmentProtocol``.

    Raises:
        ValueError: If arm task or training config is not populated.
    """
    if cfg.arm_task is None or cfg.arm_training is None:
        msg = "arm_task and arm_training configs required for arm environment"
        raise ValueError(msg)

    dof = cfg.arm.dof if cfg.arm is not None else 6

    if cfg.arm_task.task_type == "laundry_sorting":
        from mousedroid.arm.environments.laundry_sorting import LaundrySortingEnv

        _log.info("arm_env_laundry_built")
        return LaundrySortingEnv(cfg.arm_task, cfg.arm_training, dof=dof)

    from mousedroid.arm.environments.tower_of_hanoi import TowerOfHanoiEnv

    _log.info("arm_env_hanoi_built", num_disks=cfg.arm_task.num_disks)
    return TowerOfHanoiEnv(cfg.arm_task, cfg.arm_training, dof=dof)


def build_arm_controller(cfg: Settings) -> ArmControllerProtocol:
    """Build RL controller for arm manipulation.

    Composes a ``SACAgent`` with ``ActionPrimitives`` inside an
    ``ArmController`` that implements ``ArmControllerProtocol``.

    Args:
        cfg: Root settings (must have arm and arm_training populated).

    Returns:
        Controller conforming to ``ArmControllerProtocol``.

    Raises:
        ValueError: If arm or arm_training config is not populated.
    """
    if cfg.arm_training is None or cfg.arm is None:
        msg = "arm and arm_training configs required for arm controller"
        raise ValueError(msg)

    from mousedroid.arm.control.controller import ArmController
    from mousedroid.arm.control.primitives import ActionPrimitives
    from mousedroid.arm.control.sac_agent import SACAgent

    driver = build_arm_driver(cfg)
    agent = SACAgent(cfg.arm_training)
    primitives = ActionPrimitives(cfg.arm, driver)
    _log.info("arm_controller_built", algorithm=cfg.arm_training.algorithm)
    return ArmController(agent, primitives)


def build_arm_perception(
    cfg: Settings,
    hailo_runtime: HailoRuntimeProtocol | None = None,
) -> ArmPerceptionProtocol:
    """Build perception pipeline for arm manipulation.

    When a Hailo-8 runtime is provided and ``yolo_backend`` is
    ``"hailo"`` or ``"auto"``, YOLO detection is offloaded to the
    accelerator via :class:`HailoYOLODetector`.

    Args:
        cfg: Root settings (must have arm_perception and arm_task populated).
        hailo_runtime: Optional Hailo-8 runtime for accelerated YOLO detection.

    Returns:
        Perception facade conforming to ``ArmPerceptionProtocol``.

    Raises:
        ValueError: If arm_perception or arm_task config is not populated.
    """
    if cfg.arm_perception is None or cfg.arm_task is None:
        msg = "arm_perception and arm_task configs required for perception"
        raise ValueError(msg)

    import numpy as np

    from mousedroid.arm.perception.facade import ArmPerception

    # Default identity intrinsics — overridden by camera driver at runtime
    intrinsics = np.eye(3, dtype=np.float64)
    intrinsics[0, 0] = cfg.arm_perception.default_focal_length  # fx
    intrinsics[1, 1] = cfg.arm_perception.default_focal_length  # fy
    intrinsics[0, 2] = cfg.arm_perception.default_principal_x  # cx
    intrinsics[1, 2] = cfg.arm_perception.default_principal_y  # cy

    # Build Hailo YOLO detector if available
    detector = None
    if hailo_runtime is not None and cfg.arm_perception.yolo_backend in ("hailo", "auto"):
        from mousedroid.arm.perception.hailo_detector import HailoYOLODetector

        detector = HailoYOLODetector(cfg.arm_perception, hailo_runtime)
        _log.info("arm_perception_hailo_yolo_built")

    _log.info("arm_perception_built", camera=cfg.arm_perception.depth_camera_type)
    return ArmPerception(
        cfg.arm_perception,
        cfg.arm_task,
        intrinsics,
        object_detector=detector,
    )
