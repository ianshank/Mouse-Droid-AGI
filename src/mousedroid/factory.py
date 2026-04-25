"""Platform factory functions — build all components via dependency injection.

Factory functions eliminate platform branching. Each ``build_*()`` function
returns the correct implementation based on ``Settings``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.agents.base import AgentProtocol
from mousedroid.comms.protocol import ESP32CommProtocol
from mousedroid.efficiency.tensorrt import TensorRTCompilerProtocol
from mousedroid.hardware.protocols import (
    AudioProtocol,
    DistanceSensorProtocol,
    FaceDisplayProtocol,
    LidarProtocol,
    SpeakerProtocol,
    VisionProtocol,
)
from mousedroid.health.watchdog import WatchdogProtocol
from mousedroid.llm_gateway.protocol import LLMGatewayProtocol
from mousedroid.logging.setup import get_logger
from mousedroid.safety.protocol import SafetyMonitorProtocol
from mousedroid.voice.protocol import VoiceEngineProtocol
from mousedroid.world_model.protocol import WorldModelProtocol

if TYPE_CHECKING:
    from mousedroid.arm.protocols import (
        ArmControllerProtocol,
        ArmDriverProtocol,
        ArmEnvironmentProtocol,
        ArmPerceptionProtocol,
        ArmPlannerProtocol,
    )
    from mousedroid.cloud.protocol import (
        CloudExperienceExporterProtocol,
        CloudTelemetrySinkProtocol,
    )
    from mousedroid.cognitive.bdi_model import NeuralBDI
    from mousedroid.cognitive.cognitive_core import CognitiveCore
    from mousedroid.config.schema import Settings, UltrasonicConfig
    from mousedroid.curiosity.protocol import CuriosityProtocol
    from mousedroid.experience.logger import ExperienceLogger
    from mousedroid.hardware.accelerator.hailo_runtime import HailoRuntimeProtocol
    from mousedroid.health.monitor import HealthMonitor
    from mousedroid.llm_gateway.mission_parser import MissionParserProtocol
    from mousedroid.memory.tier import MemoryTier
    from mousedroid.orchestrator.face_controller import FaceController
    from mousedroid.sensing.manager import SensorManager
    from mousedroid.telemetry.log_buffer import LogRingBuffer
    from mousedroid.telemetry.metrics import MetricsRegistry
    from mousedroid.telemetry.protocol import TelemetryPublisherProtocol, TelemetryServerProtocol
    from mousedroid.voice.mock_tts import MockTTS
    from mousedroid.voice.tts import PiperTTS

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

    When a Hailo-8 runtime is provided, it is passed through to the
    camera constructor so that ``build_feature_extractor`` can select
    the :class:`HailoFeatureExtractor` backend at construction time.

    Args:
        cfg: Root settings.
        hailo_runtime: Optional Hailo-8 runtime for accelerated feature extraction.

    Returns:
        Camera driver conforming to ``VisionProtocol``.
    """
    if cfg.mock_hardware:
        from mousedroid.hardware.camera.mock_camera import MockCamera

        return MockCamera(cfg.camera)

    if cfg.camera.backend == "jetson_csi":
        from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

        return JetsonCSICamera(cfg.camera, hailo_runtime=hailo_runtime)

    if cfg.camera.backend == "picamera2":
        from mousedroid.hardware.camera.imx500 import IMX500Camera

        return IMX500Camera(cfg.camera, hailo_runtime=hailo_runtime)

    # auto: try picamera2 first, fall back to jetson_csi
    try:
        from picamera2 import Picamera2  # noqa: F401

        from mousedroid.hardware.camera.imx500 import IMX500Camera

        return IMX500Camera(cfg.camera, hailo_runtime=hailo_runtime)
    except ImportError:
        from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

        return JetsonCSICamera(cfg.camera, hailo_runtime=hailo_runtime)


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

        ultrasonic_cfg: UltrasonicConfig = cfg.ultrasonic or UltraCfg.model_validate(
            {"trigger_pin": 0, "echo_pin": 0}
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


def build_face_display(cfg: Settings) -> FaceDisplayProtocol | None:
    """Build the SSD1306 face-display driver based on config.

    Returns ``None`` when the subsystem is omitted from config or explicitly
    disabled, mirroring the other optional-hardware factories. The factory
    eagerly probes the I²C bus + address so that
    ``fallback_to_mock_on_error`` covers both:

    * import failures (``luma.oled`` / ``smbus2`` unavailable), and
    * runtime probe failures (panel disconnected, wrong address, missing
      I²C device node).

    When the probe fails and ``fallback_to_mock_on_error=True``, returns a
    :class:`MockFaceDriver` so the orchestrator can still come up. When the
    flag is ``False``, the failure is re-raised.

    Args:
        cfg: Root settings.

    Returns:
        Driver conforming to :class:`FaceDisplayProtocol`, or ``None``.
    """
    if cfg.face_display is None or not cfg.face_display.enabled:
        return None

    from mousedroid.hardware.display.mock_face_driver import MockFaceDriver

    if cfg.mock_hardware:
        _log.info("face_display_mock_built")
        return MockFaceDriver(cfg.face_display)

    try:
        from mousedroid.hardware.display.ssd1306_face_driver import SSD1306FaceDriver

        SSD1306FaceDriver.probe(cfg.face_display)
        _log.info(
            "face_display_real_built",
            i2c_bus=cfg.face_display.i2c_bus,
            i2c_address=cfg.face_display.i2c_address,
        )
        return SSD1306FaceDriver(cfg.face_display)
    except (ImportError, OSError):
        # ImportError → luma.oled/smbus2 missing; OSError → bus/addr/panel
        # unreachable.  All other exceptions propagate so programming errors
        # are never silently swallowed.
        if cfg.face_display.fallback_to_mock_on_error:
            _log.warning(
                "face_display_falling_back_to_mock",
                i2c_bus=cfg.face_display.i2c_bus,
                i2c_address=cfg.face_display.i2c_address,
                exc_info=True,
            )
            return MockFaceDriver(cfg.face_display)
        raise


def build_face_controller(
    cfg: Settings, driver: FaceDisplayProtocol | None
) -> FaceController | None:
    """Wrap a face-display driver in a :class:`FaceController`.

    Returns ``None`` when ``driver`` is ``None`` (subsystem disabled) so the
    orchestrator can simply skip face-display calls.

    Args:
        cfg: Root settings (must have ``face_display`` populated when
            ``driver`` is not ``None``).
        driver: Optional face-display driver from :func:`build_face_display`.

    Returns:
        :class:`FaceController` instance or ``None``.
    """
    if driver is None or cfg.face_display is None:
        return None

    from mousedroid.orchestrator.face_controller import FaceController

    return FaceController(driver, cfg.face_display)


def build_speaker(cfg: Settings) -> SpeakerProtocol | None:
    """Build USB speaker driver based on config.

    Args:
        cfg: Root settings.

    Returns:
        Speaker driver conforming to ``SpeakerProtocol``, or None if disabled.
    """
    if cfg.speaker is None or not cfg.speaker.enabled:
        _log.info("speaker_disabled")
        return None

    if cfg.mock_hardware:
        from mousedroid.hardware.audio.mock_speaker import MockSpeaker

        _log.info("speaker_mock_built", sample_rate=cfg.speaker.sample_rate)
        return MockSpeaker(cfg.speaker)

    from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

    _log.info(
        "speaker_built",
        sample_rate=cfg.speaker.sample_rate,
        device_name=cfg.speaker.device_name,
    )
    return UsbSpeaker(cfg.speaker)


def build_voice_engine(
    cfg: Settings,
    speaker: SpeakerProtocol | None = None,
) -> VoiceEngineProtocol | None:
    """Build Rocky voice engine based on config.

    Args:
        cfg: Root settings.
        speaker: Pre-built speaker driver (built if not provided).

    Returns:
        Voice engine conforming to ``VoiceEngineProtocol``, or None if disabled.
    """
    if not cfg.voice.enabled:
        _log.info("voice_engine_disabled")
        return None

    if speaker is None:
        speaker = build_speaker(cfg)
    if speaker is None:
        _log.warning("voice_engine_disabled_no_speaker")
        return None

    tts: MockTTS | PiperTTS
    try:
        if cfg.mock_hardware:
            from mousedroid.voice.mock_tts import MockTTS

            tts = MockTTS(cfg.voice)
        else:
            from mousedroid.voice.tts import PiperTTS

            tts = PiperTTS(cfg.voice)
    except Exception:
        _log.warning("voice_engine_tts_init_failed", exc_info=True)
        return None

    # Validate sample rate compatibility
    if speaker.sample_rate != cfg.voice.tts_sample_rate:
        _log.warning(
            "voice_engine_sample_rate_mismatch",
            speaker_rate=speaker.sample_rate,
            tts_rate=cfg.voice.tts_sample_rate,
        )
        return None

    from mousedroid.voice.rocky import RockyVoiceEngine

    engine = RockyVoiceEngine(cfg.voice, speaker, tts)
    _log.info(
        "voice_engine_built",
        personality=cfg.voice.personality,
        cooldown_s=cfg.voice.cooldown_s,
        sample_rate=speaker.sample_rate,
    )
    return engine


def build_world_model(cfg: Settings) -> WorldModelProtocol:
    """Build world model for configured platform.

    Selects :class:`~mousedroid.world_model.dual_stream_rssm.DualStreamRSSM`
    when ``cfc_hidden_dim > 0``, otherwise falls back to the classic
    :class:`~mousedroid.world_model.rssm.RSSM`.

    Args:
        cfg: Root settings.

    Returns:
        World model conforming to ``WorldModelProtocol``.
    """
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
                "world_model_dual_stream",
                gru_dim=cfg.model.hidden_dim,
                cfc_dim=cfg.model.cfc_hidden_dim,
            )
            return DualStreamRSSM(cfg.model)

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

    gateway_cfg = GatewayConfig(
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
        max_vx_norm_mps=cfg.llm.max_vx_norm_mps,
        max_vy_norm_mps=cfg.llm.max_vy_norm_mps,
        max_omega_norm_rads=cfg.llm.max_omega_norm_rads,
        max_command_len=cfg.llm.max_command_len,
        system_prompt=cfg.llm.system_prompt,
        injection_patterns=cfg.llm.injection_patterns,
    )
    _log.info("llm_gateway_built", enabled=cfg.llm.enabled)
    return LLMGateway(gateway_cfg)


def build_mission_parser(cfg: Settings) -> MissionParserProtocol:
    """Build NL mission parser with configurable speed/confidence mappings.

    Args:
        cfg: Root settings.

    Returns:
        Rule-based mission parser conforming to ``MissionParserProtocol``.
    """
    from mousedroid.llm_gateway.mission_parser import RuleBasedMissionParser

    parser = RuleBasedMissionParser(cfg.mission_parser)
    _log.info("mission_parser_built")
    return parser


def build_metrics_registry(cfg: Settings) -> MetricsRegistry | None:
    """Build the shared Prometheus metrics registry when metrics are enabled.

    Args:
        cfg: Root settings.

    Returns:
        Shared ``MetricsRegistry`` or ``None`` when metrics are disabled.
    """
    if not cfg.metrics.enabled:
        return None

    from mousedroid.telemetry.metrics import MetricsRegistry

    return MetricsRegistry(cfg.metrics)


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
    metrics_registry: MetricsRegistry | None = None,
    camera: VisionProtocol | None = None,
) -> TelemetryServerProtocol | None:
    """Build telemetry server if telemetry is enabled.

    Args:
        cfg: Root settings.
        publisher: Telemetry publisher to consume frames from.
        health_monitor: Health monitor for health endpoint.
        log_buffer: Optional log ring buffer for log streaming.
        metrics_registry: Optional shared metrics registry reused by other runtime components.
        camera: Optional vision driver; used as a raw-frame source for
            the MJPEG ``/camera/stream`` endpoint when it also implements
            :class:`RawFrameSourceProtocol`.

    Returns:
        ``TelemetryServer`` or ``None`` if telemetry disabled.
    """
    if not cfg.telemetry.enabled or publisher is None:
        return None

    if cfg.mock_hardware and not cfg.telemetry.force_real_server:
        from mousedroid.telemetry.mock_server import MockTelemetryServer

        _log.info("telemetry_mock_server_built")
        return MockTelemetryServer()

    shared_metrics_registry = metrics_registry
    metrics_path = cfg.metrics.path
    telemetry_metrics_path_default = type(cfg.telemetry).model_fields["metrics_path"].default
    metrics_path_default = type(cfg.metrics).model_fields["path"].default
    if (
        metrics_path == metrics_path_default
        and cfg.telemetry.metrics_path != telemetry_metrics_path_default
    ):
        metrics_path = cfg.telemetry.metrics_path

    if shared_metrics_registry is None and cfg.metrics.enabled:
        shared_metrics_registry = build_metrics_registry(cfg)

    from mousedroid.hardware.protocols import RawFrameSourceProtocol
    from mousedroid.telemetry.server import TelemetryServer

    raw_frame_source: RawFrameSourceProtocol | None = None
    if camera is not None and isinstance(camera, RawFrameSourceProtocol):
        raw_frame_source = camera

    _log.info(
        "telemetry_server_built",
        host=cfg.telemetry.host,
        port=cfg.telemetry.port,
        raw_frame_source=raw_frame_source is not None,
    )
    return TelemetryServer(
        cfg=cfg.telemetry,
        telemetry_queue=publisher.get_queue(),
        health_monitor=health_monitor,
        log_buffer=log_buffer,
        metrics_registry=shared_metrics_registry,
        metrics_path=metrics_path,
        publisher=publisher,
        lidar_max_range_m=cfg.lidar.max_range_m if cfg.lidar is not None else None,
        raw_frame_source=raw_frame_source,
        raw_frame_hz=cfg.telemetry.raw_frame_hz,
        cloud_enabled=cfg.gcp is not None,
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
    vision: VisionProtocol | None,
    distance: DistanceSensorProtocol | None,
    esp32: ESP32CommProtocol,
    microphone: AudioProtocol | None = None,
    lidar: LidarProtocol | None = None,
) -> SensorManager:
    """Build sensor manager for aggregated sensor reads.

    Args:
        cfg: Root settings.
        vision: Camera/vision protocol.
        distance: Distance sensor protocol.
        esp32: ESP32 communication protocol.
        microphone: Optional audio protocol.
        lidar: Optional LiDAR protocol.

    Returns:
        Configured ``SensorManager``.
    """
    from mousedroid.hardware.audio.feature_extractor import AudioFeatureExtractor
    from mousedroid.hardware.lidar.feature_extractor import LidarFeatureExtractor
    from mousedroid.sensing.manager import SensorManager

    if vision is None:
        # SensorManager requires a concrete VisionProtocol — raising here
        # keeps the protocol contract explicit for callers that forgot to
        # wire vision (rather than deferring to a late AttributeError on
        # first capture_features call).
        msg = "build_sensor_manager requires a non-None VisionProtocol (got None)"
        raise ValueError(msg)

    audio_extractor = build_audio_feature_extractor(cfg)
    typed_extractor: AudioFeatureExtractor | None = (
        audio_extractor if isinstance(audio_extractor, AudioFeatureExtractor) else None
    )

    lidar_extractor = build_lidar_feature_extractor(cfg)
    typed_lidar_extractor: LidarFeatureExtractor | None = (
        lidar_extractor if isinstance(lidar_extractor, LidarFeatureExtractor) else None
    )

    _log.info(
        "sensor_manager_built",
        audio_features_enabled=typed_extractor is not None,
        lidar_enabled=lidar is not None,
    )
    return SensorManager(
        vision=vision,
        distance=distance,
        esp32=esp32,
        cfg=cfg,
        microphone=microphone,
        audio_feature_extractor=typed_extractor,
        lidar=lidar,
        lidar_feature_extractor=typed_lidar_extractor,
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


def build_lidar(cfg: Settings) -> LidarProtocol | None:
    """Build LiDAR driver based on config.

    Returns ``MockLidar`` when ``mock_hardware`` is set, otherwise wraps
    a real ``LD19LidarDriver`` with circuit breaker + retry.

    Args:
        cfg: Root settings.

    Returns:
        LiDAR driver or ``None`` if LiDAR is disabled.
    """
    if cfg.lidar is None or not cfg.lidar.enabled:
        return None

    if cfg.mock_hardware:
        from mousedroid.hardware.lidar.mock_lidar import MockLidar

        _log.info("lidar_driver_mock_built")
        return MockLidar(cfg.lidar)

    from mousedroid.hardware.lidar.ld19_driver import LD19LidarDriver
    from mousedroid.resilience.resilient_lidar import ResilientLidarDriver

    inner = LD19LidarDriver(cfg.lidar)
    _log.info("lidar_driver_built", port=cfg.lidar.serial_port)
    return ResilientLidarDriver(inner, cfg.retry, cfg.circuit_breaker)


def build_lidar_feature_extractor(cfg: Settings) -> object | None:
    """Build LiDAR feature extractor if LiDAR is configured.

    Args:
        cfg: Root settings.

    Returns:
        ``LidarFeatureExtractor`` or ``None`` if LiDAR is disabled.
    """
    if cfg.lidar is None or not cfg.lidar.enabled:
        return None

    from mousedroid.hardware.lidar.feature_extractor import LidarFeatureExtractor

    extractor = LidarFeatureExtractor(cfg.lidar)
    _log.info("lidar_feature_extractor_built", feature_dim=extractor.feature_dim)
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
    """Instantiate Hailo-8 accelerator runtime if configured.

    Creates the runtime instance but does **not** start it — device
    discovery and HEF loading happen in ``await runtime.start()``,
    which is called by the orchestrator during its startup phase.

    Returns ``None`` when Hailo is disabled or the ``hailo_platform``
    package cannot be imported.

    Args:
        cfg: Root settings.

    Returns:
        Hailo runtime instance (not yet started) or ``None``.
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

    camera: VisionProtocol | None = None
    try:
        camera = build_camera(cfg, hailo_runtime=hailo_runtime)
    except Exception as exc:  # pylint: disable=broad-except
        _log.warning("camera_init_failed_degrading", error=str(exc))

    distance: DistanceSensorProtocol | None = None
    try:
        distance = build_distance_sensor(cfg)
    except Exception as exc:  # pylint: disable=broad-except
        _log.warning("distance_sensor_init_failed_degrading", error=str(exc))

    microphone = build_microphone(cfg)

    lidar_driver: LidarProtocol | None = None
    try:
        lidar_driver = build_lidar(cfg)
    except Exception as exc:  # pylint: disable=broad-except
        _log.warning("lidar_init_failed_degrading", error=str(exc))

    sensor_manager = build_sensor_manager(
        cfg,
        vision=camera,
        distance=distance,
        esp32=esp32,
        microphone=microphone,
        lidar=lidar_driver,
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
    from mousedroid.telemetry.log_buffer import LogRingBuffer as _LogRingBuffer

    log_buffer: _LogRingBuffer | None = None
    buffer_size = cfg.telemetry.log_stream_buffer
    if buffer_size:
        log_buffer = _LogRingBuffer(buffer_size)

    metrics_registry = build_metrics_registry(cfg)

    telemetry_server = build_telemetry_server(
        cfg,
        telemetry_publisher,
        health_monitor,
        log_buffer=log_buffer,
        metrics_registry=metrics_registry,
        camera=camera,
    )

    # LLM gateway + mission parser (optional — gated by llm.enabled)
    llm_gateway: LLMGatewayProtocol | None = None
    if cfg.llm.enabled:
        llm_gateway = build_llm_gateway(cfg)
    mission_parser: MissionParserProtocol | None = build_mission_parser(cfg)

    from mousedroid.common.tools.registry import create_default_registry

    _tool_registry = create_default_registry(
        llm_gateway=llm_gateway,
        metrics_registry=metrics_registry,
        gcp_cfg=cfg.gcp,
    )

    # Voice engine (optional — disabled by default)
    speaker = build_speaker(cfg)
    voice_engine = build_voice_engine(cfg, speaker=speaker)

    # Face display (optional — disabled by default)
    face_display = build_face_display(cfg)
    face_controller = build_face_controller(cfg, face_display)

    # Memory tier + experience logger + curiosity module (optional)
    memory_tier = build_memory_tier(cfg)
    experience_logger = build_experience_logger(cfg)
    curiosity_module = build_curiosity_module(cfg)

    # Watchdog notifier (optional — disabled by default)
    watchdog = build_watchdog(cfg)

    # GCP Digital Twin (optional — disabled when gcp=None)
    cloud_sink = build_cloud_telemetry_sink(cfg, metrics_registry=metrics_registry)
    cloud_experience_exporter = build_cloud_experience_exporter(cfg)

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
        voice_engine=voice_engine,
        hailo_runtime=hailo_runtime,
        memory_tier=memory_tier,
        experience_logger=experience_logger,
        curiosity_module=curiosity_module,
        llm_gateway=llm_gateway,
        mission_parser=mission_parser,
        watchdog=watchdog,
        cloud_sink=cloud_sink,
        cloud_experience_exporter=cloud_experience_exporter,
        tool_registry=_tool_registry,
        face_controller=face_controller,
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
