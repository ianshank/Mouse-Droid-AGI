"""Platform factory functions — build all components via dependency injection.

Factory functions eliminate platform branching. Each ``build_*()`` function
returns the correct implementation based on ``Settings``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.agents.base import AgentProtocol
from mousedroid.comms.motor_protocol import MotorControlProtocol
from mousedroid.comms.protocol import ESP32CommProtocol
from mousedroid.config.schema import PlatformType
from mousedroid.hardware.protocols import AudioProtocol, DistanceSensorProtocol, VisionProtocol
from mousedroid.logging.setup import get_logger
from mousedroid.safety.protocol import SafetyMonitorProtocol
from mousedroid.world_model.protocol import WorldModelProtocol

if TYPE_CHECKING:
    from mousedroid.cognitive.bdi_model import NeuralBDI
    from mousedroid.cognitive.cognitive_core import CognitiveCore
    from mousedroid.comms.flight_protocol import FlightControllerProtocol
    from mousedroid.config.schema import Settings, UltrasonicConfig

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


def _build_flight_controller(cfg: Settings) -> FlightControllerProtocol:
    """Build flight controller driver based on config.

    Args:
        cfg: Root settings.

    Returns:
        Flight controller conforming to ``FlightControllerProtocol``.
    """
    from mousedroid.config.schema import FlightControllerConfig

    fc_cfg = cfg.flight_controller or FlightControllerConfig()

    if cfg.mock_hardware:
        from mousedroid.comms.mock_flight_controller import MockFlightController

        return MockFlightController(fc_cfg)

    # Real hardware: currently only mock is implemented.
    # MAVLink driver would go here:
    # from mousedroid.comms.mavlink_driver import MAVLinkFlightController
    # inner = MAVLinkFlightController(fc_cfg)
    # Wrap with resilience:
    # from mousedroid.resilience.resilient_flight_controller import ResilientFlightController
    # return ResilientFlightController(inner, cfg.retry, cfg.circuit_breaker)

    from mousedroid.comms.mock_flight_controller import MockFlightController

    _log.warning(
        "real_flight_controller_not_yet_implemented_using_mock",
        protocol=fc_cfg.protocol,
    )
    return MockFlightController(fc_cfg)


def build_motor_controller(cfg: Settings) -> MotorControlProtocol:
    """Build platform-agnostic motor controller.

    For ground platform: wraps ESP32 driver in ``GroundMotorAdapter``.
    For drone platform: wraps flight controller in ``DroneMotorAdapter``.

    Args:
        cfg: Root settings.

    Returns:
        Motor controller conforming to ``MotorControlProtocol``.
    """
    if cfg.platform == PlatformType.DRONE:
        from mousedroid.comms.drone_adapter import DroneMotorAdapter

        fc = _build_flight_controller(cfg)
        _log.info("motor_controller_built", platform="drone")
        return DroneMotorAdapter(fc)

    from mousedroid.comms.ground_adapter import GroundMotorAdapter

    esp32 = build_esp32_driver(cfg)
    _log.info("motor_controller_built", platform="mouse_droid")
    return GroundMotorAdapter(esp32)


def build_camera(cfg: Settings) -> VisionProtocol:
    """Build camera driver based on config.

    Args:
        cfg: Root settings.

    Returns:
        Camera driver conforming to ``VisionProtocol``.
    """
    if cfg.mock_hardware:
        from mousedroid.hardware.camera.mock_camera import MockCamera

        return MockCamera(cfg.camera)

    if cfg.camera.backend == "jetson_csi":
        from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

        return JetsonCSICamera(cfg.camera)

    if cfg.camera.backend == "picamera2":
        from mousedroid.hardware.camera.imx500 import IMX500Camera

        return IMX500Camera(cfg.camera)

    # auto: try picamera2 first, fall back to jetson_csi
    try:
        from picamera2 import Picamera2  # noqa: F401

        from mousedroid.hardware.camera.imx500 import IMX500Camera

        return IMX500Camera(cfg.camera)
    except ImportError:
        from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

        return JetsonCSICamera(cfg.camera)


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

        ultrasonic_cfg: UltrasonicConfig = cfg.ultrasonic or UltraCfg(
            trigger_pin=0,
            echo_pin=0,
            max_range_m=4.0,
            min_range_m=0.02,
            timeout_s=0.1,
            speed_of_sound_mps=343.0,
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
    if cfg.microphone is None:
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


def build_safety_monitor(cfg: Settings) -> SafetyMonitorProtocol:
    """Build safety monitor for configured platform.

    Args:
        cfg: Root settings.

    Returns:
        Safety monitor conforming to ``SafetyMonitorProtocol``.
    """
    if cfg.platform == PlatformType.DRONE:
        from mousedroid.safety.drone_monitor import DroneSafetyMonitor

        _log.info("safety_monitor_built", platform="drone")
        return DroneSafetyMonitor(
            safety_cfg=cfg.safety,
            envelope_cfg=cfg.flight_envelope,
            geofence_cfg=cfg.geofence,
        )

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


def build_orchestrator(cfg: Settings) -> object:
    """Build fully-wired orchestrator.

    Args:
        cfg: Root settings.

    Returns:
        Fully configured ``MouseDroidOrchestrator``.
    """
    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
    from mousedroid.sensing.manager import SensorManager

    wm = build_world_model(cfg)
    agent = build_agent(cfg, wm)
    monitor = build_safety_monitor(cfg)
    motor = build_motor_controller(cfg)
    camera = build_camera(cfg)
    distance = build_distance_sensor(cfg)
    microphone = build_microphone(cfg)

    # Build platform-appropriate sensor manager.
    if cfg.platform == PlatformType.DRONE:
        from mousedroid.comms.drone_adapter import DroneMotorAdapter
        from mousedroid.sensing.drone_manager import DroneSensorManager

        # Extract the flight controller from the drone adapter.
        fc = motor._fc if isinstance(motor, DroneMotorAdapter) else _build_flight_controller(cfg)
        sensor_manager = DroneSensorManager(
            vision=camera,
            distance=distance,
            motor_controller=motor,
            flight_controller=fc,
            cfg=cfg,
            microphone=microphone,
        )
    else:
        sensor_manager = SensorManager(
            vision=camera,
            distance=distance,
            motor_controller=motor,
            cfg=cfg,
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

    _log.info(
        "orchestrator_built",
        platform=cfg.platform.value,
        mock_hardware=cfg.mock_hardware,
    )
    return MouseDroidOrchestrator(
        world_model=wm,
        agents=[agent],
        safety_monitor=monitor,
        motor_controller=motor,
        sensor_manager=sensor_manager,
        cognitive_core=cognitive_core,
        cfg=cfg,
    )
