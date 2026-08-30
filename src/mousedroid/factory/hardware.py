"""Factory builders — raw sensor/actuator hardware drivers.

ESP32, camera, distance sensor, microphone, face display, LiDAR, sensor manager,
TensorRT, Hailo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.common.imports import module_importable
from mousedroid.comms.protocol import ESP32CommProtocol
from mousedroid.hardware.protocols import (
    AudioProtocol,
    DistanceSensorProtocol,
    FaceDisplayProtocol,
    LidarProtocol,
    VisionProtocol,
)
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import (
        ESP32Config,
        Settings,
        UltrasonicConfig,
    )
    from mousedroid.efficiency.tensorrt import TensorRTCompilerProtocol
    from mousedroid.hardware.accelerator.hailo_runtime import HailoRuntimeProtocol
    from mousedroid.sensing.manager import SensorManager

_log = get_logger(__name__)


def build_esp32_driver(cfg: Settings) -> ESP32CommProtocol:
    """Build ESP32 communication driver based on config.

    Wraps the underlying driver with circuit breaker + retry for
    fault tolerance.  The wrapper implements ``ESP32CommProtocol``
    so the orchestrator doesn't need to know about it.

    When ``cfg.usbc_discovery`` is enabled and declares a ``rover_esp32``
    endpoint, that endpoint's live by-id path supersedes the literal
    ``cfg.esp32.serial_port``. This keeps the config stable across rover
    swaps (CP2102N serial numbers differ per unit).

    Args:
        cfg: Root settings.

    Returns:
        ESP32 driver conforming to ``ESP32CommProtocol``.
    """
    inner: ESP32CommProtocol

    # ``cfg.esp32.enabled = False`` is the schema-driven dev escape hatch
    # for running the orchestrator on hardware where the ESP32 isn't
    # plugged in (e.g. Jetson + camera + LiDAR + Hailo for dashboard
    # verification — see PR #104 harden-2). The mock driver short-circuits
    # connect / send_velocity / emergency_stop without touching any serial
    # port, so the orchestrator's start() doesn't crash and tick rate isn't
    # dragged down by ResilientESP32Driver's open-circuit timeouts.
    if cfg.mock_hardware or not cfg.esp32.enabled:
        from mousedroid.comms.mock_driver import MockESP32Driver

        inner = MockESP32Driver(cfg.esp32)
    elif cfg.esp32.protocol == "serial":
        from mousedroid.comms.serial_driver import SerialESP32Driver

        esp32_cfg = _resolve_esp32_serial_via_usbc_discovery(cfg)
        inner = SerialESP32Driver(esp32_cfg)
    else:
        from mousedroid.comms.wifi_driver import WiFiESP32Driver

        inner = WiFiESP32Driver(cfg.esp32)

    from mousedroid.resilience.resilient_driver import ResilientESP32Driver

    # F-025: surface the selected firmware command set at build time so a
    # smoke-log grep shows which protocol the wire will carry (the
    # ``*_built``-with-discriminator house pattern).
    _log.info(
        "esp32_driver_built",
        driver=type(inner).__name__,
        command_set=cfg.esp32.command_set,
        protocol=cfg.esp32.protocol,
        enabled=cfg.esp32.enabled,
    )
    return ResilientESP32Driver(inner, cfg.retry, cfg.circuit_breaker)


def _resolve_esp32_serial_via_usbc_discovery(cfg: Settings) -> ESP32Config:
    """Override ``esp32.serial_port`` with the live rover_esp32 by-id path.

    Returns the original ESP32Config when discovery is disabled, the
    ``rover_esp32`` endpoint is absent, or the literal serial_port path
    already exists on disk (an exact match wins — avoids surprise
    overrides when the operator pinned a specific path).
    """
    from pathlib import Path as _Path

    if cfg.usbc_discovery is None or not cfg.usbc_discovery.enabled:
        return cfg.esp32
    if _Path(cfg.esp32.serial_port).exists():
        return cfg.esp32

    from mousedroid.diagnostics.usbc import resolve_endpoint

    resolved = resolve_endpoint(cfg.usbc_discovery, "rover_esp32")
    if resolved is None:
        _log.warning(
            "esp32_serial_port_unresolved",
            literal=cfg.esp32.serial_port,
            hint="usbc_discovery has no rover_esp32 endpoint matching the bus",
        )
        return cfg.esp32

    _log.info(
        "esp32_serial_port_overridden",
        literal=cfg.esp32.serial_port,
        resolved=str(resolved),
    )
    return cfg.esp32.model_copy(update={"serial_port": str(resolved)})


def build_camera(
    cfg: Settings,
    hailo_runtime: HailoRuntimeProtocol | None = None,
) -> VisionProtocol:
    """Build camera driver based on config.

    When a Hailo-8 runtime is provided, it is passed through to the
    camera constructor so that ``build_feature_extractor`` can select
    the :class:`HailoFeatureExtractor` backend at construction time.

    Real (non-mock) backends are wrapped with circuit breaker + retry,
    mirroring ``build_esp32_driver``/``build_lidar`` — the camera capture
    path talks to real hardware over CSI/USB and can transiently fail.

    Args:
        cfg: Root settings.
        hailo_runtime: Optional Hailo-8 runtime for accelerated feature extraction.

    Returns:
        Camera driver conforming to ``VisionProtocol``.
    """
    if cfg.mock_hardware:
        from mousedroid.hardware.camera.mock_camera import MockCamera

        return MockCamera(cfg.camera)

    inner: VisionProtocol
    if cfg.camera.backend == "jetson_csi":
        from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

        inner = JetsonCSICamera(cfg.camera, hailo_runtime=hailo_runtime)
    elif cfg.camera.backend == "picamera2":
        from mousedroid.hardware.camera.imx500 import IMX500Camera

        inner = IMX500Camera(cfg.camera, hailo_runtime=hailo_runtime)
    # auto: prefer picamera2 only when its stack *actually imports* (spec
    # presence is insufficient — picamera2 can resolve a spec yet fail to
    # import when its libcamera/native bindings are absent), else fall back
    # to jetson_csi.
    elif module_importable("picamera2"):
        from mousedroid.hardware.camera.imx500 import IMX500Camera

        _log.info(
            "camera_backend_resolved",
            backend="picamera2",
            driver="IMX500Camera",
            reason="picamera2_importable",
        )
        inner = IMX500Camera(cfg.camera, hailo_runtime=hailo_runtime)
    else:
        from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

        _log.info(
            "camera_backend_resolved",
            backend="jetson_csi",
            driver="JetsonCSICamera",
            reason="picamera2_not_importable",
        )
        inner = JetsonCSICamera(cfg.camera, hailo_runtime=hailo_runtime)

    from mousedroid.resilience.resilient_camera import ResilientCamera

    return ResilientCamera(inner, cfg.retry, cfg.circuit_breaker)


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

    Returns a real ``JetsonTensorRTCompiler`` when ``cfg.jetson.tensorrt_enabled``
    is True. The real compiler itself falls back to ``torch.jit.trace`` at
    compile time if ``torch2trt`` is missing (operators get a runtime warning
    on the first compile call); the ``torch2trt_available`` field in the
    structured-log event below surfaces that decision at boot time too so
    operator dashboards can ingest it without waiting for the first inference.

    Falls back to ``MockTensorRTCompiler`` when ``tensorrt_enabled`` is False.

    F-009: consolidated the previous two log events
    (``tensorrt_compiler_built`` / ``tensorrt_compiler_mock_built``) into a
    single ``tensorrt_compiler_built`` event with a ``backend`` label so
    operator dashboards can ingest backend selection without a label split.

    Args:
        cfg: Root settings.

    Returns:
        Compiler conforming to ``TensorRTCompilerProtocol``.
    """
    # Import _TORCH2TRT_AVAILABLE once so both branches log the truthful
    # boolean. Previously the mock branch hardcoded ``torch2trt_available=False``
    # which misled dashboards on dev hosts where torch2trt IS installed but
    # tensorrt is just disabled in cfg.
    from mousedroid.efficiency.tensorrt import _TORCH2TRT_AVAILABLE

    if cfg.jetson.tensorrt_enabled:
        from mousedroid.efficiency.tensorrt import JetsonTensorRTCompiler

        _log.info(
            "tensorrt_compiler_built",
            backend="real",
            torch2trt_available=_TORCH2TRT_AVAILABLE,
            precision=cfg.jetson.precision,
            cache_dir=str(cfg.jetson.tensorrt_cache_dir),
            reason="tensorrt_enabled=true",
        )
        return JetsonTensorRTCompiler(cfg.jetson)

    from mousedroid.efficiency.tensorrt import MockTensorRTCompiler

    _log.info(
        "tensorrt_compiler_built",
        backend="mock",
        torch2trt_available=_TORCH2TRT_AVAILABLE,
        reason="tensorrt_enabled=false",
    )
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
