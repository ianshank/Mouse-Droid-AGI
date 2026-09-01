"""Factory builders — the parked AutonomousOrchestrator stack.

Zero production callers (see
docs/architecture/ADR-016-autonomous-orchestrator-disposition.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mousedroid.constants import (
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_LIDAR_BUFFER_SIZE,
    DEFAULT_LIDAR_MAX_RANGE_M,
)
from mousedroid.interfaces.protocols import (
    CameraProtocol,
    LiDARProtocol,
    MetricsRegistryProtocol,
    MotorControllerProtocol,
)
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import (
        Settings,
    )

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

    if cfg.mock_hardware or not cfg.motor.enabled:
        _log.info("building_mock_motor_controller")
        return MockMotorController(metrics=metrics)

    port = cfg.motor.serial_port
    if resolver and cfg.usbc_discovery and cfg.usbc_discovery.enabled:
        resolved = getattr(resolver, "resolve_endpoint", lambda name: None)("esp32")
        if resolved:
            port = resolved

    baud = cfg.motor.baudrate
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

    port = cfg.lidar.serial_port if cfg.lidar else "/dev/ttyUSB1"
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
