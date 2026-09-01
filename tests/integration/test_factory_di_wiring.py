"""Integration tests verifying factory-first dependency injection wiring."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mousedroid.config.schema.hardware import (
    LidarConfig,
    MotorControllerConfig,
    USBCDiscoveryConfig,
    USBCEndpointSpec,
)
from mousedroid.config.schema.root import Settings
from mousedroid.factory import (
    build_autonomous_camera,
    build_autonomous_lidar,
    build_autonomous_metrics_registry,
    build_autonomous_orchestrator,
    build_motor_controller,
)
from mousedroid.interfaces.protocols import (
    CameraProtocol,
    LiDARProtocol,
    MetricsRegistryProtocol,
    MotorControllerProtocol,
)
from mousedroid.orchestrator.autonomous import AutonomousOrchestrator


def test_factory_builders_protocols() -> None:
    """Verify all factory builders return instances satisfying runtime protocols."""
    cfg = Settings(mock_hardware=True)

    metrics = build_autonomous_metrics_registry(cfg)
    assert isinstance(metrics, MetricsRegistryProtocol)

    motor = build_motor_controller(cfg, metrics=metrics)
    assert isinstance(motor, MotorControllerProtocol)

    cam = build_autonomous_camera(cfg)
    assert isinstance(cam, CameraProtocol)

    lidar = build_autonomous_lidar(cfg)
    assert isinstance(lidar, LiDARProtocol)

    orch = build_autonomous_orchestrator(cfg, metrics=metrics)
    assert isinstance(orch, AutonomousOrchestrator)


def test_factory_builders_real_hardware_branches() -> None:
    """Verify factory builders assemble real driver classes when mock_hardware is False."""
    cfg = Settings(
        mock_hardware=False,
        motor=MotorControllerConfig(enabled=True, serial_port="/dev/ttyUSB0"),
        lidar=LidarConfig(enabled=True, serial_port="/dev/ttyUSB1"),
        usbc_discovery=USBCDiscoveryConfig(
            enabled=True,
            required_endpoints=[
                USBCEndpointSpec(name="esp32", by_id_glob="*esp32*"),
                USBCEndpointSpec(name="lidar", by_id_glob="*lidar*"),
            ],
        ),
    )

    mock_resolver = MagicMock()
    mock_resolver.resolve_endpoint = MagicMock(
        side_effect=lambda name: f"/dev/serial/by-id/usb-{name}"
    )

    motor = build_motor_controller(cfg, resolver=mock_resolver)
    assert isinstance(motor, MotorControllerProtocol)
    assert motor._port == "/dev/serial/by-id/usb-esp32"

    cam = build_autonomous_camera(cfg)
    assert isinstance(cam, CameraProtocol)

    lidar = build_autonomous_lidar(cfg, resolver=mock_resolver)
    assert isinstance(lidar, LiDARProtocol)
    assert lidar._port == "/dev/serial/by-id/usb-lidar"


def test_factory_builders_disabled_and_no_resolver() -> None:
    """Verify fallback when resolver is None or subsystem is disabled."""
    cfg_disabled = Settings(
        mock_hardware=False,
        motor=MotorControllerConfig(enabled=False),
        lidar=LidarConfig(enabled=True, serial_port="/dev/ttyUSB1"),
    )
    mock_motor = build_motor_controller(cfg_disabled)
    assert isinstance(mock_motor, MotorControllerProtocol)

    # Distinctive, non-default ports: the resolver-override branch above
    # already proves the resolver path threads ``._port`` through; this
    # branch must independently prove the *configured* serial_port is what
    # gets used when no resolver is present — a value equal to the old
    # broken getattr fallback (or the schema default) would let a
    # regression to the wrong-field-name bug (factory/autonomous.py) pass
    # silently, since both happened to equal "/dev/ttyUSB1".
    cfg_no_resolver = Settings(
        mock_hardware=False,
        motor=MotorControllerConfig(enabled=True, serial_port="/dev/ttyACM3"),
        lidar=LidarConfig(enabled=True, serial_port="/dev/ttyACM7"),
    )
    motor_no_res = build_motor_controller(cfg_no_resolver, resolver=None)
    assert isinstance(motor_no_res, MotorControllerProtocol)
    assert motor_no_res._port == "/dev/ttyACM3"

    lidar_no_res = build_autonomous_lidar(cfg_no_resolver, resolver=None)
    assert isinstance(lidar_no_res, LiDARProtocol)
    assert lidar_no_res._port == "/dev/ttyACM7"


@pytest.mark.asyncio
async def test_factory_orchestrator_integration_cycle() -> None:
    """Verify factory-assembled AutonomousOrchestrator executes safely."""
    cfg = Settings(mock_hardware=True)
    orch = build_autonomous_orchestrator(cfg)

    assert orch.validate_sensors() is True

    success = await orch.execute_mission_step("scan perimeter")
    assert success is True
    await orch.stop()
