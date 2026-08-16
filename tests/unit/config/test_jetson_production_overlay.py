"""Regression test — jetson_production overlay carries USB-C discovery block."""

from __future__ import annotations

from pathlib import Path

from mousedroid.config.loader import load_settings


def test_jetson_production_declares_usbc_endpoints() -> None:
    cfg = load_settings(Path("config/jetson_production.yaml"))
    assert cfg.usbc_discovery is not None
    assert cfg.usbc_discovery.enabled is True
    names = {ep.name for ep in cfg.usbc_discovery.required_endpoints}
    assert {"rover_esp32", "lidar_ld19"}.issubset(names)


def test_default_overlay_keeps_usbc_inert() -> None:
    """Default overlay must NOT auto-enable usbc_discovery."""
    cfg = load_settings(Path("config/default.yaml"))
    if cfg.usbc_discovery is None:
        return  # acceptable — None also satisfies "inert"
    assert cfg.usbc_discovery.enabled is False


def test_jetson_production_usbc_globs_align_with_serial_ports() -> None:
    """Wire-once: globs must match the serial_port already declared elsewhere."""
    cfg = load_settings(Path("config/jetson_production.yaml"))
    assert cfg.usbc_discovery is not None
    by_name = {ep.name: ep for ep in cfg.usbc_discovery.required_endpoints}
    # ESP32 serial port must match the rover_esp32 glob shape.
    assert "CP2102N" in by_name["rover_esp32"].by_id_glob
    assert "CP2102N" in cfg.esp32.serial_port
    # LiDAR serial port must match the lidar_ld19 glob shape.
    assert by_name["lidar_ld19"].by_id_glob.startswith("usb-Silicon_Labs_CP2102_")
    assert cfg.lidar is not None
    assert "CP2102" in cfg.lidar.serial_port
