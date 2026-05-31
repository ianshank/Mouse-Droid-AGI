"""Unit tests for the USB-C enumeration helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.schema import USBCDiscoveryConfig, USBCEndpointSpec
from mousedroid.diagnostics.usbc import (
    EndpointStatus,
    enumerate_usbc_devices,
    resolve_endpoint,
)


@pytest.fixture
def fake_by_id_root(tmp_path: Path) -> Path:
    root = tmp_path / "by-id"
    root.mkdir()
    (root / "usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_AAA-if00-port0").touch()
    (root / "usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0").touch()
    return root


def test_enumerate_resolves_required_endpoint(fake_by_id_root: Path) -> None:
    cfg = USBCDiscoveryConfig(
        enabled=True,
        by_id_root=fake_by_id_root,
        required_endpoints=[
            USBCEndpointSpec(
                name="rover_esp32",
                by_id_glob="*CP2102N_USB_to_UART_Bridge*-if00-port0",
            ),
        ],
    )
    result = enumerate_usbc_devices(cfg)
    rover = result["rover_esp32"]
    assert rover.status is EndpointStatus.PRESENT
    assert rover.resolved_path is not None
    assert "CP2102N" in rover.resolved_path.name


def test_enumerate_marks_missing_required_as_fail(fake_by_id_root: Path) -> None:
    cfg = USBCDiscoveryConfig(
        enabled=True,
        by_id_root=fake_by_id_root,
        required_endpoints=[
            USBCEndpointSpec(name="lidar", by_id_glob="*NONEXISTENT*", required=True),
        ],
    )
    result = enumerate_usbc_devices(cfg)
    assert result["lidar"].status is EndpointStatus.MISSING


def test_enumerate_marks_missing_optional_as_warn(fake_by_id_root: Path) -> None:
    cfg = USBCDiscoveryConfig(
        enabled=True,
        by_id_root=fake_by_id_root,
        required_endpoints=[
            USBCEndpointSpec(name="aux", by_id_glob="*NONEXISTENT*", required=False),
        ],
    )
    result = enumerate_usbc_devices(cfg)
    assert result["aux"].status is EndpointStatus.WARN


def test_enumerate_returns_empty_when_disabled() -> None:
    cfg = USBCDiscoveryConfig()
    assert enumerate_usbc_devices(cfg) == {}


def test_enumerate_picks_first_match_when_multiple_endpoints_match(
    fake_by_id_root: Path,
) -> None:
    """sorted() makes the resolution deterministic when multiple matches exist."""
    (
        fake_by_id_root / "usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_ZZZ-if00-port0"
    ).touch()
    cfg = USBCDiscoveryConfig(
        enabled=True,
        by_id_root=fake_by_id_root,
        required_endpoints=[
            USBCEndpointSpec(name="rover", by_id_glob="*CP2102N*-if00-port0"),
        ],
    )
    result = enumerate_usbc_devices(cfg)
    assert result["rover"].status is EndpointStatus.PRESENT
    assert result["rover"].resolved_path is not None
    # First sorted match wins → AAA before ZZZ.
    assert "AAA" in result["rover"].resolved_path.name


def test_resolve_endpoint_returns_path_when_present(fake_by_id_root: Path) -> None:
    """resolve_endpoint(name) returns the same path enumerate_usbc_devices would."""
    cfg = USBCDiscoveryConfig(
        enabled=True,
        by_id_root=fake_by_id_root,
        required_endpoints=[
            USBCEndpointSpec(name="rover_esp32", by_id_glob="*CP2102N*-if00-port0"),
        ],
    )
    path = resolve_endpoint(cfg, "rover_esp32")
    assert path is not None
    assert "CP2102N" in path.name


def test_resolve_endpoint_returns_none_when_missing(fake_by_id_root: Path) -> None:
    cfg = USBCDiscoveryConfig(
        enabled=True,
        by_id_root=fake_by_id_root,
        required_endpoints=[
            USBCEndpointSpec(name="aux", by_id_glob="*NONEXISTENT*"),
        ],
    )
    assert resolve_endpoint(cfg, "aux") is None


def test_resolve_endpoint_returns_none_when_name_unknown(fake_by_id_root: Path) -> None:
    cfg = USBCDiscoveryConfig(
        enabled=True,
        by_id_root=fake_by_id_root,
        required_endpoints=[
            USBCEndpointSpec(name="rover_esp32", by_id_glob="*CP2102N*-if00-port0"),
        ],
    )
    assert resolve_endpoint(cfg, "lidar") is None


def test_resolve_endpoint_returns_none_when_discovery_disabled() -> None:
    cfg = USBCDiscoveryConfig()
    assert resolve_endpoint(cfg, "rover_esp32") is None
