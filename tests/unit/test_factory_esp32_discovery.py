"""Tests for the factory's usbc_discovery → esp32.serial_port override."""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.schema import (
    ESP32Config,
    Settings,
    USBCDiscoveryConfig,
    USBCEndpointSpec,
)
from mousedroid.factory import _resolve_esp32_serial_via_usbc_discovery


@pytest.fixture
def fake_by_id(tmp_path: Path) -> Path:
    root = tmp_path / "by-id"
    root.mkdir()
    return root


def _make_settings(
    *,
    serial_port: str,
    usbc: USBCDiscoveryConfig | None,
) -> Settings:
    return Settings(
        mock_hardware=True,
        esp32=ESP32Config(serial_port=serial_port),
        usbc_discovery=usbc,
    )


def test_returns_original_cfg_when_discovery_disabled(fake_by_id: Path) -> None:
    cfg = _make_settings(serial_port="/dev/anything", usbc=None)
    out = _resolve_esp32_serial_via_usbc_discovery(cfg)
    assert out is cfg.esp32  # identity — no copy made


def test_returns_original_cfg_when_literal_path_exists(fake_by_id: Path) -> None:
    """Honour the operator's explicit pin when the literal path is real."""
    pinned = fake_by_id / "rover-pinned"
    pinned.touch()
    usbc = USBCDiscoveryConfig(
        enabled=True,
        by_id_root=fake_by_id,
        required_endpoints=[
            USBCEndpointSpec(
                name="rover_esp32",
                by_id_glob="*CP2102N*",
            ),
        ],
    )
    cfg = _make_settings(serial_port=str(pinned), usbc=usbc)
    out = _resolve_esp32_serial_via_usbc_discovery(cfg)
    assert out is cfg.esp32


def test_overrides_with_resolved_path_when_literal_is_stale(
    fake_by_id: Path,
) -> None:
    live = fake_by_id / "usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_AAA-if00-port0"
    live.touch()
    usbc = USBCDiscoveryConfig(
        enabled=True,
        by_id_root=fake_by_id,
        required_endpoints=[
            USBCEndpointSpec(name="rover_esp32", by_id_glob="*CP2102N*-if00-port0"),
        ],
    )
    cfg = _make_settings(serial_port="/dev/serial/by-id/stale-path-that-does-not-exist", usbc=usbc)
    out = _resolve_esp32_serial_via_usbc_discovery(cfg)
    assert out is not cfg.esp32  # a new model_copy was made
    assert out.serial_port == str(live)


def test_returns_original_when_rover_esp32_endpoint_missing(
    fake_by_id: Path,
) -> None:
    usbc = USBCDiscoveryConfig(
        enabled=True,
        by_id_root=fake_by_id,
        required_endpoints=[
            USBCEndpointSpec(name="lidar", by_id_glob="*CP2102*"),
        ],
    )
    cfg = _make_settings(serial_port="/dev/serial/by-id/missing", usbc=usbc)
    out = _resolve_esp32_serial_via_usbc_discovery(cfg)
    assert out is cfg.esp32  # no rover_esp32 endpoint → no override


def test_returns_original_when_rover_glob_matches_nothing(
    fake_by_id: Path,
) -> None:
    usbc = USBCDiscoveryConfig(
        enabled=True,
        by_id_root=fake_by_id,
        required_endpoints=[
            USBCEndpointSpec(name="rover_esp32", by_id_glob="*NONEXISTENT*"),
        ],
    )
    cfg = _make_settings(serial_port="/dev/serial/by-id/also-missing", usbc=usbc)
    out = _resolve_esp32_serial_via_usbc_discovery(cfg)
    assert out is cfg.esp32  # discovery couldn't resolve → no override


def test_override_preserves_unrelated_esp32_fields(fake_by_id: Path) -> None:
    """Only serial_port is mutated; everything else (baud, budgets, etc) stays."""
    live = fake_by_id / "usb-Silicon_Labs_CP2102N_X-if00-port0"
    live.touch()
    usbc = USBCDiscoveryConfig(
        enabled=True,
        by_id_root=fake_by_id,
        required_endpoints=[
            USBCEndpointSpec(name="rover_esp32", by_id_glob="*CP2102N*-if00-port0"),
        ],
    )
    cfg = _make_settings(serial_port="/dev/stale", usbc=usbc)
    cfg = cfg.model_copy(
        update={"esp32": cfg.esp32.model_copy(update={"serial_baud": 230400})},
    )
    out = _resolve_esp32_serial_via_usbc_discovery(cfg)
    assert out.serial_port == str(live)
    assert out.serial_baud == 230400  # preserved
