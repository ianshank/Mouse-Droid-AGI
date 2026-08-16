"""PR #106 backwards-compatibility regression tests.

PR #106 (Jetson + USB-C rover smoke validation) added the ``usbc_discovery``
config block and two ESP32 smoke-test fields. This file pins the invariants
from the project CLAUDE.md:

    > **9. Backwards compatibility**: New config fields MUST have defaults.
    > Existing YAML files must load unchanged.

and the single most safety-relevant behavior in the PR #106 surface: a pinned,
valid ``esp32.serial_port`` always wins over USB-C discovery and is never
silently shadowed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mousedroid.config.schema import Settings
from mousedroid.factory import _resolve_esp32_serial_via_usbc_discovery

# ---------------------------------------------------------------------------
# Default-value invariants — these must NEVER silently change
# ---------------------------------------------------------------------------


def test_usbc_discovery_defaults_to_none() -> None:
    """``Settings.usbc_discovery`` is ``Optional``, default ``None`` — a YAML
    file predating PR #106 has no ``usbc_discovery:`` key at all."""
    cfg = Settings.model_validate({"mock_hardware": True})
    assert cfg.usbc_discovery is None


def test_esp32_smoke_test_allow_motion_default_is_false() -> None:
    """The hard motion gate for ``tests/hardware/test_motor_smoke.py`` defaults
    ``False`` — motion during a smoke test is opt-in, never the default."""
    cfg = Settings.model_validate({"mock_hardware": True})
    assert cfg.esp32.smoke_test_allow_motion is False


def test_esp32_smoke_test_velocity_mps_default_is_005() -> None:
    """Documented default setpoint (0.05 m/s) — a change here would silently
    alter the smoke test's real-world motion profile."""
    cfg = Settings.model_validate({"mock_hardware": True})
    assert cfg.esp32.smoke_test_velocity_mps == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Existing-YAML must load unchanged
# ---------------------------------------------------------------------------


def test_minimal_legacy_yaml_loads_without_usbc_discovery() -> None:
    """A pre-PR-106 minimal YAML (no ``usbc_discovery`` key) loads cleanly and
    gets the safe defaults — no motion, discovery disabled."""
    legacy_yaml = """
    mock_hardware: true
    platform: mouse_droid
    esp32:
      protocol: serial
      serial_port: /dev/ttyUSB0
    """
    data = yaml.safe_load(legacy_yaml)
    cfg = Settings.model_validate(data)
    assert cfg.usbc_discovery is None
    assert cfg.esp32.smoke_test_allow_motion is False
    assert cfg.esp32.serial_port == "/dev/ttyUSB0"


def test_default_yaml_fixture_still_loads_clean() -> None:
    """The committed ``config/default.yaml`` loads with discovery disabled."""
    default_path = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
    if not default_path.exists():  # pragma: no cover - safety net for moved fixture
        return
    with default_path.open() as fh:
        data = yaml.safe_load(fh)
    cfg = Settings.model_validate(data)
    if cfg.usbc_discovery is not None:
        assert cfg.usbc_discovery.enabled is False


# ---------------------------------------------------------------------------
# Factory override chain — a pinned, valid serial_port always wins
# ---------------------------------------------------------------------------


def test_pinned_existing_serial_port_wins_over_discovery(tmp_path: Path) -> None:
    """When the literal ``esp32.serial_port`` exists on disk, discovery never
    overrides it — an operator's explicit pin is never silently shadowed,
    even when ``usbc_discovery.enabled=True``."""
    pinned_port = tmp_path / "ttyUSB-pinned"
    pinned_port.write_text("")  # just needs to exist on disk

    by_id_root = tmp_path / "by-id"
    by_id_root.mkdir()

    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "esp32": {"protocol": "serial", "serial_port": str(pinned_port)},
            "usbc_discovery": {
                "enabled": True,
                "by_id_root": str(by_id_root),
                "required_endpoints": [
                    {"name": "rover_esp32", "by_id_glob": "*CP2102N*-if00-port0"}
                ],
            },
        }
    )

    resolved = _resolve_esp32_serial_via_usbc_discovery(cfg)

    assert resolved.serial_port == str(pinned_port)


def test_discovery_disabled_returns_esp32_config_unchanged(tmp_path: Path) -> None:
    """``usbc_discovery`` absent (``None``) is a no-op — byte-identical
    pre-PR-106 behavior."""
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "esp32": {"protocol": "serial", "serial_port": "/dev/ttyUSB0"},
        }
    )
    assert cfg.usbc_discovery is None

    resolved = _resolve_esp32_serial_via_usbc_discovery(cfg)

    assert resolved is cfg.esp32
    assert resolved.serial_port == "/dev/ttyUSB0"
