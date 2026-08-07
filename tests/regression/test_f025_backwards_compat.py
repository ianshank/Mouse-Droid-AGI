"""Backwards-compat pins for the F-025 command-set selector.

These tests pin the invariants from the project CLAUDE.md:

    > **9. Backwards compatibility**: New config fields MUST have defaults.
    > Existing YAML files must load unchanged.

A failure here means a rover that ``git pull``ed will refuse to start or
silently switch firmware protocol — flipping a legacy-firmware rover onto
stock command shapes (or vice versa) makes every motion command parse as
zeros, which is indistinguishable from a dead board at the smoke level.
That's the #1 regression F-025 must prevent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mousedroid.config.schema import WAVESHARE_STOCK_BAUD, ESP32Config, Settings

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_command_set_default_is_legacy() -> None:
    """``ESP32Config.command_set`` defaults to ``"legacy"``.

    Any rover whose YAML doesn't mention ``command_set`` must keep
    speaking the historical private protocol after this branch lands.
    Flipping the default silently would change every wire byte on every
    existing deployment.
    """
    cfg = Settings.model_validate({"mock_hardware": True})
    assert cfg.esp32.command_set == "legacy"


def test_new_field_defaults_are_inert() -> None:
    """Heartbeat/chassis knobs default to values with zero legacy effect."""
    cfg = ESP32Config()
    assert cfg.heartbeat_enabled is True  # consumed only under waveshare_stock
    assert cfg.heartbeat_window_multiple == 3.0
    assert cfg.chassis_has_wheel_encoders is True  # historical smoke assertion kept


def test_legacy_baud_default_untouched() -> None:
    """The schema baud default stays 1_000_000 under the legacy selector.

    The stock 115200 derivation must be reachable ONLY by opting into
    ``command_set='waveshare_stock'`` — an unrelated config load must
    never see its serial speed change.
    """
    cfg = ESP32Config()
    assert cfg.serial_baud == 1_000_000


def test_old_style_yaml_loads_with_new_defaults() -> None:
    """A pre-F-025 YAML snippet loads unchanged and gains inert defaults."""
    legacy_yaml = """
mock_hardware: true
esp32:
  protocol: serial
  serial_port: /dev/ttyUSB0
  serial_baud: 1000000
  max_velocity_mps: 0.5
"""
    cfg = Settings.model_validate(yaml.safe_load(legacy_yaml))
    assert cfg.esp32.command_set == "legacy"
    assert cfg.esp32.serial_baud == 1_000_000
    assert cfg.esp32.max_velocity_mps == 0.5  # legacy fields not mutated
    assert cfg.esp32.chassis_has_wheel_encoders is True


@pytest.mark.parametrize(
    "overlay",
    sorted(p.name for p in (_REPO_ROOT / "config").glob("*.yaml")),
)
def test_shipped_overlays_load_unchanged(overlay: str) -> None:
    """Every shipped config overlay still validates against the new schema."""
    path = _REPO_ROOT / "config" / overlay
    if not path.exists():  # pragma: no cover - repo layout guard
        return
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):  # pragma: no cover - comment-only overlay
        return
    cfg = Settings.model_validate({**raw, "mock_hardware": True})
    assert isinstance(cfg, Settings)
    # No shipped overlay opts into the stock selector in this PR (the
    # config-compat gate validates overlay edits against the DEPLOYED
    # image's schema, which predates the field — env override is the lever).
    assert cfg.esp32.command_set == "legacy"


def test_command_set_settable_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``MOUSEDROID_ESP32__COMMAND_SET=waveshare_stock`` reaches the field.

    This is the documented CLAUDE.md path (``MOUSEDROID_`` prefix + ``__``
    nested delimiter) and the ONLY sanctioned lever for the live rover
    until ``deployments/jetson-image.json`` is re-pinned to a schema that
    knows the field.
    """
    monkeypatch.setenv("MOUSEDROID_ESP32__COMMAND_SET", "waveshare_stock")
    cfg = Settings(mock_hardware=True)
    assert cfg.esp32.command_set == "waveshare_stock"
    assert cfg.esp32.serial_baud == WAVESHARE_STOCK_BAUD


def test_stock_derives_baud_only_without_explicit_pin() -> None:
    """The 115200 derivation defers to any explicit operator pin."""
    derived = ESP32Config(command_set="waveshare_stock")
    assert derived.serial_baud == WAVESHARE_STOCK_BAUD
    pinned = ESP32Config(command_set="waveshare_stock", serial_baud=921600)
    assert pinned.serial_baud == 921600


def test_stock_plus_wifi_rejected_at_load() -> None:
    """stock+wifi fails at YAML-parse time, not silently at runtime.

    Stock ``General_Driver`` firmware has no HTTP ``/cmd`` API — the pairing
    could only ever no-op, which on a rover means "motion commands vanish".
    """
    with pytest.raises(ValueError, match="/cmd"):
        ESP32Config(command_set="waveshare_stock", protocol="wifi")


def test_field_coexistence_no_cross_reset() -> None:
    """Setting one new field never resets another (model-level isolation)."""
    cfg = ESP32Config(chassis_has_wheel_encoders=False)
    assert cfg.command_set == "legacy"
    assert cfg.heartbeat_enabled is True
    assert cfg.serial_baud == 1_000_000
