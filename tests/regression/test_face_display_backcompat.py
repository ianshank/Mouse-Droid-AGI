"""Backwards-compatibility regression tests for the face-display subsystem.

These tests guarantee that adding the optional ``face_display`` config and
factory builder does not change behavior for deployments that never opt in.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from mousedroid.config.schema import FaceDisplayConfig, Settings
from mousedroid.factory import build_face_controller, build_face_display


def test_default_yaml_loads_without_face_display() -> None:
    data = yaml.safe_load(Path("config/default.yaml").read_text())
    s = Settings.model_validate(data)
    assert s.face_display is None


def test_factory_returns_none_when_disabled() -> None:
    cfg = Settings.model_validate({"mock_hardware": True})
    assert build_face_display(cfg) is None
    assert build_face_controller(cfg, None) is None


def test_factory_returns_none_when_enabled_false() -> None:
    cfg = Settings.model_validate({"mock_hardware": True, "face_display": {"enabled": False}})
    assert build_face_display(cfg) is None


def test_factory_returns_mock_in_mock_hardware_mode() -> None:
    cfg = Settings.model_validate({"mock_hardware": True, "face_display": {"enabled": True}})
    drv = build_face_display(cfg)
    assert drv is not None
    fc = build_face_controller(cfg, drv)
    assert fc is not None


def test_face_display_config_defaults_match_documented_values() -> None:
    """All thresholds default to the values documented in default.yaml."""
    cfg = FaceDisplayConfig()
    assert cfg.enabled is False
    assert cfg.i2c_bus == 7
    assert cfg.i2c_address == 0x3C
    assert cfg.width == 128
    assert cfg.height == 64
    assert cfg.boot_message == "MSE-6 online"
    assert cfg.fallback_to_mock_on_error is True
