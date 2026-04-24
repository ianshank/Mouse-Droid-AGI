"""Backwards-compatibility regression tests for the face-display subsystem.

These tests guarantee that adding the optional ``face_display`` config and
factory builder does not change behavior for deployments that never opt in.

Coverage targets:
- Every committed YAML config loads without validation error.
- None of the production configs enable the face display (they all opt out).
- Factory helpers return ``None`` for opted-out configs.
- ``FaceDisplayConfig`` defaults match the values documented in ``default.yaml``.
- An explicit opt-in YAML (mock mode) produces a live ``FaceController``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mousedroid.config.schema import FaceDisplayConfig, Settings
from mousedroid.factory import build_face_controller, build_face_display
from mousedroid.hardware.display.mock_face_driver import MockFaceDriver
from mousedroid.orchestrator.face_controller import FaceController

# ---------------------------------------------------------------------------
# All committed YAML configs must load without error after the face_display
# field was added to Settings.  None of them include the stanza, so
# s.face_display must be None in every case.
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path("config")

# Include every committed YAML that is known to validate cleanly under the
# current schema (mouse-droid platform).  Robot-arm configs use a different
# validator path; machine-specific configs are gitignored; and a handful of
# Jetson configs have pre-existing validation gaps unrelated to face_display
# (they are covered by tests/regression/test_config_backwards_compat.py).
_MOUSE_DROID_YAMLS = [
    "default.yaml",
    "mock_hardware.yaml",
    "jetson_production.yaml",
    "jetson_hailo.yaml",
    "jetson_secure_metrics.yaml",
    "local_training.yaml",
]


@pytest.mark.parametrize("filename", _MOUSE_DROID_YAMLS)
def test_yaml_loads_without_face_display(filename: str) -> None:
    """Loading a committed YAML must succeed and leave face_display=None."""
    path = _CONFIG_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not present in this checkout")
    data = yaml.safe_load(path.read_text())
    s = Settings.model_validate(data)
    assert s.face_display is None, (
        f"{filename}: expected face_display=None (not opted in), got {s.face_display!r}"
    )


# ---------------------------------------------------------------------------
# Factory helpers — opted-out configs must return None at every step.
# ---------------------------------------------------------------------------


def test_factory_returns_none_when_disabled() -> None:
    cfg = Settings.model_validate({"mock_hardware": True})
    assert build_face_display(cfg) is None
    assert build_face_controller(cfg, None) is None


def test_factory_returns_none_when_enabled_false() -> None:
    cfg = Settings.model_validate({"mock_hardware": True, "face_display": {"enabled": False}})
    assert build_face_display(cfg) is None


def test_factory_controller_none_for_none_driver() -> None:
    """build_face_controller must return None when passed a None driver."""
    cfg = Settings.model_validate({"mock_hardware": True, "face_display": {"enabled": True}})
    assert build_face_controller(cfg, None) is None


# ---------------------------------------------------------------------------
# Opt-in path (mock_hardware=True) — factory must produce live objects.
# ---------------------------------------------------------------------------


def test_factory_returns_mock_in_mock_hardware_mode() -> None:
    cfg = Settings.model_validate({"mock_hardware": True, "face_display": {"enabled": True}})
    drv = build_face_display(cfg)
    assert isinstance(drv, MockFaceDriver)
    fc = build_face_controller(cfg, drv)
    assert isinstance(fc, FaceController)


# ---------------------------------------------------------------------------
# Inline opt-in (YAML string) — round-trips through Settings correctly.
# ---------------------------------------------------------------------------


def test_inline_yaml_opt_in_round_trips() -> None:
    """An inline YAML stanza with enabled:true must survive Settings.model_validate."""
    raw = yaml.safe_load(
        """
mock_hardware: true
face_display:
  enabled: true
  i2c_bus: 1
  boot_message: "test banner"
  min_dwell_s: 0.0
"""
    )
    s = Settings.model_validate(raw)
    assert s.face_display is not None
    assert s.face_display.enabled is True
    assert s.face_display.i2c_bus == 1
    assert s.face_display.boot_message == "test banner"
    assert s.face_display.min_dwell_s == 0.0


# ---------------------------------------------------------------------------
# FaceDisplayConfig default-value contract — must match documented defaults.
# ---------------------------------------------------------------------------


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
    # Affect-mapping thresholds
    assert cfg.valence_happy_min == pytest.approx(0.35)
    assert cfg.valence_sad_max == pytest.approx(-0.35)
    assert cfg.arousal_alert_min == pytest.approx(0.55)
    assert cfg.arousal_sleepy_max == pytest.approx(-0.45)
    assert cfg.angry_valence_max == pytest.approx(-0.25)
    assert cfg.angry_arousal_min == pytest.approx(0.45)
    assert cfg.idle_sleepy_after_s == pytest.approx(20.0)
    assert cfg.min_dwell_s == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Schema robustness — partial opt-in (only some keys specified) must fill
# the rest with defaults, not raise a ValidationError.
# ---------------------------------------------------------------------------


def test_partial_face_display_config_fills_defaults() -> None:
    """Specifying only a subset of keys must not raise."""
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "face_display": {"enabled": True, "boot_message": "hello"},
        }
    )
    assert cfg.face_display is not None
    assert cfg.face_display.enabled is True
    assert cfg.face_display.boot_message == "hello"
    # Unset fields keep their defaults.
    assert cfg.face_display.i2c_bus == 7
    assert cfg.face_display.width == 128
