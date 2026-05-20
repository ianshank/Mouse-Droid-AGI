"""PR #104 backwards-compatibility regression tests.

PR #104 added three new schema fields that MUST inherit safe defaults so
existing YAML configs (rover deployments, CI fixtures, the operator's
``~/.config/mousedroid/`` overlay) keep loading byte-identically after a
``git pull``:

* ``CameraConfig.snapshot_jpeg_quality`` — Pillow JPEG quality (default 90).
* ``CameraConfig.v4l2_grayscale_extract`` — IMX708 Bayer workaround (default
  ``True``).
* ``ESP32Config.enabled`` — dev escape hatch (default ``True``).

These tests pin the invariants from the project CLAUDE.md:

    > **9. Backwards compatibility**: New config fields MUST have defaults.
    > Existing YAML files must load unchanged.

A failure here means a rover that ``git pull``ed will refuse to start or
silently switch behaviour. That's the #1 regression PR #104 must prevent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mousedroid.config.schema import Settings


def _repo_root() -> Path:
    """Locate the worktree root for loading ``config/*.yaml`` fixtures."""
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Default-value invariants — these must NEVER silently change
# ---------------------------------------------------------------------------


def test_esp32_enabled_default_is_true() -> None:
    """``ESP32Config.enabled`` defaults to ``True`` — legacy rover behaviour.

    Any rover that doesn't mention ``esp32.enabled`` in its YAML must keep
    talking to the ESP32 after this branch lands. Flipping the default
    silently would brick motor control on every existing deployment.
    """
    cfg = Settings.model_validate({"mock_hardware": True})
    assert cfg.esp32.enabled is True


def test_camera_v4l2_grayscale_extract_default_is_true() -> None:
    """``v4l2_grayscale_extract`` defaults to ``True`` — IMX708 workaround on.

    The current Jetson container ships without ``nvarguscamerasrc``; in V4L2
    fallback the IMX708 outputs solid green WITHOUT this workaround. Default
    must be ``True`` until the GStreamer plugin lands in the container.
    """
    cfg = Settings.model_validate({"mock_hardware": True})
    assert cfg.camera.v4l2_grayscale_extract is True


def test_camera_snapshot_jpeg_quality_default_is_90() -> None:
    """``snapshot_jpeg_quality`` defaults to 90 — matches Pillow's MJPEG default.

    Operators inspecting historical snapshots expect the file-size profile
    not to shift between minor releases. 90 is the documented default
    referenced from ``scripts/verify_sensors.py`` + the CHANGELOG.
    """
    cfg = Settings.model_validate({"mock_hardware": True})
    assert cfg.camera.snapshot_jpeg_quality == 90


# ---------------------------------------------------------------------------
# Existing-YAML must load unchanged
# ---------------------------------------------------------------------------


def test_minimal_legacy_yaml_loads() -> None:
    """A pre-PR-104 minimal YAML (no new fields) loads cleanly + gets defaults."""
    legacy_yaml = """
    mock_hardware: true
    platform: mouse_droid
    esp32:
      protocol: serial
      serial_port: /dev/ttyUSB0
    camera:
      resolution_width: 640
      resolution_height: 480
      fps: 30
    """
    data = yaml.safe_load(legacy_yaml)
    cfg = Settings.model_validate(data)
    # The three new PR #104 fields are present + carry the documented defaults.
    assert cfg.esp32.enabled is True
    assert cfg.camera.v4l2_grayscale_extract is True
    assert cfg.camera.snapshot_jpeg_quality == 90
    # And the legacy fields are NOT mutated.
    assert cfg.esp32.protocol == "serial"
    assert cfg.esp32.serial_port == "/dev/ttyUSB0"
    assert cfg.camera.resolution_width == 640
    assert cfg.camera.resolution_height == 480
    assert cfg.camera.fps == 30


def test_default_yaml_fixture_still_loads_clean() -> None:
    """The committed ``config/default.yaml`` loads + gets all new defaults.

    Catches a footgun: if the new schema field accidentally lacks a Pydantic
    default, ``Settings.model_validate(default_yaml)`` raises ValidationError
    and CI fails. We exercise the actual fixture the rover ships with.
    """
    default_path = _repo_root() / "config" / "default.yaml"
    if not default_path.exists():  # pragma: no cover - safety net for moved fixture
        return
    with default_path.open() as fh:
        data = yaml.safe_load(fh)
    cfg = Settings.model_validate(data)
    # New fields appear with the right default (the fixture doesn't override
    # any of them; it predates PR #104).
    assert cfg.esp32.enabled is True
    assert cfg.camera.v4l2_grayscale_extract is True
    assert cfg.camera.snapshot_jpeg_quality == 90


def test_arm_default_yaml_fixture_still_loads_clean() -> None:
    """``config/robot_arm_default.yaml`` still loads after the camera + esp32 fields.

    The arm platform shares ``CameraConfig`` / ``ESP32Config`` via the root
    Settings model. A regression in the camera defaults would break arm
    deployments even though they don't drive an MSE-6 chassis.
    """
    arm_path = _repo_root() / "config" / "robot_arm_default.yaml"
    if not arm_path.exists():  # pragma: no cover - same safety net
        return
    with arm_path.open() as fh:
        data = yaml.safe_load(fh)
    cfg = Settings.model_validate(data)
    # The arm config's camera + esp32 defaults still hold.
    assert cfg.esp32.enabled is True
    assert cfg.camera.v4l2_grayscale_extract is True
    assert cfg.camera.snapshot_jpeg_quality == 90


# ---------------------------------------------------------------------------
# Schema range invariants — operators set these from YAML
# ---------------------------------------------------------------------------


def test_snapshot_jpeg_quality_range_accepts_documented_extremes() -> None:
    """JPEG quality 1 + 100 (the documented operator range) parse cleanly."""
    cfg_lo = Settings.model_validate(
        {"mock_hardware": True, "camera": {"snapshot_jpeg_quality": 1}}
    )
    assert cfg_lo.camera.snapshot_jpeg_quality == 1
    cfg_hi = Settings.model_validate(
        {"mock_hardware": True, "camera": {"snapshot_jpeg_quality": 100}}
    )
    assert cfg_hi.camera.snapshot_jpeg_quality == 100


def test_snapshot_jpeg_quality_rejects_out_of_range() -> None:
    """JPEG quality 0 / 101 are rejected — guards against silent operator typos."""
    with pytest.raises(ValidationError, match="snapshot_jpeg_quality"):
        Settings.model_validate({"mock_hardware": True, "camera": {"snapshot_jpeg_quality": 0}})
    with pytest.raises(ValidationError, match="snapshot_jpeg_quality"):
        Settings.model_validate({"mock_hardware": True, "camera": {"snapshot_jpeg_quality": 101}})


# ---------------------------------------------------------------------------
# Field-coexistence invariant: enabling one new field doesn't disable another
# ---------------------------------------------------------------------------


def test_new_fields_independent() -> None:
    """Setting one new field doesn't reset another to a non-default value."""
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "esp32": {"enabled": False},
            "camera": {
                "v4l2_grayscale_extract": False,
            },
        }
    )
    assert cfg.esp32.enabled is False
    assert cfg.camera.v4l2_grayscale_extract is False
    # The un-touched new field still carries its documented default.
    assert cfg.camera.snapshot_jpeg_quality == 90
