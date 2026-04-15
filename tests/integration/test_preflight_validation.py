"""Tests for the preflight check script.

Validates that the preflight_check.sh script exists and has the expected
structure. Full execution tests require a Linux environment with device
files, so we validate the script's existence and syntax here.
"""

from __future__ import annotations

from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
PREFLIGHT_SCRIPT = SCRIPTS_DIR / "preflight_check.sh"


def test_preflight_script_exists() -> None:
    """preflight_check.sh exists in scripts/."""
    assert PREFLIGHT_SCRIPT.exists(), f"Missing: {PREFLIGHT_SCRIPT}"


def test_preflight_script_has_shebang() -> None:
    """Script starts with bash shebang."""
    content = PREFLIGHT_SCRIPT.read_text(encoding="utf-8")
    assert content.startswith("#!/usr/bin/env bash"), "Missing bash shebang"


def test_preflight_script_uses_strict_mode() -> None:
    """Script uses set -euo pipefail for safety."""
    content = PREFLIGHT_SCRIPT.read_text(encoding="utf-8")
    assert "set -euo pipefail" in content


def test_preflight_script_checks_esp32() -> None:
    """Script checks for ESP32 device."""
    content = PREFLIGHT_SCRIPT.read_text(encoding="utf-8")
    assert "ESP32" in content or "ttyUSB0" in content


def test_preflight_script_checks_camera() -> None:
    """Script checks for camera device."""
    content = PREFLIGHT_SCRIPT.read_text(encoding="utf-8")
    assert "CAMERA" in content or "video0" in content


def test_preflight_script_checks_gpio() -> None:
    """Script checks for GPIO device files."""
    content = PREFLIGHT_SCRIPT.read_text(encoding="utf-8")
    assert "gpiochip" in content


def test_preflight_script_checks_disk_space() -> None:
    """Script checks disk space availability."""
    content = PREFLIGHT_SCRIPT.read_text(encoding="utf-8")
    assert "disk" in content.lower() or "df" in content


def test_preflight_script_checks_config() -> None:
    """Script checks for config file."""
    content = PREFLIGHT_SCRIPT.read_text(encoding="utf-8")
    assert "CONFIG" in content or "yaml" in content.lower()


def test_preflight_script_has_pass_fail_summary() -> None:
    """Script provides pass/fail summary."""
    content = PREFLIGHT_SCRIPT.read_text(encoding="utf-8")
    assert "PASS" in content
    assert "FAIL" in content


def test_preflight_script_uses_env_vars_for_device_paths() -> None:
    """Device paths are configurable via environment variables."""
    content = PREFLIGHT_SCRIPT.read_text(encoding="utf-8")
    assert "MOUSEDROID_ESP32_DEV" in content
    assert "MOUSEDROID_CAMERA_DEV" in content
    assert "MOUSEDROID_LIDAR_DEV" in content


def test_preflight_script_exits_nonzero_on_failure() -> None:
    """Script exits non-zero on failure."""
    content = PREFLIGHT_SCRIPT.read_text(encoding="utf-8")
    assert "exit 1" in content
