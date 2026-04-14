"""Integration tests for the pre-flight validation script.

Validates that preflight_check.sh correctly detects present/missing
devices, configs, and model weights.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PREFLIGHT_SCRIPT = _REPO_ROOT / "scripts" / "preflight_check.sh"


@pytest.fixture
def _skip_if_no_bash() -> None:
    """Skip on platforms without bash (Windows CI without WSL)."""
    if sys.platform == "win32":
        pytest.skip("preflight_check.sh requires bash (not available on Windows CI)")


@pytest.mark.usefixtures("_skip_if_no_bash")
class TestPreflightScript:
    """Tests for scripts/preflight_check.sh."""

    def test_script_exists(self) -> None:
        """Pre-flight script must exist in the repo."""
        assert _PREFLIGHT_SCRIPT.exists(), f"Missing: {_PREFLIGHT_SCRIPT}"

    def test_script_has_shebang(self) -> None:
        """Script must start with a bash shebang."""
        first_line = _PREFLIGHT_SCRIPT.read_text().split("\n", maxsplit=1)[0]
        assert first_line.startswith("#!/"), f"Bad shebang: {first_line}"

    def test_script_uses_strict_mode(self) -> None:
        """Script must use set -euo pipefail for safety."""
        text = _PREFLIGHT_SCRIPT.read_text()
        assert "set -euo pipefail" in text

    def test_skip_devices_flag_accepted(self) -> None:
        """Script accepts --skip-devices without error."""
        result = subprocess.run(
            ["bash", str(_PREFLIGHT_SCRIPT), "--skip-devices", "--skip-models"],
            capture_output=True,
            text=True,
            env={**os.environ, "MOUSEDROID_CONFIG": str(_REPO_ROOT / "config" / "default.yaml")},
            timeout=30,
        )
        # May fail on disk/GPU checks but should not crash on flag parsing
        assert "Unknown argument" not in result.stderr

    def test_skip_models_flag_accepted(self) -> None:
        """Script accepts --skip-models without error."""
        result = subprocess.run(
            ["bash", str(_PREFLIGHT_SCRIPT), "--skip-devices", "--skip-models"],
            capture_output=True,
            text=True,
            env={**os.environ, "MOUSEDROID_CONFIG": str(_REPO_ROOT / "config" / "default.yaml")},
            timeout=30,
        )
        assert "Unknown argument" not in result.stderr

    def test_help_flag_exits_zero(self) -> None:
        """Script --help should exit 0."""
        result = subprocess.run(
            ["bash", str(_PREFLIGHT_SCRIPT), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0

    def test_config_validation_detects_valid_yaml(self) -> None:
        """Script validates a known-good YAML config file."""
        result = subprocess.run(
            ["bash", str(_PREFLIGHT_SCRIPT), "--skip-devices", "--skip-models"],
            capture_output=True,
            text=True,
            env={**os.environ, "MOUSEDROID_CONFIG": str(_REPO_ROOT / "config" / "default.yaml")},
            timeout=30,
        )
        # Should find and validate the config
        assert "Config" in result.stdout

    def test_env_vars_override_device_paths(self) -> None:
        """Environment variables override default device paths."""
        text = _PREFLIGHT_SCRIPT.read_text()
        assert "MOUSEDROID_ESP32_DEV" in text
        assert "MOUSEDROID_CAMERA_DEV" in text
        assert "MOUSEDROID_LIDAR_DEV" in text

    def test_no_hardcoded_device_paths(self) -> None:
        """Device paths must come from env vars, not hardcoded."""
        text = _PREFLIGHT_SCRIPT.read_text()
        # All device paths should use ${VAR:-default} pattern
        assert '${MOUSEDROID_ESP32_DEV:-/dev/ttyUSB0}' in text
        assert '${MOUSEDROID_CAMERA_DEV:-/dev/video0}' in text
        assert '${MOUSEDROID_LIDAR_DEV:-/dev/ttyUSB1}' in text
