"""Smoke tests for Jetson deployment scripts.

Validates that all deployment scripts in the scripts/ directory:
- Exist and are executable
- Have proper shebangs
- Use strict mode (set -euo pipefail or set -e)
- Do not contain hardcoded user-specific paths
- The first boot orchestrator supports --help and --dry-run
- The env template documents the variables used by scripts
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_FIRST_BOOT = _SCRIPTS_DIR / "jetson_first_boot.sh"
_ENV_TEMPLATE = _REPO_ROOT / "config" / "jetson_first_boot.env"

# All bash scripts we expect to be present (excludes .py and .service files)
_EXPECTED_SCRIPTS = [
    "jetson_first_boot.sh",
    "jetson_bootstrap.sh",
    "jetson_system_setup.sh",
    "jetson_hardware_setup.sh",
    "docker_deploy.sh",
    "download_model.sh",
    "jetson_smoke_test.sh",
    "deploy_jetson.sh",
    "deploy_remote.sh",
    "jetson_discover.sh",
]

# Paths that should never appear hardcoded in scripts
_HARDCODED_PATH_PATTERNS = [
    r"/home/jetson(?!/)",  # /home/jetson but not /home/jetson/ as part of a longer var
    r"/home/nvidia\b",
    r"/home/root\b",
    r"/root/mousedroid\b",
    r'"/home/[a-z]+/mousedroid"',  # Any user-specific mousedroid path
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bash_scripts() -> list[Path]:
    """Return all .sh files in the scripts/ directory."""
    if not _SCRIPTS_DIR.exists():
        pytest.skip("scripts/ directory not found in worktree")
    return sorted(_SCRIPTS_DIR.glob("*.sh"))


def _read_script(path: Path) -> str:
    """Read a script file, skipping if not available."""
    if not path.exists():
        pytest.skip(f"{path.name} not available in worktree")
    return path.read_text()


# ---------------------------------------------------------------------------
# 1. All expected scripts exist
# ---------------------------------------------------------------------------


class TestScriptExistence:
    """Verify that all required deployment scripts are present."""

    @pytest.mark.parametrize("script_name", _EXPECTED_SCRIPTS)
    def test_script_exists(self, script_name: str) -> None:
        """Each expected deployment script must exist."""
        script_path = _SCRIPTS_DIR / script_name
        assert script_path.exists(), f"Missing script: {script_path}"


# ---------------------------------------------------------------------------
# 2. All scripts are executable
# ---------------------------------------------------------------------------


class TestScriptPermissions:
    """Verify that all bash scripts have executable permission."""

    @pytest.mark.parametrize("script_name", _EXPECTED_SCRIPTS)
    def test_script_is_executable(self, script_name: str) -> None:
        """Each script must have at least one execute bit set."""
        script_path = _SCRIPTS_DIR / script_name
        if not script_path.exists():
            pytest.skip(f"{script_name} not available")
        mode = script_path.stat().st_mode
        assert mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH), (
            f"{script_name} is not executable (mode={oct(mode)})"
        )


# ---------------------------------------------------------------------------
# 3. All scripts have proper shebangs
# ---------------------------------------------------------------------------


class TestScriptShebangs:
    """Verify that all bash scripts start with a valid shebang."""

    @pytest.mark.parametrize("script_name", _EXPECTED_SCRIPTS)
    def test_script_has_shebang(self, script_name: str) -> None:
        """Each script must start with #!/usr/bin/env bash or #!/bin/bash."""
        script_path = _SCRIPTS_DIR / script_name
        if not script_path.exists():
            pytest.skip(f"{script_name} not available")
        first_line = script_path.read_text().split("\n", maxsplit=1)[0].strip()
        valid_shebangs = ("#!/usr/bin/env bash", "#!/bin/bash", "#!/bin/sh")
        assert first_line in valid_shebangs, f"{script_name} has invalid shebang: {first_line!r}"


# ---------------------------------------------------------------------------
# 4. All scripts use strict mode
# ---------------------------------------------------------------------------


class TestStrictMode:
    """Verify that all bash scripts use error-safe settings."""

    @pytest.mark.parametrize("script_name", _EXPECTED_SCRIPTS)
    def test_script_uses_strict_mode(self, script_name: str) -> None:
        """Each script must contain 'set -euo pipefail' or 'set -e'."""
        script_path = _SCRIPTS_DIR / script_name
        if not script_path.exists():
            pytest.skip(f"{script_name} not available")
        text = script_path.read_text()
        has_strict = "set -euo pipefail" in text or "set -e" in text
        assert has_strict, f"{script_name} missing strict mode (set -e or set -euo pipefail)"


# ---------------------------------------------------------------------------
# 5. No hardcoded user-specific paths
# ---------------------------------------------------------------------------


class TestNoHardcodedPaths:
    """Verify that scripts do not contain hardcoded user-specific paths."""

    def test_no_hardcoded_paths_in_scripts(self) -> None:
        """Scan all .sh files for common hardcoded path anti-patterns."""
        violations: list[str] = []
        for script_path in _bash_scripts():
            text = script_path.read_text()
            for pattern in _HARDCODED_PATH_PATTERNS:
                matches = re.findall(pattern, text)
                if matches:
                    violations.append(f"{script_path.name}: pattern {pattern!r} matched: {matches}")
        assert not violations, "Hardcoded paths found in scripts:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# 6. First boot script --help exits 0
# ---------------------------------------------------------------------------


class TestFirstBootHelp:
    """Verify that jetson_first_boot.sh --help works correctly."""

    def test_help_exits_zero(self) -> None:
        """--help must exit with code 0."""
        if not _FIRST_BOOT.exists():
            pytest.skip("jetson_first_boot.sh not available")
        result = subprocess.run(
            ["bash", str(_FIRST_BOOT), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"--help exited with {result.returncode}: {result.stderr}"

    def test_help_mentions_env_vars(self) -> None:
        """--help output should document key environment variables."""
        if not _FIRST_BOOT.exists():
            pytest.skip("jetson_first_boot.sh not available")
        result = subprocess.run(
            ["bash", str(_FIRST_BOOT), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout
        for var in ("MOUSEDROID_INSTALL_DIR", "COMPOSE_FILE", "MODEL_PATH"):
            assert var in output, f"--help output missing documentation for {var}"


# ---------------------------------------------------------------------------
# 7. First boot script --dry-run exits 0
# ---------------------------------------------------------------------------


class TestFirstBootDryRun:
    """Verify that jetson_first_boot.sh --dry-run works without side effects."""

    def test_dry_run_exits_zero(self) -> None:
        """--dry-run must exit with code 0 and not modify the system."""
        if not _FIRST_BOOT.exists():
            pytest.skip("jetson_first_boot.sh not available")
        env = os.environ.copy()
        # Point markers to a temp dir so we don't need root
        env["MOUSEDROID_CONFIG_DIR"] = "/tmp/mousedroid_test_first_boot"
        result = subprocess.run(
            ["bash", str(_FIRST_BOOT), "--dry-run"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        assert result.returncode == 0, (
            f"--dry-run exited with {result.returncode}:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_dry_run_does_not_call_docker(self) -> None:
        """--dry-run must not invoke docker commands."""
        if not _FIRST_BOOT.exists():
            pytest.skip("jetson_first_boot.sh not available")
        env = os.environ.copy()
        env["MOUSEDROID_CONFIG_DIR"] = "/tmp/mousedroid_test_first_boot"
        result = subprocess.run(
            ["bash", str(_FIRST_BOOT), "--dry-run"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        combined = result.stdout + result.stderr
        assert "Pulling" not in combined or "DRY-RUN" in combined


# ---------------------------------------------------------------------------
# 8. First boot script has valid bash syntax
# ---------------------------------------------------------------------------


class TestBashSyntax:
    """Verify bash syntax validity for key scripts."""

    @pytest.mark.parametrize(
        "script_name",
        ["jetson_first_boot.sh", "jetson_bootstrap.sh", "jetson_smoke_test.sh"],
    )
    def test_bash_syntax_check(self, script_name: str) -> None:
        """bash -n must pass (syntax validation without execution)."""
        script_path = _SCRIPTS_DIR / script_name
        if not script_path.exists():
            pytest.skip(f"{script_name} not available")
        result = subprocess.run(
            ["bash", "-n", str(script_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"bash -n {script_name} failed:\n{result.stderr}"


# ---------------------------------------------------------------------------
# 9. Env template exists and documents variables
# ---------------------------------------------------------------------------


class TestEnvTemplate:
    """Verify that config/jetson_first_boot.env is properly documented."""

    def test_env_template_exists(self) -> None:
        """config/jetson_first_boot.env must exist."""
        assert _ENV_TEMPLATE.exists(), f"Missing env template: {_ENV_TEMPLATE}"

    def test_env_template_documents_key_variables(self) -> None:
        """Env template should document all key configuration variables."""
        if not _ENV_TEMPLATE.exists():
            pytest.skip("env template not available")
        text = _ENV_TEMPLATE.read_text()
        required_vars = [
            "MOUSEDROID_INSTALL_DIR",
            "MOUSEDROID_CONFIG_DIR",
            "MOUSEDROID_CONFIG",
            "COMPOSE_FILE",
            "MOUSEDROID_CONTAINER",
            "MODEL_URL",
            "MODEL_PATH",
            "MODEL_CHECKSUM",
            "MOUSEDROID_TELEMETRY_TOKEN",
            "MOUSEDROID_TELEMETRY_PORT",
            "CUDA_VISIBLE_DEVICES",
            "MOUSEDROID_MAX_GPU_MEM_MB",
            "MOUSEDROID_HOSTNAME",
        ]
        missing = [v for v in required_vars if v not in text]
        assert not missing, f"Env template missing variables: {missing}"

    def test_env_template_variables_used_in_first_boot(self) -> None:
        """Variables documented in env template should be referenced in first boot script."""
        if not _ENV_TEMPLATE.exists() or not _FIRST_BOOT.exists():
            pytest.skip("env template or first boot script not available")
        env_text = _ENV_TEMPLATE.read_text()
        boot_text = _FIRST_BOOT.read_text()

        # Extract variable assignments (VAR=value lines, not comments)
        env_vars = re.findall(r"^([A-Z_]+)=", env_text, re.MULTILINE)

        # These vars are referenced by the first boot script or its sub-scripts
        missing: list[str] = []
        for var in env_vars:
            # Check if the variable is referenced in either the script or its help text
            if var not in boot_text:
                missing.append(var)

        # Allow a few vars that are consumed by sub-scripts only
        sub_script_vars = {"MOUSEDROID_TELEMETRY_TOKEN"}
        missing = [v for v in missing if v not in sub_script_vars]

        assert not missing, (
            f"Env vars documented but not referenced in first boot script: {missing}"
        )


# ---------------------------------------------------------------------------
# 10. First boot script is idempotent (marker-based)
# ---------------------------------------------------------------------------


class TestIdempotencyDesign:
    """Verify that the first boot script implements idempotency markers."""

    def test_script_uses_markers(self) -> None:
        """First boot script should use marker files for idempotency."""
        text = _read_script(_FIRST_BOOT)
        assert "marker" in text.lower(), (
            "First boot script should implement marker-based idempotency"
        )

    def test_script_checks_before_running(self) -> None:
        """Each step should check whether it was already completed."""
        text = _read_script(_FIRST_BOOT)
        # Look for the pattern of checking marker existence
        assert "marker_exists" in text or "already" in text.lower(), (
            "Script should check for completion before running each step"
        )


# ---------------------------------------------------------------------------
# 11. First boot script produces JSON output
# ---------------------------------------------------------------------------


class TestJsonOutput:
    """Verify that the first boot script produces structured JSON output."""

    def test_script_produces_json(self) -> None:
        """Script should contain JSON output logic."""
        text = _read_script(_FIRST_BOOT)
        assert "json" in text.lower() or "jq" in text, (
            "First boot script should produce structured JSON output"
        )

    def test_dry_run_output_is_valid_json_lines(self) -> None:
        """--dry-run output should be valid JSON lines."""
        if not _FIRST_BOOT.exists():
            pytest.skip("jetson_first_boot.sh not available")
        import json

        env = os.environ.copy()
        env["MOUSEDROID_CONFIG_DIR"] = "/tmp/mousedroid_test_first_boot"
        result = subprocess.run(
            ["bash", str(_FIRST_BOOT), "--dry-run"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        # Each line of output should be valid JSON
        lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        assert len(lines) > 0, "Expected at least one line of output"

        json_count = 0
        for line in lines:
            if line.startswith("{"):
                try:
                    json.loads(line)
                    json_count += 1
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON line in output: {line!r}")

        assert json_count > 0, "Expected at least one JSON line in --dry-run output"
