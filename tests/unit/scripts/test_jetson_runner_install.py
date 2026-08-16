"""Structural + dry-run tests for the Jetson self-hosted runner installer.

These tests run on every CI host (not just Jetson). They never invoke the
real install path — only the `--dry-run` branch and structural assertions
on the systemd unit template + setup doc.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "jetson-runner-install.sh"
_SERVICE = _REPO_ROOT / "scripts" / "github-actions-runner.service.template"
_DOC = _REPO_ROOT / "docs" / "jetson-runner-setup.md"

_BASH = shutil.which("bash")
_BASH_AVAILABLE = pytest.mark.skipif(
    _BASH is None,
    reason="bash unavailable on this host",
)


def _run_script(
    *args: str,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke the installer with a pruned env so secrets cannot leak in."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
    }
    if env_extra:
        env.update(env_extra)
    assert _BASH is not None  # mypy
    return subprocess.run(
        [_BASH, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


# ---------------------------------------------------------------------------
# File-presence + permission structure
# ---------------------------------------------------------------------------


def test_installer_script_exists() -> None:
    assert _SCRIPT.is_file(), f"missing: {_SCRIPT}"


def test_systemd_template_exists() -> None:
    assert _SERVICE.is_file(), f"missing: {_SERVICE}"


def test_setup_doc_exists() -> None:
    assert _DOC.is_file(), f"missing: {_DOC}"


def test_systemd_template_has_required_sections() -> None:
    content = _SERVICE.read_text(encoding="utf-8")
    for section in ("[Unit]", "[Service]", "[Install]"):
        assert section in content, f"systemd template missing {section}"
    for key in ("ExecStart=", "Restart=", "User="):
        assert key in content, f"systemd template missing {key}"


def test_systemd_template_uses_placeholders() -> None:
    """The installer substitutes @RUNNER_USER@ + @RUNNER_INSTALL_DIR@ at install time."""
    content = _SERVICE.read_text(encoding="utf-8")
    assert "@RUNNER_USER@" in content
    assert "@RUNNER_INSTALL_DIR@" in content


def test_setup_doc_has_required_sections() -> None:
    content = _DOC.read_text(encoding="utf-8")
    for section in (
        "## Prerequisites",
        "## Install",
        "## Verify",
        "## Troubleshooting",
        "## Cross-Reference",
        "## Security",
    ):
        assert section in content, f"setup doc missing {section}"


# ---------------------------------------------------------------------------
# --dry-run behaviour
# ---------------------------------------------------------------------------


@_BASH_AVAILABLE
def test_dry_run_no_token_exits_zero() -> None:
    """--dry-run plus no RUNNER_TOKEN must succeed and print the plan."""
    out = _run_script("--dry-run")
    assert out.returncode == 0, f"stderr: {out.stderr}"
    assert "DRY-RUN" in out.stdout
    # The plan must reference the install directory + label set so an operator
    # can sanity-check before pasting the real token.
    assert "/opt/actions-runner" in out.stdout
    assert "self-hosted,jetson" in out.stdout


@_BASH_AVAILABLE
def test_dry_run_with_token_still_no_side_effects() -> None:
    """Even with RUNNER_TOKEN set, --dry-run must not curl/sudo anything."""
    out = _run_script("--dry-run", env_extra={"RUNNER_TOKEN": "fake-dry-run-token"})
    assert out.returncode == 0
    # We can't assert no side effects positively, but the dry-run path uses
    # `log` only — it never reaches the curl/sudo block. Smoke-check that the
    # plan text appears and the token doesn't leak into stdout.
    assert "DRY-RUN" in out.stdout
    assert "fake-dry-run-token" not in out.stdout
    assert "fake-dry-run-token" not in out.stderr


@_BASH_AVAILABLE
def test_real_run_without_token_exits_two_with_usage() -> None:
    """Real run (no --dry-run) without RUNNER_TOKEN must error with usage."""
    out = _run_script()
    assert out.returncode == 2, f"got rc={out.returncode}, stderr={out.stderr}"
    assert "RUNNER_TOKEN" in out.stderr
    assert "scripts/jetson-runner-install.sh --dry-run" in out.stderr


@_BASH_AVAILABLE
def test_unknown_arg_rejected() -> None:
    """Unknown CLI args must fail fast — keeps the contract obvious."""
    out = _run_script("--definitely-not-a-real-flag")
    assert out.returncode == 2
    assert "unknown arg" in out.stderr


@_BASH_AVAILABLE
def test_help_flag_prints_header() -> None:
    """--help / -h prints the file header (the ops contract docstring)."""
    out = _run_script("--help")
    assert out.returncode == 0
    assert "jetson-runner-install.sh" in out.stdout


# ---------------------------------------------------------------------------
# Cross-reference: the workflow consuming this runner exists.
# ---------------------------------------------------------------------------


def test_jetson_nightly_workflow_present() -> None:
    workflow = _REPO_ROOT / ".github" / "workflows" / "jetson-nightly.yml"
    assert workflow.is_file(), "Phase 3 has no workflow consumer"


def test_jetson_nightly_workflow_uses_self_hosted_label() -> None:
    workflow_text = (_REPO_ROOT / ".github" / "workflows" / "jetson-nightly.yml").read_text(
        encoding="utf-8"
    )
    # The runner registers with these labels; the workflow MUST select them.
    assert "self-hosted" in workflow_text
    assert "jetson" in workflow_text
