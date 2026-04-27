"""Regression tests for the mousedroid-docker.service systemd unit file.

Verifies structural properties of the service file so that automation-critical
lines (e.g. overlay-sync ExecStartPre) are not accidentally removed by a
future commit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SERVICE_FILE = Path("scripts/mousedroid-docker.service")


@pytest.fixture(scope="module")
def service_text() -> str:
    """Return the raw content of the systemd service file."""
    if not _SERVICE_FILE.exists():
        pytest.skip(f"{_SERVICE_FILE} not present in this checkout")
    return _SERVICE_FILE.read_text()


# ---------------------------------------------------------------------------
# Overlay-sync ExecStartPre presence
# ---------------------------------------------------------------------------


def test_sync_overlay_script_is_in_exec_start_pre(service_text: str) -> None:
    """An ExecStartPre line for sync_jetson_overlay.sh must be present."""
    lines = service_text.splitlines()
    matching = [ln for ln in lines if "sync_jetson_overlay.sh" in ln]
    assert matching, (
        "No ExecStartPre line containing 'sync_jetson_overlay.sh' found in "
        f"{_SERVICE_FILE}. Was the overlay-sync step removed?"
    )


def test_sync_overlay_uses_nonfatal_dash_prefix(service_text: str) -> None:
    """The overlay-sync ExecStartPre line must use the non-fatal dash prefix (=-).

    The dash prefix prevents a missing sync_jetson_overlay.sh from blocking
    service start on a freshly imaged Jetson.
    """
    lines = service_text.splitlines()
    for line in lines:
        if "sync_jetson_overlay.sh" in line and "ExecStartPre" in line:
            assert "ExecStartPre=-" in line, (
                f"overlay-sync ExecStartPre must use non-fatal form 'ExecStartPre=-', got: {line!r}"
            )
            return
    pytest.fail("sync_jetson_overlay.sh ExecStartPre line not found")


def test_install_dir_scripts_use_absolute_shell_wrapper(service_text: str) -> None:
    """Install-dir scripts must be launched via an absolute shell executable.

    systemd validates the first token in ExecStartPre as an executable path
    before it expands environment variables.  Shell-wrapping keeps the first
    token absolute while still allowing MOUSEDROID_INSTALL_DIR overrides.
    """
    lines = service_text.splitlines()
    sync_line = next(
        (ln for ln in lines if "ExecStartPre" in ln and "sync_jetson_overlay.sh" in ln),
        None,
    )
    preflight_line = next(
        (ln for ln in lines if "ExecStartPre" in ln and "preflight_check.sh" in ln),
        None,
    )

    assert sync_line is not None, "sync_jetson_overlay.sh ExecStartPre line not found"
    assert preflight_line is not None, "preflight_check.sh ExecStartPre line not found"
    assert sync_line.startswith("ExecStartPre=-/bin/bash -lc "), (
        "overlay-sync ExecStartPre must start with an absolute shell wrapper, "
        f"got: {sync_line!r}"
    )
    assert preflight_line.startswith("ExecStartPre=/bin/bash -lc "), (
        "preflight ExecStartPre must start with an absolute shell wrapper, "
        f"got: {preflight_line!r}"
    )


def test_sync_overlay_comes_before_preflight(service_text: str) -> None:
    """Overlay sync must appear before preflight_check.sh in the unit file.

    The config overlay must be on disk before preflight validation reads it.
    """
    lines = service_text.splitlines()
    sync_idx = next((i for i, ln in enumerate(lines) if "sync_jetson_overlay.sh" in ln), None)
    preflight_idx = next((i for i, ln in enumerate(lines) if "preflight_check.sh" in ln), None)
    assert sync_idx is not None, "sync_jetson_overlay.sh not found in service file"
    assert preflight_idx is not None, "preflight_check.sh not found in service file"
    assert sync_idx < preflight_idx, (
        "sync_jetson_overlay.sh must appear before preflight_check.sh "
        f"(got lines {sync_idx} and {preflight_idx})"
    )


# ---------------------------------------------------------------------------
# Environment-variable indirection (no bare hardcoded paths outside variables)
# ---------------------------------------------------------------------------


def test_env_file_line_present(service_text: str) -> None:
    """EnvironmentFile line with /etc/mousedroid/docker.env must be present."""
    assert "/etc/mousedroid/docker.env" in service_text


def test_compose_file_uses_env_var_indirection(service_text: str) -> None:
    """ExecStart* lines reference COMPOSE_FILE via ${COMPOSE_FILE} (no bare path)."""
    lines = service_text.splitlines()
    compose_lines = [
        ln
        for ln in lines
        if re.search(r"^\s*(ExecStartPre|ExecStart|ExecStop)=", ln) and "docker compose" in ln
    ]
    assert compose_lines, "No docker-compose ExecStart* lines found"
    for line in compose_lines:
        assert "${COMPOSE_FILE}" in line, (
            f"docker compose ExecStart line must use bare ${{COMPOSE_FILE}} (no default): {line!r}"
        )


def test_no_shell_default_expansion_in_exec_lines(service_text: str) -> None:
    """ExecStart* lines must not use bash-style ${VAR:-default} syntax.

    systemd performs its own env-var expansion and does not run commands
    through a shell by default.  The :- defaulting syntax would be passed
    literally to the process rather than expanded.
    """
    lines = service_text.splitlines()
    exec_lines = [ln for ln in lines if re.search(r"^\s*(ExecStartPre|ExecStart|ExecStop)=", ln)]
    for line in exec_lines:
        assert ":-" not in line, (
            f"ExecStart* line uses unsupported bash ':-' default syntax: {line!r}"
        )


def test_environment_defaults_are_set(service_text: str) -> None:
    """Service file must declare Environment= defaults for MOUSEDROID_INSTALL_DIR and COMPOSE_FILE.

    These allow ExecStart* lines to reference bare ${VAR} without shell defaulting.
    The EnvironmentFile can override them at deployment time.
    """
    assert "Environment=MOUSEDROID_INSTALL_DIR=" in service_text, (
        "Missing 'Environment=MOUSEDROID_INSTALL_DIR=' default in service file"
    )
    assert "Environment=COMPOSE_FILE=" in service_text, (
        "Missing 'Environment=COMPOSE_FILE=' default in service file"
    )


# ---------------------------------------------------------------------------
# Required [Unit] / [Service] / [Install] sections
# ---------------------------------------------------------------------------


def test_service_file_has_required_sections(service_text: str) -> None:
    """Service file must have [Unit], [Service], and [Install] sections."""
    for section in ("[Unit]", "[Service]", "[Install]"):
        assert section in service_text, f"Missing section: {section}"


def test_restart_policy_is_on_failure(service_text: str) -> None:
    """Restart=on-failure is the expected restart policy."""
    assert "Restart=on-failure" in service_text
