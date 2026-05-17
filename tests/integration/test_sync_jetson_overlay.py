"""F-013 regression: scripts/sync_jetson_overlay.sh strengthening.

The smoke-stability sprint surfaced that the deployed
``/etc/mousedroid/jetson_production.yaml`` on the live Jetson was 4 days stale
because ``sync_jetson_overlay.sh`` silently no-op'd when the source was
missing and didn't emit any audible "synced" / "drifted" log lines. The
production container loaded the stale yaml, which lacked the
``mock_force_real_when_enabled: true`` line, which built ``MockTelemetryServer``,
which never bound port 8080 — dashboard dark.

The strengthened script adds:

* A ``--verify`` flag for an operator-runnable smoke (compare repo vs deployed
  hash, exit non-zero on drift, never mutate state).
* Audible ``overlay_sync_match`` / ``overlay_sync_replacing`` /
  ``overlay_sync_replaced`` / ``overlay_sync_drift`` log events.
* A WARN (not silent skip) when the source yaml is missing.

These integration tests exercise the script via ``subprocess`` against a
sandbox temp dir so they don't touch ``/etc/mousedroid`` or any host state.
Skipped on Windows where ``bash`` is not guaranteed available — the
operator runbook is Linux-only anyway.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "sync_jetson_overlay.sh"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "scripts/sync_jetson_overlay.sh is a Linux-only operator runbook script "
        "(invoked by scripts/mousedroid-docker.service as ExecStartPre). The "
        "Windows Python-to-bash subprocess bridge is unreliable (RPC handle "
        "mismatch in WSL/Git-Bash interop); Linux CI exercises this end-to-end."
    ),
)


def _run_script(
    *args: str,
    install_dir: Path,
    overlay_dst: Path,
) -> subprocess.CompletedProcess[str]:
    """Invoke sync_jetson_overlay.sh in a sandboxed env."""
    env = {
        "MOUSEDROID_INSTALL_DIR": str(install_dir),
        "MOUSEDROID_OVERLAY_DST": str(overlay_dst),
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }
    return subprocess.run(
        ["bash", str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _setup_sandbox(tmp_path: Path, src_text: str = "platform: mouse_droid\n") -> tuple[Path, Path]:
    """Create a fake install dir + dest path inside tmp_path."""
    install_dir = tmp_path / "install"
    config_dir = install_dir / "config"
    config_dir.mkdir(parents=True)
    src = config_dir / "jetson_production.yaml"
    src.write_text(src_text, encoding="utf-8")
    overlay_dst = tmp_path / "etc" / "mousedroid" / "jetson_production.yaml"
    return install_dir, overlay_dst


def test_verify_passes_when_src_matches_dst(tmp_path: Path) -> None:
    """``--verify`` returns 0 + logs ``overlay_sync_match`` when hashes match."""
    install_dir, overlay_dst = _setup_sandbox(tmp_path)
    overlay_dst.parent.mkdir(parents=True)
    overlay_dst.write_text((install_dir / "config" / "jetson_production.yaml").read_text("utf-8"))

    result = _run_script("--verify", install_dir=install_dir, overlay_dst=overlay_dst)
    assert result.returncode == 0, result.stderr
    assert "overlay_sync_match" in result.stderr


def test_verify_fails_on_drift(tmp_path: Path) -> None:
    """``--verify`` returns non-zero + logs ``overlay_sync_drift`` on hash mismatch."""
    install_dir, overlay_dst = _setup_sandbox(tmp_path)
    overlay_dst.parent.mkdir(parents=True)
    overlay_dst.write_text("stale content\n", encoding="utf-8")

    result = _run_script("--verify", install_dir=install_dir, overlay_dst=overlay_dst)
    assert result.returncode != 0
    assert "overlay_sync_drift" in result.stderr


def test_verify_fails_when_dst_missing(tmp_path: Path) -> None:
    """``--verify`` returns non-zero + logs ``overlay_sync_dst_missing`` when dst absent."""
    install_dir, overlay_dst = _setup_sandbox(tmp_path)
    # Do NOT create overlay_dst.

    result = _run_script("--verify", install_dir=install_dir, overlay_dst=overlay_dst)
    assert result.returncode != 0
    assert "overlay_sync_dst_missing" in result.stderr


def test_default_mode_replaces_drifted_dst(tmp_path: Path) -> None:
    """Default mode (no ``--verify``) replaces a drifted dst + logs ``overlay_sync_replaced``."""
    install_dir, overlay_dst = _setup_sandbox(tmp_path, src_text="platform: real_value\n")
    overlay_dst.parent.mkdir(parents=True)
    overlay_dst.write_text("platform: stale_value\n", encoding="utf-8")

    result = _run_script(install_dir=install_dir, overlay_dst=overlay_dst)
    assert result.returncode == 0, result.stderr
    assert "overlay_sync_replaced" in result.stderr
    assert overlay_dst.read_text(encoding="utf-8") == "platform: real_value\n"


def test_default_mode_skips_when_match(tmp_path: Path) -> None:
    """Default mode skips the copy when hashes already match — no spurious replace."""
    install_dir, overlay_dst = _setup_sandbox(tmp_path)
    overlay_dst.parent.mkdir(parents=True)
    overlay_dst.write_text((install_dir / "config" / "jetson_production.yaml").read_text("utf-8"))

    result = _run_script(install_dir=install_dir, overlay_dst=overlay_dst)
    assert result.returncode == 0, result.stderr
    assert "overlay_sync_match" in result.stderr
    assert "overlay_sync_replaced" not in result.stderr


def test_default_mode_warns_when_src_missing(tmp_path: Path) -> None:
    """Default mode WARNs (not silent) when the source yaml is missing.

    Backwards-compat: still returns 0 so the systemd unit doesn't fail on a
    partial repo checkout (the `-` prefix on ExecStartPre also makes this
    non-fatal). The visible WARN is the operator-facing improvement.
    """
    install_dir = tmp_path / "install" / "config"  # exists as empty dir, no yaml
    install_dir.mkdir(parents=True)
    overlay_dst = tmp_path / "etc" / "mousedroid" / "jetson_production.yaml"

    result = _run_script(install_dir=tmp_path / "install", overlay_dst=overlay_dst)
    assert result.returncode == 0
    assert "overlay_sync_source_missing" in result.stderr
