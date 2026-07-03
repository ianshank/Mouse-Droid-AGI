"""Regression pins for ``scripts/host_bootstrap.sh`` (F-017, WS-3.1).

Three layers, matching the repo's bash-testing conventions:

* ``bash -n`` syntax parse,
* source-text pins on the safety contract (pipefail, env-overridable paths,
  backup-before-overwrite, dry-run gate through ``run()``),
* a real subprocess ``--dry-run`` execution against a tmp
  ``MOUSEDROID_CONFIG_DIR`` proving the plan is printed and NOTHING is
  written.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "host_bootstrap.sh"


def _source() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


class TestSourceContract:
    def test_script_parses(self) -> None:
        subprocess.run(["bash", "-n", str(_SCRIPT)], check=True)

    def test_strict_mode_enabled(self) -> None:
        assert "set -euo pipefail" in _source()

    def test_paths_are_env_overridable_not_hardcoded(self) -> None:
        src = _source()
        assert 'INSTALL_DIR="${MOUSEDROID_INSTALL_DIR:-' in src
        assert 'CONFIG_DIR="${MOUSEDROID_CONFIG_DIR:-' in src
        # The literal default may appear ONLY inside the :- default expansion.
        for line in src.splitlines():
            if "/etc/mousedroid" in line and ":-" not in line and not line.strip().startswith("#"):
                raise AssertionError(f"hardcoded /etc/mousedroid outside default expansion: {line}")

    def test_backup_before_force_overwrite(self) -> None:
        src = _source()
        assert ".bak." in src, "force-overwrite must write a timestamped backup"
        assert "--rollback" in src, "a rollback path must exist"

    def test_mutations_go_through_run_gate(self) -> None:
        # Every mutation must route through the dry-run-aware run() wrapper.
        src = _source()
        assert "run cp" in src
        assert "run mkdir" in src
        assert "run systemctl" in src


class TestDryRunSubprocess:
    def test_dry_run_prints_plan_and_writes_nothing(self, tmp_path: Path) -> None:
        install_dir = tmp_path / "opt"
        config_dir = tmp_path / "etc"
        (install_dir / "config").mkdir(parents=True)
        (install_dir / "scripts").mkdir()
        (install_dir / "config" / "docker.env.example").write_text("A=1\n", encoding="utf-8")
        (install_dir / "scripts" / "mousedroid-docker.service").write_text(
            "[Unit]\n", encoding="utf-8"
        )

        proc = subprocess.run(
            ["bash", str(_SCRIPT), "--dry-run"],
            env={
                "PATH": "/usr/bin:/bin",
                "MOUSEDROID_INSTALL_DIR": str(install_dir),
                "MOUSEDROID_CONFIG_DIR": str(config_dir),
                "SYSTEMD_UNIT_DIR": str(tmp_path / "systemd"),
            },
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "DRY-RUN" in proc.stdout, "the plan must be echoed"
        assert not config_dir.exists(), "--dry-run must not create the config dir"
        assert not (tmp_path / "systemd").exists(), "--dry-run must not install units"

    def test_unknown_flag_is_rejected(self) -> None:
        proc = subprocess.run(
            ["bash", str(_SCRIPT), "--bogus"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 2
        assert "unknown argument" in proc.stderr

    def test_help_exits_zero(self) -> None:
        proc = subprocess.run(
            ["bash", str(_SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        assert "host_bootstrap.sh" in proc.stdout


def test_python_sees_same_interpreter() -> None:
    # Guard against the suite running under an unexpected interpreter — the
    # subprocess tests above assume a POSIX bash on PATH (matches CI + Jetson).
    assert sys.platform.startswith("linux") or sys.platform == "darwin"
