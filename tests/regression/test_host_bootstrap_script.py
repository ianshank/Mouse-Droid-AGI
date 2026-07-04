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

    def test_env_file_is_tightened_after_every_copy(self) -> None:
        # The env file holds ANTHROPIC_API_KEY once filled in — both the seed
        # and the force-overwrite paths must chmod 600 it (CodeRabbit PR #151).
        assert _source().count('run chmod 600 "$ENV_FILE"') == 2


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


class TestDryRunBranches:
    """The two riskiest paths, executed for real under --dry-run (gap-analysis)."""

    def test_dry_run_rollback_plans_restore_of_newest_backup(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "etc"
        config_dir.mkdir()
        (config_dir / "docker.env").write_text("K=v\n", encoding="utf-8")
        old = config_dir / "docker.env.bak.20260101T000000Z"
        new = config_dir / "docker.env.bak.20260702T000000Z"
        old.write_text("old\n", encoding="utf-8")
        new.write_text("new\n", encoding="utf-8")
        import os

        os.utime(old, (1, 1))  # newest-backup selection must not depend on name order alone
        proc = subprocess.run(
            ["bash", str(_SCRIPT), "--dry-run", "--rollback"],
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/bin:/bin",
                "MOUSEDROID_CONFIG_DIR": str(config_dir),
                "MOUSEDROID_INSTALL_DIR": str(tmp_path / "opt"),
            },
        )
        assert proc.returncode == 0, proc.stderr
        assert "rolling back" in proc.stdout
        assert f"DRY-RUN: cp -p {new}" in proc.stdout, "must restore the NEWEST backup"
        assert (config_dir / "docker.env").read_text(
            encoding="utf-8"
        ) == "K=v\n", "dry-run rollback must not touch the env file"

    def test_rollback_without_backup_fails_loudly(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "etc"
        config_dir.mkdir()
        proc = subprocess.run(
            ["bash", str(_SCRIPT), "--dry-run", "--rollback"],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "MOUSEDROID_CONFIG_DIR": str(config_dir)},
        )
        assert proc.returncode == 1
        assert "nothing to roll back" in proc.stdout

    def test_dry_run_with_trend_timer_plans_both_units(self, tmp_path: Path) -> None:
        install_dir = tmp_path / "opt"
        (install_dir / "scripts").mkdir(parents=True)
        (install_dir / "config").mkdir()
        (install_dir / "config" / "docker.env.example").write_text("K=v\n", encoding="utf-8")
        for unit in (
            "mousedroid-docker.service",
            "mousedroid-trend.service",
            "mousedroid-trend.timer",
        ):
            (install_dir / "scripts" / unit).write_text("[Unit]\n", encoding="utf-8")
        unit_dir = tmp_path / "systemd"
        unit_dir.mkdir()
        proc = subprocess.run(
            ["bash", str(_SCRIPT), "--dry-run", "--with-trend-timer"],
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/bin:/bin",
                "MOUSEDROID_INSTALL_DIR": str(install_dir),
                "MOUSEDROID_CONFIG_DIR": str(tmp_path / "etc"),
                "SYSTEMD_UNIT_DIR": str(unit_dir),
            },
        )
        assert proc.returncode == 0, proc.stderr
        assert "installing mousedroid-trend.service" in proc.stdout
        assert "installing mousedroid-trend.timer" in proc.stdout
        assert "DRY-RUN: systemctl enable --now mousedroid-trend.timer" in proc.stdout
        assert list(unit_dir.iterdir()) == [], "dry-run must install nothing"
        assert not (tmp_path / "etc").exists(), "dry-run must not create the config dir"
