"""Smoke-pass: jetson_full_validation.sh arg surface (caching + multi-phase).

Exercises the script's dispatch + new ``--phases`` / ``--no-cache`` flags in
``--dry-run`` (which runs nothing and touches no hardware/docker), so the
contract holds on any host. Hardware behaviour is covered on the rover.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "jetson_full_validation.sh"


def _run(args: list[str], report_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "MOUSEDROID_VALIDATION_REPORT_ROOT": str(report_root)},
        cwd=str(_REPO_ROOT),
        timeout=60,
        check=False,
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
@pytest.mark.skipif(
    __import__("sys").platform == "win32",
    reason="jetson_full_validation.sh requires Unix PATH semantics",
)
class TestArgSurface:
    def test_help_exits_zero(self, tmp_path: Path) -> None:
        result = _run(["--help"], tmp_path)
        assert result.returncode == 0
        assert "--phases" in result.stdout
        assert "--no-cache" in result.stdout

    def test_invalid_phase_exits_two(self, tmp_path: Path) -> None:
        result = _run(["--phases", "9"], tmp_path)
        assert result.returncode == 2

    def test_dry_run_multi_phase_writes_summary(self, tmp_path: Path) -> None:
        result = _run(["--dry-run", "--phases", "0,1"], tmp_path)
        assert result.returncode == 0
        assert "PHASE 0" in result.stdout
        assert "PHASE 1" in result.stdout
        summaries = list(tmp_path.glob("*/SUMMARY.md"))
        assert len(summaries) == 1
        assert "PASS=" in summaries[0].read_text()

    def test_no_cache_flag_parses(self, tmp_path: Path) -> None:
        result = _run(["--dry-run", "--no-cache", "--phase", "1"], tmp_path)
        assert result.returncode == 0
