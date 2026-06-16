"""Regression: the spec-harness CLI shims preserve their backwards-compatible
contract (ADR-012).

`scripts/validate.py` and `scripts/select_next.py` were refactored into thin
shims over `mousedroid.harness.spec`. CI (`.github/workflows/harness.yml`),
`scripts/ci.sh`, and the seeded `features.yaml` commands all depend on the exact
exit codes and output shape, so this locks them against a future refactor.

Hermetic + fast: a synthetic catalog with ``validation_command: "true"`` is used
so the real (heavy, pytest-delegating) commands never run, and the shims are
invoked from a foreign CWD to pin the repo-root resolution.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VALIDATE = _REPO_ROOT / "scripts" / "validate.py"
_SELECT = _REPO_ROOT / "scripts" / "select_next.py"
_SCHEMA = _REPO_ROOT / "features.schema.json"


def _write_catalog(tmp_path: Path) -> Path:
    catalog = tmp_path / "features.yaml"
    catalog.write_text(
        textwrap.dedent(
            """\
            features:
              - id: "F-001"
                name: "synthetic done"
                category: "infrastructure"
                priority: "critical"
                status: "done"
                tier: "fast"
                verification: ["always true"]
                validation_command: "true"
                implemented_in: "HEAD"
                depends_on: []
              - id: "F-002"
                name: "synthetic todo"
                category: "functional"
                priority: "high"
                status: "todo"
                tier: "fast"
                verification: ["pending"]
                depends_on: ["F-001"]
            """
        )
    )
    return catalog


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_validate_passes_with_ok_summary(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path)
    # cwd=tmp_path (foreign) proves the shim resolves git/commands at the repo root.
    r = _run(
        [str(_VALIDATE), "--features", str(catalog), "--schema", str(_SCHEMA), "--tier", "fast"],
        cwd=tmp_path,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK:" in r.stdout
    assert "1 done" in r.stdout


def test_validate_fails_when_command_fails(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path)
    catalog.write_text(catalog.read_text().replace('"true"', '"false"'))
    r = _run(
        [str(_VALIDATE), "--features", str(catalog), "--schema", str(_SCHEMA), "--tier", "fast"],
        cwd=tmp_path,
    )
    assert r.returncode == 1
    assert "VALIDATION FAILED" in r.stdout


def test_validate_check_single_feature(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path)
    r = _run([str(_VALIDATE), "--features", str(catalog), "--check", "F-001"], cwd=tmp_path)
    assert r.returncode == 0
    assert "F-001: OK" in r.stdout


def test_validate_check_unknown_feature_exits_1(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path)
    r = _run([str(_VALIDATE), "--features", str(catalog), "--check", "F-999"], cwd=tmp_path)
    assert r.returncode == 1
    assert "unknown feature F-999" in r.stdout


def test_select_next_prints_ready_feature(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path)
    r = _run([str(_SELECT), str(catalog)], cwd=tmp_path)
    assert r.returncode == 0
    assert "F-002" in r.stdout


@pytest.mark.parametrize("script", [_VALIDATE, _SELECT])
def test_shims_exist(script: Path) -> None:
    assert script.is_file()
