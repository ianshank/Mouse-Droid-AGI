"""Planted-defect tests for the dead-code audit pipeline (F-020, WS-8.4).

The audit itself must be trustworthy: a planted unused function is found, a
clean package is clean, the allowlist suppresses verified-alive symbols, and
the JSON report lands where pointed. Uses the vulture Python API in-process
(same path the script takes), so no PATH/binary games.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("vulture")

from tests._script_loader import load_script_module

_audit = load_script_module("dead_code_audit")


def _package_with_dead_function(tmp_path: Path) -> Path:
    pkg = tmp_path / "planted"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mod.py").write_text(
        "def _never_called_anywhere():\n    return 42\n",
        encoding="utf-8",
    )
    return pkg


def _clean_package(tmp_path: Path) -> Path:
    pkg = tmp_path / "clean"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mod.py").write_text(
        "def used():\n    return 1\n\n\nused()\n",
        encoding="utf-8",
    )
    return pkg


def test_planted_unused_function_is_found(tmp_path: Path) -> None:
    findings = _audit.run_audit(
        [_package_with_dead_function(tmp_path)],
        allowlist=None,
        min_confidence=60,
    )
    assert any(f["name"] == "_never_called_anywhere" for f in findings), findings


def test_clean_package_yields_no_findings(tmp_path: Path) -> None:
    findings = _audit.run_audit([_clean_package(tmp_path)], allowlist=None, min_confidence=60)
    assert findings == []


def test_allowlist_suppresses_verified_alive_symbol(tmp_path: Path) -> None:
    pkg = _package_with_dead_function(tmp_path)
    allowlist = tmp_path / "allow.py"
    allowlist.write_text("_._never_called_anywhere\n", encoding="utf-8")
    findings = _audit.run_audit([pkg], allowlist=allowlist, min_confidence=60)
    assert not any(f["name"] == "_never_called_anywhere" for f in findings), findings


def test_report_written_to_requested_dir(tmp_path: Path) -> None:
    pkg = _package_with_dead_function(tmp_path)
    report_dir = tmp_path / "reports"
    rc = _audit.main(
        [
            "--paths",
            str(pkg),
            "--allowlist",
            "",
            "--report-dir",
            str(report_dir),
        ]
    )
    assert rc == 0, "advisory mode must exit 0 despite findings"
    reports = list(report_dir.glob("*.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert any(f["name"] == "_never_called_anywhere" for f in payload)


def test_strict_mode_gates_on_findings(tmp_path: Path) -> None:
    pkg = _package_with_dead_function(tmp_path)
    rc = _audit.main(
        [
            "--paths",
            str(pkg),
            "--allowlist",
            "",
            "--report-dir",
            str(tmp_path / "r"),
            "--strict",
        ]
    )
    assert rc == 1


def test_main_clean_package_prints_clean_and_writes_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pkg = _clean_package(tmp_path)
    report_dir = tmp_path / "reports"
    rc = _audit.main(["--paths", str(pkg), "--allowlist", "", "--report-dir", str(report_dir)])
    assert rc == 0
    assert "clean ->" in capsys.readouterr().out
    reports = list(report_dir.glob("*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text(encoding="utf-8")) == []


def test_max_print_truncates_console_but_not_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pkg = _package_with_dead_function(tmp_path)
    rc = _audit.main(
        [
            "--paths",
            str(pkg),
            "--allowlist",
            "",
            "--report-dir",
            str(tmp_path / "r"),
            "--max-print",
            "0",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "more in" in out, "truncation notice must point at the full JSON report"
    payload = json.loads(next((tmp_path / "r").glob("*.json")).read_text(encoding="utf-8"))
    assert payload, "the JSON report must carry the full finding list"
