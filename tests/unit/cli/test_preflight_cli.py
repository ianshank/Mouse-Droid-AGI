"""Smoke-pass: preflight CLI argparse + exit-code tests."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mousedroid.cli.preflight import main


def _extract_json_block(captured_out: str) -> dict[str, object]:
    match = re.search(r"(\{[\s\S]*\})\s*$", captured_out.strip())
    if match is None:
        msg = f"no JSON object found in captured stdout: {captured_out!r}"
        raise AssertionError(msg)
    return json.loads(match.group(1))


def test_mock_hardware_run_returns_exit_code_0(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--mock-hardware`` exits 0 with every check OK."""
    rc = main(["--mock-hardware"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "overall=ok" in captured.out


def test_json_mode_emits_parseable_document(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--json --mock-hardware`` writes a JSON doc + exits 0."""
    rc = main(["--mock-hardware", "--json"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = _extract_json_block(captured.out)
    assert "checks" in payload
    assert len(payload["checks"]) == 6  # camera/microphone/speaker/lidar/esp32/config


def test_checks_filter_restricts_dispatch(capsys: pytest.CaptureFixture[str]) -> None:
    """``--checks camera,esp32`` runs only the two named checks."""
    rc = main(["--mock-hardware", "--checks", "camera,esp32", "--json"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = _extract_json_block(captured.out)
    names = {c["name"] for c in payload["checks"]}
    assert names == {"camera", "esp32"}


def test_help_flag_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    """``--help`` exits 0 and prints argparse usage."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "preflight" in captured.out.lower()


def test_trend_without_journal_path_errors() -> None:
    """``--trend`` without ``--journal-path`` is a usage error (exit 2)."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--mock-hardware", "--trend"])
    assert excinfo.value.code == 2


def test_journal_path_records_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--journal-path`` persists the run as a parseable JSONL line."""
    import json as _json

    journal = tmp_path / "validation.jsonl"
    rc = main(["--mock-hardware", "--journal-path", str(journal), "--run-id", "t1"])
    assert rc == 0
    capsys.readouterr()
    lines = [ln for ln in journal.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    entry = _json.loads(lines[0])
    assert entry["event"] == "preflight_report"
    assert entry["payload"]["run_id"] == "t1"


def test_trend_prints_summary_after_two_runs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Second ``--trend`` run prints a trend verdict; mock runs are stable."""
    journal = tmp_path / "trend.jsonl"
    args = ["--mock-hardware", "--journal-path", str(journal), "--trend"]
    assert main([*args, "--run-id", "a"]) == 0
    capsys.readouterr()
    assert main([*args, "--run-id", "b"]) == 0
    out = capsys.readouterr().out
    assert "Trend:" in out


def test_fail_report_exits_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A FAIL aggregate status returns exit code 1 (the gate contract)."""
    from mousedroid.validation.preflight import (
        PreflightCheckResult,
        PreflightReport,
        PreflightStatus,
    )

    async def _fake_run_preflight(_cfg: object, **_kw: object) -> PreflightReport:
        return PreflightReport(
            checks=[
                PreflightCheckResult(name="esp32", status=PreflightStatus.FAIL, detail="down"),
            ],
            total_elapsed_s=0.0,
        )

    monkeypatch.setattr("mousedroid.cli.preflight.run_preflight", _fake_run_preflight)
    rc = main(["--mock-hardware"])
    assert rc == 1
    assert "overall=fail" in capsys.readouterr().out


def test_trend_threshold_flags_are_accepted(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Operator-tunable trend thresholds parse and run (no hardcoded call site)."""
    journal = tmp_path / "trend_thresh.jsonl"
    args = [
        "--mock-hardware",
        "--journal-path",
        str(journal),
        "--trend",
        "--trend-slow-ratio",
        "2.0",
        "--trend-slow-floor-s",
        "0.1",
    ]
    assert main([*args, "--run-id", "a"]) == 0
    capsys.readouterr()
    assert main([*args, "--run-id", "b"]) == 0
    assert "Trend:" in capsys.readouterr().out
