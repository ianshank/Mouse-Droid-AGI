"""Smoke-pass: preflight CLI argparse + exit-code tests."""

from __future__ import annotations

import json
import re

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
