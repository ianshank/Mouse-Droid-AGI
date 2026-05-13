"""Unit tests for :mod:`mousedroid.tools.print_healthcheck_env`."""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.tools.print_healthcheck_env import main


def test_main_returns_zero_with_default_config(capsys: pytest.CaptureFixture[str]) -> None:
    """Default Settings render to four KEY='VALUE' lines, exit 0."""
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 4
    for line in lines:
        # KEY='value' shape — single-quoted so shell can dot-source safely
        assert "='" in line, line
        assert line.endswith("'"), line


def test_main_emits_all_required_keys(capsys: pytest.CaptureFixture[str]) -> None:
    """All four contract keys appear in the output."""
    main([])
    out = capsys.readouterr().out
    for key in (
        "MOUSEDROID_HEARTBEAT_PATH",
        "MOUSEDROID_HEARTBEAT_STALE_S",
        "MOUSEDROID_START_GRACE_S",
        "MOUSEDROID_START_GRACE_FILE",
    ):
        assert f"{key}='" in out, f"missing key {key!r} in output: {out!r}"


def test_main_propagates_config_arg(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--config`` overlay flows through ``load_settings`` to the env output."""
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "loop:\n"
        "  watchdog_interval_s: 7.5\n"
        "  watchdog_tolerance_factor: 2.0\n"
        "mock_hardware: true\n",
        encoding="utf-8",
    )

    rc = main(["--config", str(overlay)])
    assert rc == 0
    out = capsys.readouterr().out
    # 7.5 * 2.0 = 15.000
    assert "MOUSEDROID_HEARTBEAT_STALE_S='15.000'" in out, out


def test_main_output_is_shell_sourceable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Output round-trips through a naive parser without breaking on quotes."""
    main([])
    out = capsys.readouterr().out

    parsed: dict[str, str] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        key, _, raw = line.partition("=")
        # Strip the wrapping single quotes that the CLI emits.
        assert raw.startswith("'"), line
        assert raw.endswith("'"), line
        parsed[key] = raw[1:-1]

    assert parsed["MOUSEDROID_HEARTBEAT_PATH"] == "/tmp/mousedroid_heartbeat"
    # No unescaped quotes inside any value (would break dot-sourcing).
    for value in parsed.values():
        assert "'" not in value, value
