"""Unit tests for :mod:`mousedroid.tools.print_healthcheck_env`."""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.loader import load_settings
from mousedroid.health.healthcheck_env import derive_healthcheck_env
from mousedroid.tools.print_healthcheck_env import main


def test_main_returns_zero_with_default_config(capsys: pytest.CaptureFixture[str]) -> None:
    """Default Settings render one KEY='VALUE' line per derived key, exit 0.

    The count is read from ``derive_healthcheck_env`` rather than hardcoded, so
    this pins the rendering (every key, one line each) while the key-set
    contract itself stays pinned exactly once, in ``test_healthcheck_env.py``.
    """
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    expected = derive_healthcheck_env(load_settings())
    assert len(lines) == len(expected)
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


def test_a_prefix_the_probe_cannot_use_does_not_stop_the_entrypoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The entrypoint runs this under ``set -eu`` before it execs the rover.

    A non-zero exit here is a container crash loop, so a schema-valid
    ``api_prefix`` the deploy probe cannot use must still exit 0 -- with the
    health path published empty ("cannot vouch") and the healthcheck's own
    keys written.
    """
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        'telemetry:\n  api_prefix: "/api v1"\nmock_hardware: true\n',
        encoding="utf-8",
    )

    rc = main(["--config", str(overlay)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "MOUSEDROID_RESOLVED_HEALTH_PATH=''\n" in out
    assert "MOUSEDROID_HEARTBEAT_PATH='" in out
