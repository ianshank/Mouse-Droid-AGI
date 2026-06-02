"""Unit tests for scripts/translate_mission.py (the operator dry-run probe).

The probe loads Settings, builds the LLM gateway via the factory, translates a
single NL mission, prints the resulting GoalVector + degraded/tier state, and
exits. The gateway is mocked end-to-end so no network, API key, or GGUF is
needed. Mirrors the test style of the greeting CLI.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mousedroid.llm_gateway.protocol import GoalVector

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "translate_mission.py"


def _load_cli():
    """Import the script module by path (it lives in scripts/, not a package)."""
    spec = importlib.util.spec_from_file_location("translate_mission", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_gateway(vector: GoalVector, *, degraded: bool = False) -> MagicMock:
    gw = MagicMock()
    gw.is_ready = True
    gw.is_degraded = degraded
    gw.start = AsyncMock(return_value=None)
    gw.stop = AsyncMock(return_value=None)
    gw.translate_mission = AsyncMock(return_value=vector)
    return gw


def test_translate_prints_goalvector_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    cli = _load_cli()
    gw = _fake_gateway(GoalVector(vx_target=0.4, vy_target=0.0, omega_target=-0.2))
    with (
        patch.object(cli, "load_settings", return_value=SimpleNamespace()),
        patch.object(cli, "build_llm_gateway", return_value=gw),
    ):
        rc = cli.main(["--mission", "patrol left then stop"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "vx_target" in out
    assert "0.4" in out
    gw.translate_mission.assert_awaited_once_with("patrol left then stop")
    gw.start.assert_awaited_once()
    gw.stop.assert_awaited_once()


def test_degraded_gateway_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    cli = _load_cli()
    gw = _fake_gateway(GoalVector(), degraded=True)
    with (
        patch.object(cli, "load_settings", return_value=SimpleNamespace()),
        patch.object(cli, "build_llm_gateway", return_value=gw),
    ):
        rc = cli.main(["--mission", "stop"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "degraded" in out.lower()


def test_missing_mission_arg_exits_nonzero() -> None:
    cli = _load_cli()
    with pytest.raises(SystemExit) as exc:  # argparse error
        cli.main([])
    assert exc.value.code != 0
