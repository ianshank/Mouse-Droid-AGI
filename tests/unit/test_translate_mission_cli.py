"""Unit tests for scripts/translate_mission.py (the operator dry-run probe).

The probe loads Settings, builds the LLM gateway via the factory, translates a
single NL mission, prints the resulting GoalVector + which tier served, and
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
    """A non-composite single gateway (no ``_primary``)."""
    gw = MagicMock(spec=["is_ready", "is_degraded", "start", "stop", "translate_mission"])
    gw.is_ready = True
    gw.is_degraded = degraded
    gw.start = AsyncMock(return_value=None)
    gw.stop = AsyncMock(return_value=None)
    gw.translate_mission = AsyncMock(return_value=vector)
    return gw


def _fake_composite(
    vector: GoalVector, *, primary_degraded: bool, both_degraded: bool = False
) -> MagicMock:
    """A composite gateway exposing ``_primary`` (mirrors FallbackLLMGateway)."""
    gw = MagicMock(
        spec=["is_ready", "is_degraded", "start", "stop", "translate_mission", "_primary"]
    )
    gw.is_ready = True
    gw.is_degraded = both_degraded
    gw._primary = SimpleNamespace(is_degraded=primary_degraded)
    gw.start = AsyncMock(return_value=None)
    gw.stop = AsyncMock(return_value=None)
    gw.translate_mission = AsyncMock(return_value=vector)
    return gw


# --------------------------------------------------------------------------- #
# Happy path + output formatting
# --------------------------------------------------------------------------- #
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
    assert "tier=primary" in out
    gw.translate_mission.assert_awaited_once_with("patrol left then stop")
    gw.start.assert_awaited_once()
    gw.stop.assert_awaited_once()


def test_single_gateway_degraded_tier_reported(capsys: pytest.CaptureFixture[str]) -> None:
    cli = _load_cli()
    gw = _fake_gateway(GoalVector(), degraded=True)
    with (
        patch.object(cli, "load_settings", return_value=SimpleNamespace()),
        patch.object(cli, "build_llm_gateway", return_value=gw),
    ):
        rc = cli.main(["--mission", "stop"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tier=degraded" in out


def test_composite_secondary_tier_reported(capsys: pytest.CaptureFixture[str]) -> None:
    """When the primary is degraded, the composite serves via the secondary."""
    cli = _load_cli()
    gw = _fake_composite(GoalVector(vx_target=0.1), primary_degraded=True)
    with (
        patch.object(cli, "load_settings", return_value=SimpleNamespace()),
        patch.object(cli, "build_llm_gateway", return_value=gw),
    ):
        rc = cli.main(["--mission", "creep forward"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "secondary" in out
    assert "tier=primary " not in out  # must NOT misreport as primary


def test_composite_primary_tier_reported(capsys: pytest.CaptureFixture[str]) -> None:
    cli = _load_cli()
    gw = _fake_composite(GoalVector(vx_target=0.3), primary_degraded=False)
    with (
        patch.object(cli, "load_settings", return_value=SimpleNamespace()),
        patch.object(cli, "build_llm_gateway", return_value=gw),
    ):
        rc = cli.main(["--mission", "go forward"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tier=primary" in out


def test_composite_both_degraded_tier_reported(capsys: pytest.CaptureFixture[str]) -> None:
    cli = _load_cli()
    gw = _fake_composite(GoalVector(), primary_degraded=True, both_degraded=True)
    with (
        patch.object(cli, "load_settings", return_value=SimpleNamespace()),
        patch.object(cli, "build_llm_gateway", return_value=gw),
    ):
        rc = cli.main(["--mission", "stop"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "degraded" in out  # both tiers down must be visible


# --------------------------------------------------------------------------- #
# Config resolution (the production gap: MOUSEDROID_CONFIG must be honored)
# --------------------------------------------------------------------------- #
def test_explicit_config_is_passed_to_load_settings() -> None:
    cli = _load_cli()
    gw = _fake_gateway(GoalVector())
    captured: dict[str, object] = {}

    def _capture(*paths: Path) -> SimpleNamespace:
        captured["paths"] = paths
        return SimpleNamespace()

    with (
        patch.object(cli, "load_settings", side_effect=_capture),
        patch.object(cli, "build_llm_gateway", return_value=gw),
    ):
        rc = cli.main(["--mission", "go", "--config", "/etc/mousedroid/jetson_production.yaml"])
    assert rc == 0
    assert captured["paths"] == (Path("/etc/mousedroid/jetson_production.yaml"),)


def test_env_config_resolved_when_no_cli_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """MOUSEDROID_CONFIG must be resolved (the bug this fixes)."""
    cli = _load_cli()
    gw = _fake_gateway(GoalVector())
    captured: dict[str, object] = {}

    def _capture(*paths: Path) -> SimpleNamespace:
        captured["paths"] = paths
        return SimpleNamespace()

    monkeypatch.delenv("MOUSEDROID_CONFIGS", raising=False)
    monkeypatch.delenv("MOUSEDROID_JETSON_CONFIGS", raising=False)
    monkeypatch.setenv("MOUSEDROID_CONFIG", "/etc/mousedroid/jetson_production.yaml")
    with (
        patch.object(cli, "load_settings", side_effect=_capture),
        patch.object(cli, "build_llm_gateway", return_value=gw),
    ):
        rc = cli.main(["--mission", "go"])
    assert rc == 0
    assert captured["paths"] == (Path("/etc/mousedroid/jetson_production.yaml"),)


# --------------------------------------------------------------------------- #
# Error-exit contract + cleanup guarantee
# --------------------------------------------------------------------------- #
def test_config_load_error_exits_2() -> None:
    cli = _load_cli()
    with patch.object(cli, "load_settings", side_effect=FileNotFoundError("no such file")):
        rc = cli.main(["--mission", "go"])
    assert rc == 2


def test_build_error_exits_1() -> None:
    cli = _load_cli()
    with (
        patch.object(cli, "load_settings", return_value=SimpleNamespace()),
        patch.object(cli, "build_llm_gateway", side_effect=RuntimeError("build failed")),
    ):
        rc = cli.main(["--mission", "go"])
    assert rc == 1


def test_runtime_error_exits_1_and_still_stops() -> None:
    cli = _load_cli()
    gw = _fake_gateway(GoalVector())
    gw.translate_mission = AsyncMock(side_effect=RuntimeError("network down"))
    with (
        patch.object(cli, "load_settings", return_value=SimpleNamespace()),
        patch.object(cli, "build_llm_gateway", return_value=gw),
    ):
        rc = cli.main(["--mission", "go"])
    assert rc == 1
    gw.stop.assert_awaited_once()  # cleanup runs even when translate raises


def test_start_failure_still_stops() -> None:
    cli = _load_cli()
    gw = _fake_gateway(GoalVector())
    gw.start = AsyncMock(side_effect=RuntimeError("start boom"))
    with (
        patch.object(cli, "load_settings", return_value=SimpleNamespace()),
        patch.object(cli, "build_llm_gateway", return_value=gw),
    ):
        rc = cli.main(["--mission", "go"])
    assert rc == 1
    gw.stop.assert_awaited_once()  # stop() guaranteed even if start() raises


def test_missing_mission_arg_exits_nonzero() -> None:
    cli = _load_cli()
    with pytest.raises(SystemExit) as exc:  # argparse error
        cli.main([])
    assert exc.value.code != 0
