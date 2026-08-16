"""Unit tests for scripts/translate_mission.py (the operator dry-run probe).

The probe loads Settings, builds the LLM gateway via the factory, translates a
single NL mission, prints the resulting GoalVector + which tier served, and
exits. The gateway is mocked end-to-end so no network, API key, or GGUF is
needed. Mirrors the test style of the greeting CLI.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mousedroid.llm_gateway.protocol import GoalVector
from tests._script_loader import load_script_module


@pytest.fixture(scope="module")
def cli() -> ModuleType:
    """Load the script module by path once per test module (it lives in scripts/)."""
    return load_script_module("translate_mission")


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
    vector: GoalVector,
    *,
    primary_degraded: bool,
    both_degraded: bool = False,
    stop_clears_degraded: bool = False,
) -> MagicMock:
    """A composite gateway exposing ``_primary`` (mirrors FallbackLLMGateway).

    When ``stop_clears_degraded`` is set, ``stop()`` resets the degraded flags —
    mirroring the real ``AnthropicLLMGateway.stop()``, which clears ``_degraded``.
    This lets a test prove the probe captures the serving tier BEFORE ``stop()``.
    """
    gw = MagicMock(
        spec=["is_ready", "is_degraded", "start", "stop", "translate_mission", "_primary"]
    )
    primary = SimpleNamespace(is_degraded=primary_degraded)
    gw.is_ready = True
    gw.is_degraded = both_degraded
    gw._primary = primary
    gw.start = AsyncMock(return_value=None)
    gw.translate_mission = AsyncMock(return_value=vector)

    async def _stop() -> None:
        if stop_clears_degraded:
            primary.is_degraded = False
            gw.is_degraded = False

    gw.stop = AsyncMock(side_effect=_stop)
    return gw


# --------------------------------------------------------------------------- #
# Happy path + output formatting
# --------------------------------------------------------------------------- #
def test_translate_prints_goalvector_and_exits_zero(
    cli: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """Happy path: prints the GoalVector, reports tier=primary, exits 0."""
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


def test_single_gateway_degraded_tier_reported(
    cli: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """A degraded single gateway reports tier=degraded."""
    gw = _fake_gateway(GoalVector(), degraded=True)
    with (
        patch.object(cli, "load_settings", return_value=SimpleNamespace()),
        patch.object(cli, "build_llm_gateway", return_value=gw),
    ):
        rc = cli.main(["--mission", "stop"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tier=degraded" in out


def test_composite_secondary_tier_reported(
    cli: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """When the primary is degraded, the composite serves via the secondary."""
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


def test_composite_tier_captured_before_stop_clears_degraded(
    cli: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression (Copilot finding): the serving tier must be read BEFORE stop().

    The real AnthropicLLMGateway.stop() clears its degraded flag. If the probe
    described the tier AFTER stop(), it would misreport a secondary-served call
    as 'primary'. This double clears the degraded flags in stop().
    """
    gw = _fake_composite(
        GoalVector(vx_target=0.2), primary_degraded=True, stop_clears_degraded=True
    )
    with (
        patch.object(cli, "load_settings", return_value=SimpleNamespace()),
        patch.object(cli, "build_llm_gateway", return_value=gw),
    ):
        rc = cli.main(["--mission", "hold position"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "secondary" in out  # captured before stop() cleared the degraded flag
    gw.stop.assert_awaited_once()


def test_composite_primary_tier_reported(
    cli: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """A usable primary in the composite reports tier=primary."""
    gw = _fake_composite(GoalVector(vx_target=0.3), primary_degraded=False)
    with (
        patch.object(cli, "load_settings", return_value=SimpleNamespace()),
        patch.object(cli, "build_llm_gateway", return_value=gw),
    ):
        rc = cli.main(["--mission", "go forward"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tier=primary" in out


def test_composite_both_degraded_tier_reported(
    cli: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both tiers degraded must be surfaced in the output."""
    gw = _fake_composite(GoalVector(), primary_degraded=True, both_degraded=True)
    with (
        patch.object(cli, "load_settings", return_value=SimpleNamespace()),
        patch.object(cli, "build_llm_gateway", return_value=gw),
    ):
        rc = cli.main(["--mission", "stop"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "degraded" in out


# --------------------------------------------------------------------------- #
# Config resolution (the production gap: MOUSEDROID_CONFIG must be honored)
# --------------------------------------------------------------------------- #
def test_explicit_config_is_passed_to_load_settings(cli: ModuleType) -> None:
    """An explicit --config overlay is forwarded to load_settings."""
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


def test_env_config_resolved_when_no_cli_flag(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MOUSEDROID_CONFIG must be resolved when --config is omitted (the bug this fixes)."""
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
def test_config_load_error_exits_2(cli: ModuleType) -> None:
    """A config-load failure exits 2 (config error)."""
    with patch.object(cli, "load_settings", side_effect=FileNotFoundError("no such file")):
        rc = cli.main(["--mission", "go"])
    assert rc == 2


def test_build_error_exits_1(cli: ModuleType) -> None:
    """A gateway-build failure exits 1 (runtime error)."""
    with (
        patch.object(cli, "load_settings", return_value=SimpleNamespace()),
        patch.object(cli, "build_llm_gateway", side_effect=RuntimeError("build failed")),
    ):
        rc = cli.main(["--mission", "go"])
    assert rc == 1


def test_runtime_error_exits_1_and_still_stops(cli: ModuleType) -> None:
    """A translate failure exits 1 and still calls stop() (cleanup guarantee)."""
    gw = _fake_gateway(GoalVector())
    gw.translate_mission = AsyncMock(side_effect=RuntimeError("network down"))
    with (
        patch.object(cli, "load_settings", return_value=SimpleNamespace()),
        patch.object(cli, "build_llm_gateway", return_value=gw),
    ):
        rc = cli.main(["--mission", "go"])
    assert rc == 1
    gw.stop.assert_awaited_once()  # cleanup runs even when translate raises


def test_start_failure_still_stops(cli: ModuleType) -> None:
    """A start() failure exits 1 and still calls stop() (cleanup guarantee)."""
    gw = _fake_gateway(GoalVector())
    gw.start = AsyncMock(side_effect=RuntimeError("start boom"))
    with (
        patch.object(cli, "load_settings", return_value=SimpleNamespace()),
        patch.object(cli, "build_llm_gateway", return_value=gw),
    ):
        rc = cli.main(["--mission", "go"])
    assert rc == 1
    gw.stop.assert_awaited_once()  # stop() guaranteed even if start() raises


def test_missing_mission_arg_exits_nonzero(cli: ModuleType) -> None:
    """Missing --mission is an argparse error (non-zero exit)."""
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code != 0
