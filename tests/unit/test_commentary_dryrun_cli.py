"""Unit tests for scripts/commentary_dryrun.py (operator offline probe).

Mirrors test_ask_rover_cli / test_translate_mission_cli: the script is loaded by
path via the shared loader, and load_settings / build_commentary_composer /
build_voice_engine are patched so no config, gateway, or speaker is needed.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests._script_loader import load_script_module


@pytest.fixture(scope="module")
def cli() -> ModuleType:
    return load_script_module("commentary_dryrun")


def _settings(*, enabled: bool = True, intensity: float = 0.6) -> SimpleNamespace:
    return SimpleNamespace(
        commentary=SimpleNamespace(enabled=enabled, excitement_intensity=intensity)
    )


def _composer(text: str = "lots of open room here") -> MagicMock:
    comp = MagicMock()
    comp.compose = AsyncMock(return_value=text)
    return comp


def test_happy_path_text_only_exits_zero(
    cli: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    comp = _composer("my power is getting low")
    with (
        patch.object(cli, "load_settings", return_value=_settings()),
        patch.object(cli, "build_commentary_composer", return_value=comp),
        patch.object(cli, "build_voice_engine", return_value=None),  # no speaker
    ):
        rc = cli.main(["--config", "x.yaml", "--battery-v", "10.5"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "my power is getting low" in out
    assert "samples=0" in out  # no voice -> text-only
    # Facts derived from CLI flags reached the composer.
    facts = comp.compose.call_args.args[0]
    assert facts.battery_v == 10.5


def test_speaks_through_voice_when_available(cli: ModuleType) -> None:
    comp = _composer("very loud noise around me")
    voice = AsyncMock()
    voice.play_phrase = AsyncMock(return_value=(2048, 0.7))
    with (
        patch.object(cli, "load_settings", return_value=_settings()),
        patch.object(cli, "build_commentary_composer", return_value=comp),
        patch.object(cli, "build_voice_engine", return_value=voice),
    ):
        rc = cli.main(["--config", "x.yaml", "--audio-rms", "0.5"])
    assert rc == 0
    voice.play_phrase.assert_awaited_once()
    voice.start.assert_awaited_once()
    voice.stop.assert_awaited_once()


def test_disabled_commentary_exits_2(cli: ModuleType) -> None:
    with patch.object(cli, "load_settings", return_value=_settings(enabled=False)):
        rc = cli.main(["--config", "x.yaml"])
    assert rc == 2


def test_config_load_error_exits_2(cli: ModuleType) -> None:
    with patch.object(cli, "load_settings", side_effect=FileNotFoundError("nope")):
        rc = cli.main(["--config", "x.yaml"])
    assert rc == 2


def test_no_composer_exits_1(cli: ModuleType) -> None:
    with (
        patch.object(cli, "load_settings", return_value=_settings()),
        patch.object(cli, "build_commentary_composer", return_value=None),
    ):
        rc = cli.main(["--config", "x.yaml"])
    assert rc == 1


def test_empty_composition_not_spoken(cli: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    comp = _composer("")  # nothing to say
    voice = AsyncMock()
    voice.play_phrase = AsyncMock(return_value=(0, 0.0))
    with (
        patch.object(cli, "load_settings", return_value=_settings()),
        patch.object(cli, "build_commentary_composer", return_value=comp),
        patch.object(cli, "build_voice_engine", return_value=voice),
    ):
        rc = cli.main(["--config", "x.yaml"])
    assert rc == 0
    voice.play_phrase.assert_not_awaited()  # empty -> never spoken
    assert "samples=0" in capsys.readouterr().out
