"""The boot hint a latched rover logs is a command that works (F-059 task 4.7).

``estop_latch_active_at_boot`` carries a ``rearm_cmd`` for the operator to copy. It used to
be the bare ``python -m mousedroid.cli.rearm --operator <you> --confirm-area-clear``, which
on the production container loaded ``default.yaml``, saw no latch and exited 0 -- the hint
that was supposed to fix the problem reproduced it. It now carries ``--config`` for every
overlay the process was *started with*, taken from ``Settings.overlay_paths`` (what
``load_settings`` actually merged), not from the environment: the two can differ, and the
hint's whole value is that it names the stack that located this latch.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import mousedroid.orchestrator._lifecycle_mixin as lifecycle_mod
from mousedroid.cli.rearm import _build_parser
from mousedroid.config.loader import load_settings
from mousedroid.config.schema import Settings
from mousedroid.factory.safety import build_emergency_latch
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

_LEGACY_HINT = "python -m mousedroid.cli.rearm --operator <you> --confirm-area-clear"
_EVENT = "estop_latch_active_at_boot"


class _Recorder:
    """Module-logger stand-in: records calls, independent of structlog's global state."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def __getattr__(self, level: str) -> Any:
        def _record(event: str, **kw: Any) -> None:
            self.events.append((level, event, kw))

        return _record


def _overlay(path: Path, *, experience: Path | None = None, arm: bool = True) -> Path:
    doc: dict[str, Any] = {}
    if experience is not None:
        doc["experience"] = {"path": str(experience)}
    if arm:
        doc["safety"] = {"emergency_latch": {"enabled": True, "fsync": False}}
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


async def _latched_orchestrator(cfg: Settings) -> MouseDroidOrchestrator:
    """An orchestrator over a root where a *previous run* left a persisted latch."""
    previous_run = build_emergency_latch(cfg)
    assert previous_run is not None, "the settings must arm the latch or this test is vacuous"
    previous_run.trip("lidar_emergency", causes=("lidar_emergency",))
    await previous_run.persist()
    return MouseDroidOrchestrator(
        world_model=MagicMock(),
        agents=[MagicMock()],
        safety_monitor=MagicMock(),
        esp32=AsyncMock(),
        sensor_manager=AsyncMock(),
        cfg=cfg,
        emergency_latch=build_emergency_latch(cfg),
    )


async def _boot_hint(orch: MouseDroidOrchestrator, monkeypatch: pytest.MonkeyPatch) -> str:
    log = _Recorder()
    monkeypatch.setattr(lifecycle_mod, "_log", log)
    await orch._load_emergency_latch()
    events = [kw for _, name, kw in log.events if name == _EVENT]
    assert len(events) == 1, f"expected exactly one {_EVENT} event, got {log.events}"
    hint = events[0]["rearm_cmd"]
    assert isinstance(hint, str)
    return hint


def _config_values(hint: str) -> list[str]:
    argv = shlex.split(hint)
    return [argv[i + 1] for i, arg in enumerate(argv) if arg == "--config"]


async def test_the_hint_names_the_overlay_the_process_was_started_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overlay = _overlay(tmp_path / "prod.yaml", experience=tmp_path / "exp")
    orch = await _latched_orchestrator(load_settings(overlay))

    hint = await _boot_hint(orch, monkeypatch)

    assert _config_values(hint) == [str(overlay.absolute())]
    assert hint.startswith("python -m mousedroid.cli.rearm --config ")
    assert hint.endswith(" --operator <you> --confirm-area-clear")


async def test_the_hint_names_every_overlay_in_the_order_they_were_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    locate = _overlay(tmp_path / "locate.yaml", experience=tmp_path / "exp", arm=False)
    arm = _overlay(tmp_path / "arm.yaml")
    orch = await _latched_orchestrator(load_settings(locate, arm))

    hint = await _boot_hint(orch, monkeypatch)

    assert _config_values(hint) == [str(locate.absolute()), str(arm.absolute())]


async def test_a_path_with_spaces_survives_a_shell_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator pastes the hint into a shell; an unquoted space would split the path."""
    spaced = tmp_path / "etc mousedroid"
    spaced.mkdir()
    overlay = _overlay(spaced / "jetson production.yaml", experience=tmp_path / "exp")
    orch = await _latched_orchestrator(load_settings(overlay))

    hint = await _boot_hint(orch, monkeypatch)

    assert _config_values(hint) == [str(overlay.absolute())]


async def test_without_any_overlay_the_hint_is_the_legacy_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Settings built directly (tests, embedders) have no provenance to name."""
    cfg = Settings(mock_hardware=True)
    cfg.experience.path = str(tmp_path)
    cfg.safety.emergency_latch.enabled = True
    cfg.safety.emergency_latch.fsync = False
    assert cfg.overlay_paths == ()
    orch = await _latched_orchestrator(cfg)

    assert await _boot_hint(orch, monkeypatch) == _LEGACY_HINT


async def test_the_hint_is_a_command_the_rearm_cli_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hint and the CLI are written in two modules; this is the contract between them."""
    overlay = _overlay(tmp_path / "prod.yaml", experience=tmp_path / "exp")
    orch = await _latched_orchestrator(load_settings(overlay))

    argv = shlex.split(await _boot_hint(orch, monkeypatch))

    assert argv[:3] == ["python", "-m", "mousedroid.cli.rearm"]
    args = _build_parser().parse_args(argv[3:])
    assert args.config == [overlay.absolute()]
    assert args.operator == "<you>"
    assert args.confirm_area_clear is True
    assert args.status is False


async def test_the_hint_survives_the_mock_hardware_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main --mock-hardware`` does ``cfg.model_copy(update=...)`` before building."""
    overlay = _overlay(tmp_path / "prod.yaml", experience=tmp_path / "exp")
    cfg = load_settings(overlay).model_copy(update={"mock_hardware": True})
    orch = await _latched_orchestrator(cfg)

    assert _config_values(await _boot_hint(orch, monkeypatch)) == [str(overlay.absolute())]
