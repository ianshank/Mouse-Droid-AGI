"""The re-arm CLI finds its overlay the way the rest of the runtime does (F-059 task 4.7).

Before this, ``python -m mousedroid.cli.rearm`` with no ``--config`` called
``load_settings()`` with no overlay. On the production container that loads
``default.yaml``, where the latch is off, so it printed "nothing to clear" and exited 0:
a silent no-op that looks exactly like a rover with nothing latched. The container does
set ``MOUSEDROID_CONFIG``, but nothing in the CLI read it.

These tests build a *real* latch under one experience root and then ask the CLI to look
for it, so they can only pass if the CLI loaded the overlay stack that names that root --
asserting on a mocked ``load_settings`` call would pass for a CLI that loaded the right
paths and then ignored them.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

import mousedroid.cli.rearm as rearm_mod
from mousedroid.cli.rearm import EXIT_OK, EXIT_STILL_LATCHED, _build_parser, main
from mousedroid.config.loader import load_settings
from mousedroid.validation.runtime import RUNTIME_CONFIG_ENV_VARS

_CSV_VARS = ("MOUSEDROID_CONFIGS", "MOUSEDROID_JETSON_CONFIGS")
_SINGLE_VARS = ("MOUSEDROID_CONFIG", "MOUSEDROID_JETSON_CONFIG")

_DISABLED_MESSAGE = (
    "safety.emergency_latch.enabled is false: no latch is in use, so there is nothing to clear."
)
_CLEAR_MESSAGE = "No emergency stop is latched."


class _Recorder:
    """Stand-in for the module logger: independent of structlog's global state.

    structlog caches bound loggers, so a suite that configured logging earlier can
    leave ``rearm._log`` writing to a chain ``capture_logs`` never sees. Recording at the
    module attribute asserts what the CLI *logged*, not what a global happened to route.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def _record(self, level: str, event: str, **kw: Any) -> None:
        self.events.append((level, event, kw))

    def debug(self, event: str, **kw: Any) -> None:
        self._record("debug", event, **kw)

    def info(self, event: str, **kw: Any) -> None:
        self._record("info", event, **kw)

    def warning(self, event: str, **kw: Any) -> None:
        self._record("warning", event, **kw)

    def error(self, event: str, **kw: Any) -> None:
        self._record("error", event, **kw)

    def exception(self, event: str, **kw: Any) -> None:
        self._record("exception", event, **kw)

    def named(self, event: str) -> list[dict[str, Any]]:
        return [kw for _, name, kw in self.events if name == event]


class _Streams:
    """Exactly what the CLI itself wrote -- no structlog lines from other modules.

    The unconfigured structlog default prints every level to stdout, so ``capsys`` mixes
    the loader's and factory's log lines (with timestamps) into the CLI's own message and
    exact-match assertions become order-dependent. Swapping the module's ``sys`` isolates
    the CLI's writes.
    """

    def __init__(self) -> None:
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    @property
    def out(self) -> str:
        return self.stdout.getvalue()

    @property
    def err(self) -> str:
        return self.stderr.getvalue()


@pytest.fixture
def log(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(rearm_mod, "_log", recorder)
    return recorder


@pytest.fixture
def streams(monkeypatch: pytest.MonkeyPatch) -> _Streams:
    captured = _Streams()
    monkeypatch.setattr(rearm_mod, "sys", captured)
    return captured


@pytest.fixture(autouse=True)
def _no_ambient_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    """The developer's shell must not decide which overlay the CLI loads."""
    for var in RUNTIME_CONFIG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _write(path: Path, doc: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


class _World:
    """Two experience roots; only ``right`` holds a tripped latch."""

    def __init__(self, root: Path) -> None:
        self.right_root = root / "exp_right"
        self.wrong_root = root / "exp_wrong"
        # The "right" stack is two files: neither alone locates AND arms the latch, so a
        # CLI that loaded only the first or last overlay finds nothing.
        self.right_base = _write(
            root / "right_base.yaml", {"experience": {"path": str(self.right_root)}}
        )
        self.right_arm = _write(
            root / "right_arm.yaml",
            {"safety": {"emergency_latch": {"enabled": True, "fsync": False}}},
        )
        self.right_stack = (self.right_base, self.right_arm)
        # Same latch settings, different root: a CLI that loaded this reports "clear".
        self.wrong = _write(
            root / "wrong.yaml",
            {
                "experience": {"path": str(self.wrong_root)},
                "safety": {"emergency_latch": {"enabled": True, "fsync": False}},
            },
        )
        self.latch_file = self._trip()

    def _trip(self) -> Path:
        from mousedroid.factory.safety import build_emergency_latch

        latch = build_emergency_latch(load_settings(*self.right_stack))
        assert latch is not None, "the right stack must arm the latch or this world is vacuous"
        latch.trip("battery_critical", causes=("battery_critical",))
        asyncio.run(latch.persist())
        path: Path = latch.path  # type: ignore[attr-defined]
        assert path.exists()
        return path


@pytest.fixture
def world(tmp_path: Path) -> _World:
    return _World(tmp_path)


def _csv(paths: Sequence[Path]) -> str:
    return ",".join(str(p) for p in paths)


# --- env fallback ---------------------------------------------------------------------


@pytest.mark.parametrize("var", _CSV_VARS)
def test_an_env_csv_stack_is_what_the_latch_is_built_from(
    world: _World,
    monkeypatch: pytest.MonkeyPatch,
    streams: _Streams,
    log: _Recorder,
    var: str,
) -> None:
    monkeypatch.setenv(var, _csv(world.right_stack))

    rc = main(["--status"])

    assert rc == EXIT_STILL_LATCHED
    assert "EMERGENCY STOP LATCHED" in streams.out
    assert "battery_critical" in streams.out
    (event,) = log.named("rearm_config_resolved")
    assert event["source"] == "env_csv"
    assert event["env_var"] == var
    assert event["paths"] == [str(p) for p in world.right_stack]


@pytest.mark.parametrize("var", _SINGLE_VARS)
def test_an_env_single_overlay_is_honoured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    streams: _Streams,
    log: _Recorder,
    var: str,
) -> None:
    """One file that both locates and arms the latch -- the production container's shape."""
    overlay = _write(
        tmp_path / "prod.yaml",
        {
            "experience": {"path": str(tmp_path / "exp")},
            "safety": {"emergency_latch": {"enabled": True, "fsync": False}},
        },
    )
    from mousedroid.factory.safety import build_emergency_latch

    latch = build_emergency_latch(load_settings(overlay))
    assert latch is not None
    latch.trip("sensor_stale", causes=("sensor_stale",))
    asyncio.run(latch.persist())
    monkeypatch.setenv(var, str(overlay))

    rc = main(["--status"])

    assert rc == EXIT_STILL_LATCHED
    assert "EMERGENCY STOP LATCHED" in streams.out
    (event,) = log.named("rearm_config_resolved")
    assert (event["source"], event["env_var"], event["paths"]) == (
        "env_single",
        var,
        [str(overlay)],
    )


def test_the_env_fallback_can_clear_the_latch(
    world: _World, monkeypatch: pytest.MonkeyPatch, streams: _Streams
) -> None:
    """Not just ``--status``: the same resolution applies to the clearing path."""
    monkeypatch.setenv("MOUSEDROID_JETSON_CONFIGS", _csv(world.right_stack))

    refused = main(["--operator", "tester"])
    assert refused == EXIT_STILL_LATCHED
    assert world.latch_file.exists(), "a missing --confirm-area-clear must not clear"

    cleared = main(["--operator", "tester", "--confirm-area-clear"])

    assert cleared == EXIT_OK
    assert not world.latch_file.exists()
    assert "Cleared by tester" in streams.out


# --- precedence ------------------------------------------------------------------------


def test_the_cli_flag_beats_the_environment(
    world: _World, monkeypatch: pytest.MonkeyPatch, streams: _Streams, log: _Recorder
) -> None:
    """Env names the wrong root (it would say 'clear'); the flag names the right one."""
    monkeypatch.setenv("MOUSEDROID_JETSON_CONFIGS", str(world.wrong))
    monkeypatch.setenv("MOUSEDROID_CONFIG", str(world.wrong))

    flags = [arg for p in world.right_stack for arg in ("--config", str(p))]
    rc = main([*flags, "--status"])

    assert rc == EXIT_STILL_LATCHED, "the environment overrode an explicit --config"
    assert "EMERGENCY STOP LATCHED" in streams.out
    (event,) = log.named("rearm_config_resolved")
    assert event["source"] == "cli"
    assert event["env_var"] is None


def test_the_cli_flag_beats_the_environment_in_the_other_direction(
    world: _World, monkeypatch: pytest.MonkeyPatch, streams: _Streams
) -> None:
    """Env names the right stack (it would say LATCHED); the flag names the wrong root."""
    monkeypatch.setenv("MOUSEDROID_JETSON_CONFIGS", _csv(world.right_stack))

    rc = main(["--config", str(world.wrong), "--status"])

    assert rc == EXIT_OK, "the environment overrode an explicit --config"
    assert _CLEAR_MESSAGE in streams.out


def test_an_env_csv_list_outranks_an_env_single_path(
    world: _World, monkeypatch: pytest.MonkeyPatch, streams: _Streams, log: _Recorder
) -> None:
    """The runtime's own order: lists first, then the single path."""
    monkeypatch.setenv("MOUSEDROID_JETSON_CONFIGS", _csv(world.right_stack))
    monkeypatch.setenv("MOUSEDROID_CONFIG", str(world.wrong))

    rc = main(["--status"])

    assert rc == EXIT_STILL_LATCHED
    assert "EMERGENCY STOP LATCHED" in streams.out
    assert log.named("rearm_config_resolved")[0]["source"] == "env_csv"


# --- nothing supplied ------------------------------------------------------------------


def test_with_no_flag_and_no_env_the_behaviour_is_unchanged_but_it_says_why(
    streams: _Streams, log: _Recorder
) -> None:
    rc = main(["--status"])

    assert rc == EXIT_OK
    assert streams.out == _DISABLED_MESSAGE + "\n"
    (event,) = log.named("rearm_config_resolved")
    assert (event["source"], event["env_var"], event["paths"]) == ("default", None, [])

    note_lines = streams.err.splitlines()
    assert len(note_lines) == 1, f"exactly one line, not a wall of text: {streams.err!r}"
    note = note_lines[0]
    assert "no config overlay" in note
    assert "--config" in note
    for var in RUNTIME_CONFIG_ENV_VARS:
        assert var in note, f"the note must name every var the resolver reads, missing {var}"


def test_the_note_is_not_printed_when_an_overlay_was_supplied(
    world: _World, streams: _Streams
) -> None:
    main(["--config", str(world.wrong), "--status"])

    assert streams.err == ""


# --- argparse --------------------------------------------------------------------------


class _Spy:
    """Records the overlay paths ``load_settings`` receives and returns real settings."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, ...]] = []

    def __call__(self, *paths: Path) -> Any:
        self.calls.append(paths)
        return load_settings()


def test_config_accepts_several_files_in_one_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _Spy()
    monkeypatch.setattr(rearm_mod, "load_settings", spy)

    main(["--config", "a.yaml", "b.yaml", "--status"])

    assert spy.calls == [(Path("a.yaml"), Path("b.yaml"))]


def test_a_repeated_config_flag_extends_rather_than_replaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The silent-drop trap: with plain ``nargs`` the last ``--config`` would win."""
    spy = _Spy()
    monkeypatch.setattr(rearm_mod, "load_settings", spy)

    main(["--config", "a.yaml", "--config", "b.yaml", "c.yaml", "--status"])

    assert spy.calls == [(Path("a.yaml"), Path("b.yaml"), Path("c.yaml"))]


def test_a_single_config_still_means_exactly_that_file(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _Spy()
    monkeypatch.setattr(rearm_mod, "load_settings", spy)

    main(["--config", "only.yaml", "--status"])

    assert spy.calls == [(Path("only.yaml"),)]


def test_the_parser_defaults_are_unchanged() -> None:
    args = _build_parser().parse_args([])

    assert args.config is None
    assert args.operator is None
    assert args.confirm_area_clear is False
    assert args.status is False


def test_the_help_text_names_the_env_fallback() -> None:
    parser = _build_parser()
    config_action = next(a for a in parser._actions if "--config" in a.option_strings)

    help_text = config_action.help or ""
    for var in RUNTIME_CONFIG_ENV_VARS:
        assert var in help_text


# --- failing loud ----------------------------------------------------------------------


def test_an_env_path_that_does_not_exist_fails_loudly_not_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: _Recorder
) -> None:
    """A stale MOUSEDROID_CONFIG must not degrade back into 'nothing to clear'."""
    monkeypatch.setenv("MOUSEDROID_CONFIG", str(tmp_path / "gone.yaml"))

    rc = main(["--status"])

    assert rc == EXIT_STILL_LATCHED
    assert log.named("rearm_failed"), "the failure must be logged, with its traceback"
