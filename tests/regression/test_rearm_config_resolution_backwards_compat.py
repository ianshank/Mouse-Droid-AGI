"""Backwards-compat pin -- F-059 task 4.7: what must NOT change when the CLI learns the env.

The re-arm CLI is the one thing between a latched rover and motion, so everything it already
did stays byte-for-byte: the exit codes, every message, the two flags that gate a clear, and
what ``--config X`` means. Only the *absence* of ``--config`` changes meaning, and even there
the stdout message and exit code are unchanged -- the new behaviour is one extra line on
stderr and a log event.

Equally pinned: ``Settings`` gains no schema field (existing YAML loads unchanged, invariant
6), the shared resolver's precedence table is the one that shipped, and a latched boot
without any recorded overlay still logs the exact hint it always did.
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

import mousedroid.cli.rearm as rearm_mod
from mousedroid.cli.rearm import EXIT_OK, EXIT_STILL_LATCHED, main
from mousedroid.config.loader import load_settings
from mousedroid.config.schema import Settings
from mousedroid.validation.runtime import resolve_runtime_config_paths

_REPO_ROOT = Path(__file__).resolve().parents[2]

# The four names the resolver has always read. Literal on purpose: this file pins what
# shipped, so it must not follow the shared constant when the constant changes.
_ENV_VARS = (
    "MOUSEDROID_CONFIGS",
    "MOUSEDROID_JETSON_CONFIGS",
    "MOUSEDROID_CONFIG",
    "MOUSEDROID_JETSON_CONFIG",
)

# Copied from the pre-4.7 source, deliberately not imported: a pin that reads the constant
# it protects cannot notice the constant changing.
_DISABLED = (
    "safety.emergency_latch.enabled is false: no latch is in use, so there is nothing to clear.\n"
)
_CLEAR = "No emergency stop is latched.\n"
_REFUSED_TAIL = "Inspect the rover and its path first, then re-run with both.\n"
_CLEARED = (
    "\nCleared by tester. Restart the service to arm the rover; "
    "clearing the latch does not by itself command motion.\n"
)
_LEGACY_HINT = "python -m mousedroid.cli.rearm --operator <you> --confirm-area-clear"


class _Streams:
    def __init__(self) -> None:
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()


@pytest.fixture
def streams(monkeypatch: pytest.MonkeyPatch) -> _Streams:
    """Only what the CLI itself wrote (other modules' structlog lines are excluded)."""
    captured = _Streams()
    monkeypatch.setattr(rearm_mod, "sys", captured)
    return captured


@pytest.fixture(autouse=True)
def _no_ambient_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _latched_overlay(tmp_path: Path) -> tuple[Path, Path]:
    from mousedroid.factory.safety import build_emergency_latch

    overlay = tmp_path / "prod.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "experience": {"path": str(tmp_path / "exp")},
                "safety": {"emergency_latch": {"enabled": True, "fsync": False}},
            }
        ),
        encoding="utf-8",
    )
    latch = build_emergency_latch(load_settings(overlay))
    assert latch is not None
    latch.trip("battery_critical", causes=("battery_critical",))
    asyncio.run(latch.persist())
    return overlay, latch.path  # type: ignore[attr-defined]


def test_the_exit_codes_are_the_two_documented_ones() -> None:
    assert (EXIT_OK, EXIT_STILL_LATCHED) == (0, 1)


def test_no_flag_and_no_env_prints_the_same_message_and_exits_the_same(
    streams: _Streams,
) -> None:
    """Today's behaviour kept: it is still 'nothing to clear', exit 0, on stdout."""
    rc = main(["--status"])

    assert rc == 0
    assert streams.stdout.getvalue() == _DISABLED


def test_the_added_diagnostic_is_on_stderr_so_stdout_scrapers_see_no_change(
    streams: _Streams,
) -> None:
    main(["--status"])

    assert streams.stdout.getvalue() == _DISABLED
    assert streams.stderr.getvalue().count("\n") == 1


def test_an_explicit_config_prints_every_message_byte_for_byte(
    tmp_path: Path, streams: _Streams
) -> None:
    overlay, latch_file = _latched_overlay(tmp_path)
    iso = json.loads(latch_file.read_text(encoding="utf-8"))["tripped_at_iso"]
    banner = (
        "EMERGENCY STOP LATCHED\n"
        "  reason:     battery_critical\n"
        "  causes:     battery_critical\n"
        f"  tripped at: {iso}\n"
    )
    config = ["--config", str(overlay)]

    # --status: report only.
    assert main([*config, "--status"]) == EXIT_STILL_LATCHED
    assert streams.stdout.getvalue() == banner
    assert latch_file.exists()

    # Both gate flags missing.
    streams.stdout.seek(0)
    streams.stdout.truncate()
    assert main(config) == EXIT_STILL_LATCHED
    assert streams.stdout.getvalue() == (
        banner
        + "\nRefusing to clear: --operator and --confirm-area-clear required.\n"
        + _REFUSED_TAIL
    )

    # Only one missing -- the message names exactly the missing one.
    streams.stdout.seek(0)
    streams.stdout.truncate()
    assert main([*config, "--operator", "tester"]) == EXIT_STILL_LATCHED
    assert streams.stdout.getvalue() == (
        banner + "\nRefusing to clear: --confirm-area-clear required.\n" + _REFUSED_TAIL
    )
    assert latch_file.exists()

    # Cleared.
    streams.stdout.seek(0)
    streams.stdout.truncate()
    assert main([*config, "--operator", "tester", "--confirm-area-clear"]) == EXIT_OK
    assert streams.stdout.getvalue() == banner + _CLEARED
    assert not latch_file.exists()

    # And now it reads clear.
    streams.stdout.seek(0)
    streams.stdout.truncate()
    assert main([*config, "--status"]) == EXIT_OK
    assert streams.stdout.getvalue() == _CLEAR
    assert streams.stderr.getvalue() == "", "an explicit --config must add no diagnostic at all"


def test_a_single_config_still_loads_exactly_that_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--config X`` used to be a single ``Path``; it must still mean 'load X', no more."""
    loaded: list[tuple[Path, ...]] = []

    def _spy(*paths: Path) -> Any:
        loaded.append(paths)
        return load_settings()

    monkeypatch.setattr(rearm_mod, "load_settings", _spy)
    monkeypatch.setenv("MOUSEDROID_CONFIG", "/etc/mousedroid/somewhere-else.yaml")

    main(["--config", "/etc/mousedroid/jetson_production.yaml", "--status"])

    assert loaded == [(Path("/etc/mousedroid/jetson_production.yaml"),)]


@pytest.mark.parametrize(
    "argv",
    [
        ["--status"],
        ["--operator", "x", "--confirm-area-clear"],
        ["--config", "c.yaml", "--status"],
        ["--config", "c.yaml", "--operator", "x", "--confirm-area-clear"],
    ],
)
def test_every_pre_existing_invocation_still_parses(argv: list[str]) -> None:
    rearm_mod._build_parser().parse_args(argv)


def test_the_gate_flags_still_have_no_short_form_and_no_default() -> None:
    """'No short form and no default, deliberately' -- an unattributed re-arm is not one."""
    parser = rearm_mod._build_parser()
    by_dest = {a.dest: a for a in parser._actions}

    assert by_dest["confirm_area_clear"].option_strings == ["--confirm-area-clear"]
    assert by_dest["confirm_area_clear"].default is False
    assert by_dest["operator"].option_strings == ["--operator"]
    assert by_dest["operator"].default is None


# --- the boot hint without a recorded overlay -------------------------------------------


async def test_a_latched_boot_with_no_recorded_overlay_logs_the_exact_legacy_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import AsyncMock, MagicMock

    import mousedroid.orchestrator._lifecycle_mixin as lifecycle_mod
    from mousedroid.factory.safety import build_emergency_latch
    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    cfg = Settings(mock_hardware=True)
    cfg.experience.path = str(tmp_path)
    cfg.safety.emergency_latch.enabled = True
    cfg.safety.emergency_latch.fsync = False
    previous = build_emergency_latch(cfg)
    assert previous is not None
    previous.trip("sensor_stale", causes=("sensor_stale",))
    await previous.persist()
    orch = MouseDroidOrchestrator(
        world_model=MagicMock(),
        agents=[MagicMock()],
        safety_monitor=MagicMock(),
        esp32=AsyncMock(),
        sensor_manager=AsyncMock(),
        cfg=cfg,
        emergency_latch=build_emergency_latch(cfg),
    )
    hints: list[object] = []

    class _Log:
        def __getattr__(self, _level: str) -> Any:
            def _record(event: str, **kw: Any) -> None:
                if event == "estop_latch_active_at_boot":
                    hints.append(kw["rearm_cmd"])

            return _record

    monkeypatch.setattr(lifecycle_mod, "_log", _Log())

    await orch._load_emergency_latch()

    assert hints == [_LEGACY_HINT]


# --- Settings gains provenance, not a schema field -----------------------------------------


def test_settings_gains_no_schema_field() -> None:
    """Invariant 6: a new field is a YAML-visible surface; provenance must not be one."""
    assert "overlay_paths" not in Settings.model_fields
    assert not any("overlay" in name for name in Settings.model_fields)


def test_a_yaml_key_cannot_forge_the_provenance() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({"mock_hardware": True, "overlay_paths": ["/etc/other.yaml"]})


def test_the_default_provenance_is_empty() -> None:
    assert Settings(mock_hardware=True).overlay_paths == ()
    assert load_settings().overlay_paths == ()


def test_the_shipped_production_overlay_loads_and_names_itself() -> None:
    production = _REPO_ROOT / "config" / "jetson_production.yaml"

    cfg = load_settings(production)

    assert cfg.overlay_paths == (production.absolute(),)


# --- the shared resolver's precedence table is the one that shipped -------------------------


@pytest.mark.parametrize(
    ("explicit", "env", "expected"),
    [
        (None, {}, ()),
        (["x.yaml"], {"MOUSEDROID_CONFIG": "e.yaml"}, ("x.yaml",)),
        (None, {"MOUSEDROID_CONFIG": "e.yaml"}, ("e.yaml",)),
        (None, {"MOUSEDROID_JETSON_CONFIG": "e.yaml"}, ("e.yaml",)),
        (None, {"MOUSEDROID_CONFIGS": "a.yaml, b.yaml"}, ("a.yaml", "b.yaml")),
        (None, {"MOUSEDROID_JETSON_CONFIGS": "a.yaml,b.yaml"}, ("a.yaml", "b.yaml")),
        (
            None,
            {"MOUSEDROID_CONFIGS": "list.yaml", "MOUSEDROID_CONFIG": "single.yaml"},
            ("list.yaml",),
        ),
        (
            None,
            {"MOUSEDROID_CONFIGS": "l1.yaml", "MOUSEDROID_JETSON_CONFIGS": "l2.yaml"},
            ("l1.yaml",),
        ),
        (
            None,
            {"MOUSEDROID_CONFIG": "s1.yaml", "MOUSEDROID_JETSON_CONFIG": "s2.yaml"},
            ("s1.yaml",),
        ),
        (None, {"MOUSEDROID_CONFIGS": " , ", "MOUSEDROID_CONFIG": "s.yaml"}, ("s.yaml",)),
    ],
)
def test_the_resolver_precedence_table_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    explicit: list[str] | None,
    env: dict[str, str],
    expected: tuple[str, ...],
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert resolve_runtime_config_paths(explicit) == tuple(Path(p) for p in expected)
