"""AQA -- F-059 task 4.7: the re-arm CLI and the boot hint cannot disagree with the runtime.

Ruling 36 (found reviewing PR-D): ``python -m mousedroid.cli.rearm`` with no ``--config``
called ``load_settings()`` with no overlay. On the production container that loads
``default.yaml``, where the latch is off, so it printed "nothing to clear" and exited 0 --
a silent no-op indistinguishable from a rover with nothing latched. Nothing in the CLI read
``MOUSEDROID_CONFIG``, which compose does set.

The fix is deliberately *not* a new ``os.getenv`` in the CLI. ``AGENTS.md`` names that as a
hidden configuration source, and it would be a third, divergent opinion about which env
vars name the overlay stack. The CLI routes through
``validation.runtime.resolve_runtime_config`` -- the resolver ``validation.runtime`` already
provides to the smoke and probe scripts -- and these pins hold that line:

* the CLI reads no environment of its own, and its resolver *is* the shared one;
* every env var the runtime honours reaches the CLI, including any added later (the
  parametrisation is derived from the shared constant, not copied from it);
* the CLI loads exactly the overlays the runtime resolver returns, for a precedence matrix;
* the boot hint is built from ``Settings.overlay_paths`` (what was loaded), never the
  environment, so it still names the right overlay when the two differ;
* the boot hint is the ``--status`` form and **never carries the attestation** (Ruling 44):
  ``--confirm-area-clear`` is the operator's statement that the area is physically clear,
  so a logged command that already spells it out turns "paste" into "clear". The flag may
  appear in ``_lifecycle_mixin.py`` only inside the prose constant logged under a separate
  key;
* every documented re-arm command carries ``--config``. The env fallback is a safety net,
  not the documented form.
"""

from __future__ import annotations

import ast
import asyncio
import re
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

import mousedroid.cli.rearm as rearm_mod
from mousedroid.cli.rearm import EXIT_STILL_LATCHED, main
from mousedroid.config.loader import load_settings
from mousedroid.validation.runtime import (
    RUNTIME_CONFIG_ENV_VARS,
    resolve_runtime_config,
    resolve_runtime_config_paths,
)
from mousedroid.validation.runtime import _shared as shared_mod

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src" / "mousedroid"
_RUNBOOK = _REPO_ROOT / "docs" / "runbooks" / "emergency-stop-rearm.md"
_COMPOSE = _REPO_ROOT / "docker-compose.jetson.yml"


@pytest.fixture(autouse=True)
def _no_ambient_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in RUNTIME_CONFIG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _reads_the_environment(source_path: Path) -> list[str]:
    """Every ``os.getenv`` / ``os.environ`` / ``from os import ...`` the module contains."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"getenv", "environ", "environb"}:
            found.append(f"line {node.lineno}: .{node.attr}")
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            found.append(f"line {node.lineno}: from os import ...")
        elif isinstance(node, ast.Import) and any(alias.name == "os" for alias in node.names):
            found.append(f"line {node.lineno}: import os")
    return found


# --- one resolver, no second opinion ---------------------------------------------------


def test_the_rearm_cli_reads_no_environment_itself() -> None:
    """AGENTS.md: an ``os.getenv`` deep in a module is a hidden configuration source."""
    offenders = _reads_the_environment(_SRC / "cli" / "rearm.py")

    assert offenders == [], f"cli/rearm.py reads the environment directly: {offenders}"


def test_the_boot_hint_reads_no_environment_itself() -> None:
    offenders = _reads_the_environment(_SRC / "orchestrator" / "_lifecycle_mixin.py")

    assert offenders == [], f"the hint builder reads the environment: {offenders}"


def test_the_rearm_cli_uses_the_shared_resolver_not_a_copy() -> None:
    assert rearm_mod.resolve_runtime_config is shared_mod.resolve_runtime_config
    assert resolve_runtime_config is shared_mod.resolve_runtime_config


def test_the_two_public_entry_points_share_one_precedence() -> None:
    """``resolve_runtime_config_paths`` is defined as the resolution's paths, not re-derived."""
    source = (_SRC / "validation" / "runtime" / "_shared.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    func = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "resolve_runtime_config_paths"
    )
    calls = {
        n.func.id
        for n in ast.walk(func)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "resolve_runtime_config" in calls, (
        "resolve_runtime_config_paths must delegate to resolve_runtime_config; two copies of "
        "the precedence are how the CLI and the runtime would come to disagree"
    )


# --- every var the runtime honours reaches the CLI -------------------------------------


def _tripped_overlay(tmp_path: Path) -> Path:
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
    latch.trip("human_proximity", causes=("human_proximity",))
    asyncio.run(latch.persist())
    return overlay


@pytest.mark.parametrize("var", RUNTIME_CONFIG_ENV_VARS)
def test_every_env_var_the_runtime_honours_reaches_the_rearm_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, var: str
) -> None:
    """Derived from the shared constant: a var added to the resolver is covered for free."""
    overlay = _tripped_overlay(tmp_path)
    monkeypatch.setenv(var, str(overlay))

    assert main(["--status"]) == EXIT_STILL_LATCHED, (
        f"{var} is honoured by the runtime resolver but the re-arm CLI ignored it: "
        "a latched rover would read as clear"
    )


_PRECEDENCE_MATRIX: list[dict[str, str]] = [
    {},
    {"MOUSEDROID_CONFIG": "single.yaml"},
    {"MOUSEDROID_JETSON_CONFIG": "legacy-single.yaml"},
    {"MOUSEDROID_CONFIG": "single.yaml", "MOUSEDROID_JETSON_CONFIG": "legacy-single.yaml"},
    {"MOUSEDROID_CONFIGS": "a.yaml,b.yaml"},
    {"MOUSEDROID_JETSON_CONFIGS": "c.yaml, d.yaml"},
    {"MOUSEDROID_CONFIGS": "a.yaml", "MOUSEDROID_JETSON_CONFIGS": "c.yaml"},
    {"MOUSEDROID_JETSON_CONFIGS": "c.yaml", "MOUSEDROID_CONFIG": "single.yaml"},
    {"MOUSEDROID_CONFIGS": " , ", "MOUSEDROID_CONFIG": "single.yaml"},
]


@pytest.mark.parametrize("env", _PRECEDENCE_MATRIX, ids=lambda e: "+".join(sorted(e)) or "empty")
@pytest.mark.parametrize("flags", [[], ["--config", "x.yaml"], ["--config", "x.yaml", "y.yaml"]])
def test_the_cli_loads_exactly_the_overlays_the_runtime_resolver_returns(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], flags: list[str]
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    loaded: list[tuple[Path, ...]] = []

    def _spy(*paths: Path) -> Any:
        loaded.append(paths)
        return load_settings()

    monkeypatch.setattr(rearm_mod, "load_settings", _spy)

    main([*flags, "--status"])

    cli_paths = tuple(Path(p) for p in flags[1:])
    assert loaded == [resolve_runtime_config_paths(cli_paths or None)]


# --- the hint comes from what was loaded, not from the environment ----------------------


async def _boot_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Boot a latched orchestrator over a freshly loaded overlay; return the hint event.

    The overlay is ``tmp_path / "jetson_production.yaml"``; the returned dict is the
    ``estop_latch_active_at_boot`` event's fields (``rearm_cmd``, ``clear_requires``, ...).
    """
    from unittest.mock import AsyncMock, MagicMock

    import mousedroid.orchestrator._lifecycle_mixin as lifecycle_mod
    from mousedroid.factory.safety import build_emergency_latch
    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    overlay = tmp_path / "jetson_production.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "experience": {"path": str(tmp_path / "exp")},
                "safety": {"emergency_latch": {"enabled": True, "fsync": False}},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_settings(overlay)
    previous = build_emergency_latch(cfg)
    assert previous is not None
    previous.trip("loop_overrun", causes=("loop_overrun",))
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
    events: list[dict[str, Any]] = []

    class _Log:
        def __getattr__(self, _level: str) -> Any:
            def _record(event: str, **kw: Any) -> None:
                if event == "estop_latch_active_at_boot":
                    events.append(kw)

            return _record

    monkeypatch.setattr(lifecycle_mod, "_log", _Log())

    await orch._load_emergency_latch()

    (event,) = events
    return event


async def test_the_boot_hint_names_the_loaded_overlay_even_when_the_environment_disagrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dockerfile.jetson bakes ``MOUSEDROID_CONFIG=.../default.yaml``; compose overrides it.

    A container run with only ``command: --config <overlay>`` would carry the *image's*
    env, so a hint built from the environment would name ``default.yaml`` -- the exact
    wrong file this task exists to stop pointing operators at.
    """
    monkeypatch.setenv("MOUSEDROID_CONFIG", "/etc/mousedroid/default.yaml")

    event = await _boot_event(tmp_path, monkeypatch)

    tokens = shlex.split(event["rearm_cmd"])
    assert tokens[tokens.index("--config") + 1] == str(
        (tmp_path / "jetson_production.yaml").absolute()
    )
    assert "/etc/mousedroid/default.yaml" not in event["rearm_cmd"]


async def test_the_boot_hint_inspects_and_never_pre_attests_the_area_is_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ruling 44: the logged command is ``--status``; it must not carry the attestation.

    ``--confirm-area-clear`` is the operator's own statement that the area is physically
    clear. A hint that already spells it out (plus ``--operator``) is one placeholder
    substitution away from a clear, so pasting from the log would attest on the operator's
    behalf. Clearing stays a typed act; the hint only tells the operator where to look.
    """
    event = await _boot_event(tmp_path, monkeypatch)

    hint = event["rearm_cmd"]
    assert "--status" in shlex.split(hint)
    assert "--confirm-area-clear" not in hint
    assert "--operator" not in hint
    # ...and the operator is still told what clearing takes, as prose under its own key.
    assert "--operator <you> --confirm-area-clear" in event["clear_requires"]


def _string_constants_naming_the_attestation(tree: ast.Module) -> dict[str, int]:
    """Top-level statement -> how many runtime string literals under it spell the flag.

    Docstrings are documentation, not strings the program can log, so they are excluded;
    comments are not in the AST at all.
    """
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    found: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            names = [f"<{type(node).__name__}:{getattr(node, 'name', '?')}>"]
        hits = sum(
            1
            for sub in ast.walk(node)
            if isinstance(sub, ast.Constant)
            and isinstance(sub.value, str)
            and id(sub) not in docstrings
            and "--confirm-area-clear" in sub.value
        )
        if hits:
            found[",".join(names)] = hits
    return found


def test_the_attestation_flag_appears_in_the_hint_module_only_as_the_prose_constant() -> None:
    """A second code path that re-adds the flag to a command would not trip a behavioural
    test that only boots one orchestrator; this pins where the literal may live at all."""
    path = _SRC / "orchestrator" / "_lifecycle_mixin.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    assert _string_constants_naming_the_attestation(tree) == {"_REARM_CLEAR_REQUIRES": 1}, (
        "--confirm-area-clear may be spelled in _lifecycle_mixin.py only inside the prose "
        "constant logged as `clear_requires`; anywhere else it risks a paste-to-clear hint"
    )


# --- the documented form is explicit; the fallback is a safety net -----------------------

_REARM_INVOCATION = re.compile(r"mousedroid\.cli\.rearm([^`\n]*)")


def _documented_invocations() -> list[str]:
    """Arguments of every ``mousedroid.cli.rearm`` command in the runbook.

    Shell continuations are joined first, and the boot-log excerpt (a line carrying
    ``rearm_cmd=``) is source output rather than something an operator types, so it is
    checked separately.
    """
    joined = re.sub(r"\\\r?\n", " ", _RUNBOOK.read_text(encoding="utf-8"))
    commands: list[str] = []
    for match in _REARM_INVOCATION.finditer(joined):
        line_start = joined.rfind("\n", 0, match.start()) + 1
        args = " ".join(match.group(1).split())
        if args and "rearm_cmd=" not in joined[line_start : match.start()]:
            commands.append(args)
    return commands


def test_the_runbook_documents_at_least_one_command() -> None:
    """Guard the guard: an empty sweep would pass the assertion below vacuously."""
    assert len(_documented_invocations()) >= 3


def test_every_documented_rearm_command_carries_an_explicit_config() -> None:
    bare = [args for args in _documented_invocations() if not args.startswith("--config ")]

    assert bare == [], (
        "these runbook commands omit --config; the env fallback is a safety net, not the "
        f"documented form (and bare metal has no MOUSEDROID_CONFIG in the operator's shell): {bare}"
    )


def test_the_runbooks_boot_log_excerpt_shows_the_status_hint_with_its_config() -> None:
    """Ruling 44: the documented hint inspects; the clear form is typed by the operator."""
    text = _RUNBOOK.read_text(encoding="utf-8")
    lines = text.splitlines()

    excerpt = next(line for line in lines if line.strip().startswith("rearm_cmd="))

    assert "--config " in excerpt
    assert "--status" in excerpt
    assert "--confirm-area-clear" not in excerpt
    assert "--operator" not in excerpt
    # What clearing takes is the next logged field, as prose rather than a command.
    clause = next(line for line in lines if line.strip().startswith("clear_requires="))
    assert "--operator <you> --confirm-area-clear" in clause
    assert "mousedroid.cli.rearm" not in clause


def test_the_runbook_explains_the_fallback_without_promoting_it() -> None:
    text = _RUNBOOK.read_text(encoding="utf-8")

    for var in RUNTIME_CONFIG_ENV_VARS:
        assert var in text, f"the runbook's fallback paragraph must name {var}"
    # Whitespace-normalised: a hard wrap between the two words must not fail the pin.
    assert "safety net" in " ".join(text.split())
    assert "not the documented form" in " ".join(text.split())


# --- the safety net has something to read in the shipped container ------------------------


def test_the_shipped_compose_pins_a_variable_the_cli_falls_back_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``docker compose exec`` inherits the container's inline ``environment:``.

    So the fallback works on the production container only while compose pins one of the
    variables the shared resolver reads. (test_f052_aqa pins that it equals the file the
    rover is started with; this pins that the re-arm CLI would read it.)
    """
    compose = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    raw = compose["services"]["mousedroid"]["environment"]
    pinned = (
        {str(k): str(v) for k, v in raw.items()}
        if isinstance(raw, dict)
        else dict(str(item).split("=", 1) for item in raw if "=" in str(item))
    )
    names = [name for name in pinned if name in RUNTIME_CONFIG_ENV_VARS]
    assert names, "compose pins none of the variables the re-arm CLI falls back to"

    monkeypatch.setenv(names[0], pinned[names[0]])
    resolution = resolve_runtime_config()

    assert resolution.source == "env_single"
    assert resolution.paths == (Path(pinned[names[0]]),)
