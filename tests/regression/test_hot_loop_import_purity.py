"""Hot-loop import purity — external agents never in the control path.

Acceptance gate: importing the autonomous orchestrator or the factory
mission module must NOT pull ``google.adk``, ``honcho``, ``composio``,
``composio_core``, ``anthropic`` or ``claude_agent_sdk`` into ``sys.modules``
(CHARTER section 4, invariant 10: the 30 Hz loop stays LLM-free).

The two Anthropic names matter since F-059: ``AnthropicLLMGateway`` is now
reachable from config and ``tools/`` may grow an Agent-SDK reviewer. The
gateway imports ``anthropic`` lazily inside ``start()`` - a function-level
import, which this import-time check cannot (and should not) see - so what the
pin guards is a MODULE-SCOPE import creeping into the orchestrator import graph.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_FORBIDDEN_MODULES: tuple[str, ...] = (
    "google.adk",
    "honcho",
    "composio",
    "composio_core",
    "anthropic",
    "claude_agent_sdk",
)


def test_orchestrator_autonomous_does_not_import_external_agents() -> None:
    """Importing the 30 Hz orchestrator must not load external agent SDKs."""
    for mod in _FORBIDDEN_MODULES:
        sys.modules.pop(mod, None)

    importlib.import_module("mousedroid.orchestrator.autonomous")

    for mod in _FORBIDDEN_MODULES:
        assert mod not in sys.modules, (
            f"{mod!r} found in sys.modules after importing mousedroid.orchestrator.autonomous"
        )


def test_factory_mission_does_not_import_external_agents() -> None:
    """Importing the factory mission module must not load external agent SDKs."""
    for mod in _FORBIDDEN_MODULES:
        sys.modules.pop(mod, None)

    importlib.import_module("mousedroid.factory.mission")

    for mod in _FORBIDDEN_MODULES:
        assert mod not in sys.modules, (
            f"{mod!r} found in sys.modules after importing mousedroid.factory.mission"
        )


# --------------------------------------------------------------------------- #
# Order-independent twin: the same question asked of a FRESH interpreter
# --------------------------------------------------------------------------- #
# The two tests above pop the forbidden names and re-import their target, but
# ``importlib.import_module`` hands back an ALREADY-imported module without
# executing it again. pytest imports every test module at collection time, so in
# a full run (or any run that also collects a file importing the orchestrator)
# the orchestrator is cached before these tests execute, and a module-scope
# ``import anthropic`` added to it stays invisible to them. Measured, not
# assumed: with exactly that import planted in ``orchestrator/_action_mixin.py``
# the pin goes red when this file runs alone and stays green when
# ``test_orchestrator_mixin_surface.py`` is collected first. A new interpreter has
# no import history, so the tests below see the import graph itself.
#
# Reading ``sys.modules`` alone is still blind in CI, which installs none of the
# SDKs: ``try: import anthropic / except ImportError`` at module scope would
# leave nothing behind there. So the probe also installs a ``sys.meta_path``
# recorder that notes every import the machinery is ASKED to resolve for a
# forbidden name, whether or not the package exists.
_FRESH_IMPORT_PROBE = textwrap.dedent(
    """
    import importlib, json, sys

    targets, forbidden, attempt = (json.loads(arg) for arg in sys.argv[1:4])
    requested = set()


    class _Recorder:
        def find_spec(self, name, path=None, target=None):
            for banned in forbidden:
                if name == banned or name.startswith(banned + "."):
                    requested.add(banned)
            return None


    sys.meta_path.insert(0, _Recorder())
    for name in attempt:
        try:
            importlib.import_module(name)
        except ImportError:
            pass
    for target in targets:
        importlib.import_module(target)
    loaded = {name for name in forbidden if name in sys.modules}
    print(json.dumps(sorted(loaded | requested)))
    """
)


def _loaded_in_fresh_interpreter(
    targets: tuple[str, ...],
    forbidden: tuple[str, ...],
    *,
    attempt: tuple[str, ...] = (),
    extra_path: Path | None = None,
) -> list[str]:
    """Import ``targets`` in a new interpreter; return the ``forbidden`` names it loaded.

    ``attempt`` names are imported first with any ImportError swallowed (used only
    by the probe's own positive control); ``extra_path`` is put on ``PYTHONPATH`` so a
    test can supply a throwaway module as a target.
    """
    paths = [str(_REPO_ROOT / "src")]
    if extra_path is not None:
        paths.append(str(extra_path))
    pythonpath = os.pathsep.join(p for p in (*paths, os.environ.get("PYTHONPATH", "")) if p)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _FRESH_IMPORT_PROBE,
            json.dumps(targets),
            json.dumps(forbidden),
            json.dumps(attempt),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
        cwd=_REPO_ROOT,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"fresh-interpreter import failed:\n{proc.stderr}"
    loaded: list[str] = json.loads(proc.stdout.strip().splitlines()[-1])
    return loaded


def test_fresh_interpreter_probe_sees_an_imported_module() -> None:
    """Positive control: the probe can report an import at all (stdlib stand-in).

    Without it, an empty result below could mean "clean" or "the probe sees
    nothing" - and the second reading is the failure this file exists to prevent.
    """
    assert _loaded_in_fresh_interpreter(("sqlite3",), ("sqlite3", "honcho")) == ["sqlite3"]


def test_fresh_interpreter_probe_reports_an_import_that_was_attempted_but_failed() -> None:
    """Positive control for the ``sys.meta_path`` recorder.

    CI installs none of the SDKs, so an import of one fails there. If the probe only
    read ``sys.modules`` it would be blind in exactly the environment that enforces
    this pin; the recorder notes the REQUEST. ``honcho`` is reported whether or not
    it happens to be installed, so this holds on every machine.
    """
    assert _loaded_in_fresh_interpreter((), ("honcho",), attempt=("honcho",)) == ["honcho"]


def test_a_guarded_import_is_caught_even_when_the_sdk_is_absent(tmp_path: Path) -> None:
    """``try: import <sdk> / except ImportError`` at module scope is still a hot-loop import.

    The orchestrator would then run with or without the SDK depending on the host,
    which is exactly the coupling invariant 10 forbids. ``claude_agent_sdk`` is the
    stand-in because it is not installed in the development or CI environment.
    """
    (tmp_path / "f059_guarded_import_probe.py").write_text(
        "try:\n    import claude_agent_sdk\nexcept ImportError:\n    claude_agent_sdk = None\n",
        encoding="utf-8",
    )
    loaded = _loaded_in_fresh_interpreter(
        ("f059_guarded_import_probe",), ("claude_agent_sdk",), extra_path=tmp_path
    )
    assert loaded == ["claude_agent_sdk"]


def test_orchestrator_import_graph_loads_no_external_sdk_in_a_fresh_interpreter() -> None:
    """Importing the 30 Hz orchestrator and the mission factory loads none of the SDKs."""
    loaded = _loaded_in_fresh_interpreter(
        ("mousedroid.orchestrator.autonomous", "mousedroid.factory.mission"),
        _FORBIDDEN_MODULES,
    )
    assert loaded == [], f"module-scope imports reachable from the hot loop: {loaded}"
