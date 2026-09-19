"""Shared helper to import a standalone ``scripts/*.py`` module by path.

Several operator CLIs / probes live in ``scripts/`` — outside the importable
``mousedroid`` package — so tests load them through :mod:`importlib` to exercise
their ``main()`` and helper functions. This centralises the
``spec_from_file_location`` dance that was previously copy-pasted into every
such test (``test_translate_mission_cli``, ``test_ask_rover_cli``, the probe
tests, ...), so the loading convention lives in exactly one place.

Follows the repo's ``tests/_<name>.py`` shared-helper convention (cf.
``tests/_jetson_hardware.py``); import as
``from tests._script_loader import load_script_module``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# Repo root is the parent of the ``tests/`` package directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"


def load_script_module(name: str, path: str | Path | None = None) -> ModuleType:
    """Import a standalone script module by path and return the executed module.

    Args:
        name: Import name to register the module under (e.g. ``"ask_rover"``).
            Also used to resolve ``scripts/<name>.py`` when ``path`` is omitted.
        path: Explicit filesystem path to the ``.py`` file. When ``None``,
            resolves ``scripts/<name>.py`` relative to the repo root.

    Returns:
        The fully-executed module object.

    Raises:
        FileNotFoundError: If the resolved path does not exist (clearer than the
            ``importlib`` failure that would otherwise surface).
        ImportError: If the import spec or its loader cannot be constructed.
    """
    script_path = Path(path) if path is not None else _SCRIPTS_DIR / f"{name}.py"
    if not script_path.is_file():
        msg = f"script module not found: {script_path}"
        raise FileNotFoundError(msg)
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        msg = f"could not build an import spec for {script_path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec_module, per the importlib "importing a source file
    # directly" recipe. Not cosmetic: a script that declares a `@dataclass` with a
    # bare-identifier annotation (`n: int`) under `from __future__ import
    # annotations` makes `dataclasses._process_class` resolve that string through
    # `sys.modules[cls.__module__].__dict__` while scanning for `KW_ONLY`, which
    # raises `AttributeError: 'NoneType' object has no attribute '__dict__'` when
    # the module is absent. Any decorator that introspects its own module hits the
    # same wall.
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Leave no half-initialised module behind for the next loader call.
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module
