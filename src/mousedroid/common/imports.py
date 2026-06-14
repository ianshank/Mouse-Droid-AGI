"""Optional-dependency probing without triggering an import side effect.

Several modules guard hardware-/accelerator-specific stacks (``picamera2``,
``hailo_platform``, ``torch2trt``, ``isaaclab``, ``serial``) behind an
availability check. Using :func:`importlib.util.find_spec` keeps those guards
free of an unused ``import x`` (and the ``# noqa: F401`` that used to mark it),
while :func:`module_available` adds the small amount of robustness needed for
tests that inject fakes into :data:`sys.modules`.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

__all__ = ["module_available", "module_importable"]


def module_available(name: str) -> bool:
    """Return ``True`` iff ``name`` can be imported in the current environment.

    The check is import-side-effect free: it resolves the module spec rather
    than importing the package. A module already present in :data:`sys.modules`
    (e.g. a test fake injected via ``monkeypatch.setitem``) counts as available;
    a spec finder that raises ``ImportError``/``ValueError`` counts as
    unavailable.

    Args:
        name: Fully-qualified top-level module name to probe.

    Returns:
        Whether the module is importable.
    """
    if sys.modules.get(name) is not None:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def module_importable(name: str) -> bool:
    """Return ``True`` iff ``name`` actually imports in the current environment.

    Unlike :func:`module_available` (which only resolves the module *spec*),
    this performs a real, guarded ``importlib.import_module``. A package whose
    spec is discoverable but whose import fails at runtime — typically because a
    native/accelerator dependency is missing (e.g. ``picamera2`` without its
    libcamera bindings) — counts as **not** importable here.

    Use this (rather than :func:`module_available`) whenever a positive result
    is immediately followed by importing the package; spec-presence is not a
    sufficient guarantee that the subsequent import will succeed.

    Args:
        name: Fully-qualified top-level module name to probe.

    Returns:
        Whether the module can actually be imported.
    """
    if sys.modules.get(name) is not None:
        return True
    try:
        importlib.import_module(name)
    except Exception:
        # Any import-time failure (native deps missing, side-effect error,
        # etc.) means the module is not usable in this environment.
        return False
    return True
