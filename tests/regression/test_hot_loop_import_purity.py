"""Hot-loop import purity — external agents never in the control path.

Acceptance gate: importing the autonomous orchestrator or the factory
mission module must NOT pull ``google.adk``, ``honcho``, ``composio``,
or ``composio_core`` into ``sys.modules``.
"""

from __future__ import annotations

import importlib
import sys

_FORBIDDEN_MODULES: tuple[str, ...] = (
    "google.adk",
    "honcho",
    "composio",
    "composio_core",
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
