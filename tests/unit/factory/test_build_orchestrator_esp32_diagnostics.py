"""``build_orchestrator`` hands the MCP ``esp32_diagnostics`` tool the driver it owns (F-059, 4.3).

The orchestrator holds the exclusive serial port, so the diagnostics tool has to read through
THAT driver (and see THAT ``Settings``) instead of opening a second port. The hand-off is a pair
of keyword arguments at the one production call site of ``create_default_registry``; these tests
pin it there, where a refactor of the factory could silently drop it and leave the tool answering
``unconfigured`` on a live rover.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from mousedroid.common.tools import registry as registry_mod
from mousedroid.config.schema import Settings
from mousedroid.factory import build_orchestrator


class _Built:
    """What ``build_orchestrator`` built, plus what it passed to ``create_default_registry``."""

    def __init__(self) -> None:
        self.cfg = Settings(mock_hardware=True)
        self.kwargs: dict[str, Any] = {}
        self.registry: registry_mod.ToolRegistry | None = None
        self.orchestrator: Any = None


@pytest.fixture(scope="module")
def built() -> _Built:
    """Build one mock-hardware orchestrator with a spy on the registry builder."""
    out = _Built()
    real = registry_mod.create_default_registry

    def spy(*args: Any, **kwargs: Any) -> registry_mod.ToolRegistry:
        out.kwargs = dict(kwargs)
        out.registry = real(*args, **kwargs)
        return out.registry

    # ``build_orchestrator`` imports the builder inside the function, so patching the
    # defining module is what it resolves.
    with patch("mousedroid.common.tools.registry.create_default_registry", side_effect=spy):
        out.orchestrator = build_orchestrator(out.cfg)
    return out


def test_registry_builder_gets_the_settings_and_the_drivers_the_orchestrator_holds(
    built: _Built,
) -> None:
    assert built.kwargs["settings"] is built.cfg
    assert built.kwargs["esp32_driver"] is built.orchestrator._esp32
    # The registry inspected here is the one the orchestrator dispatches through.
    assert built.orchestrator._tool_registry is built.registry


async def test_wired_tool_answers_from_the_settings_not_unconfigured(built: _Built) -> None:
    """A mock-hardware build says ``mock``; ``unconfigured`` would mean the hand-off was lost."""
    assert built.registry is not None
    result = await built.registry.dispatch("esp32_diagnostics")
    assert result == {"status": "mock", "enabled": True, "mock_hardware": True}
