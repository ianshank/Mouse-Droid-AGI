"""Unit tests for the ``mock_hardware_resolved`` boot log (F-014 follow-up).

The orchestrator emits the RESOLVED ``cfg.mock_hardware`` boolean ONCE at the
start of ``start()`` — right after ``orchestrator_starting`` and before any
subsystem is brought up. ``health_check`` already exposes the same value, but
that is an on-demand API response; this log makes the resolved boolean visible
in container logs at boot so an operator can tell at a glance whether real or
mock drivers were wired.

These tests drive ``start()`` with ``AsyncMock`` subsystem stand-ins so no real
audio / hardware is touched, mirroring the construction discipline of
``test_greeting_startup.py``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog.testing

from mousedroid.config.schema import Settings
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator


def _make_orch(*, mock_hardware: bool) -> MouseDroidOrchestrator:
    """Build an orchestrator with mocked subsystems and the given flag.

    Args:
        mock_hardware: The resolved ``cfg.mock_hardware`` boolean to wire.

    Returns:
        An orchestrator whose ``start()`` touches no real hardware.
    """
    cfg = Settings(mock_hardware=mock_hardware)
    return MouseDroidOrchestrator(
        world_model=MagicMock(),
        agents=[MagicMock()],
        safety_monitor=MagicMock(),
        esp32=AsyncMock(),
        sensor_manager=AsyncMock(),
        cfg=cfg,
    )


def _resolved_values(logs: list[dict[str, object]]) -> list[object]:
    """Extract the ``value`` of every ``mock_hardware_resolved`` event.

    Args:
        logs: Captured structlog event dicts.

    Returns:
        The ``value`` field of each matching event, in emission order.
    """
    return [e["value"] for e in logs if e["event"] == "mock_hardware_resolved"]


@pytest.mark.asyncio
async def test_start_logs_resolved_mock_hardware_true() -> None:
    """``start()`` emits ``mock_hardware_resolved`` with ``value=True``."""
    orch = _make_orch(mock_hardware=True)
    with structlog.testing.capture_logs() as logs:
        await orch.start()
    assert _resolved_values(logs) == [True], (
        "start() must log the resolved mock_hardware boolean exactly once"
    )
    assert orch._running is True


@pytest.mark.asyncio
async def test_start_logs_resolved_mock_hardware_matches_cfg() -> None:
    """The logged value mirrors ``cfg.mock_hardware`` (not a hardcoded literal)."""
    orch = _make_orch(mock_hardware=True)
    with structlog.testing.capture_logs() as logs:
        await orch.start()
    assert _resolved_values(logs) == [orch._cfg.mock_hardware]
