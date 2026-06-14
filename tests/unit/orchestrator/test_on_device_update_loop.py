"""Unit tests for the orchestrator's Phase-6 WS3 on-device slow-cadence loop.

Pins the defensive guard + enable predicate so the slow-cadence background loop
is a safe no-op when the coordinator is absent — and the enable predicate
survives ``-O`` (it uses an explicit ``None`` check, not ``assert``).
"""

from __future__ import annotations

import pytest

from mousedroid.config.schema import Settings
from mousedroid.factory import build_orchestrator


@pytest.mark.asyncio
async def test_update_loop_returns_immediately_without_coordinator() -> None:
    """The loop returns at once when the coordinator is ``None`` (guard branch)."""
    cfg = Settings.model_validate({"mock_hardware": True, "on_device_learning": {"enabled": True}})
    orchestrator = build_orchestrator(cfg)
    # Null out the coordinator the factory wired so the guard branch fires.
    orchestrator._on_device_coordinator = None  # type: ignore[attr-defined]

    # Must complete (return) without awaiting the sleep loop.
    await orchestrator._on_device_update_loop()  # type: ignore[attr-defined]


def test_enable_predicate_false_when_block_absent() -> None:
    """``_on_device_learning_enabled`` is False when the block is absent."""
    cfg = Settings.model_validate({"mock_hardware": True})
    orchestrator = build_orchestrator(cfg)

    assert orchestrator._on_device_learning_enabled() is False  # type: ignore[attr-defined]


def test_enable_predicate_true_when_enabled() -> None:
    """``_on_device_learning_enabled`` is True when the block is enabled."""
    cfg = Settings.model_validate({"mock_hardware": True, "on_device_learning": {"enabled": True}})
    orchestrator = build_orchestrator(cfg)

    assert orchestrator._on_device_learning_enabled() is True  # type: ignore[attr-defined]
