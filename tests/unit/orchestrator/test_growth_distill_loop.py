"""Unit tests for the orchestrator's growth-distillation slow-cadence loop.

Pins the defensive guard + enable predicate so the slow-cadence background loop
is a safe no-op when the coordinator is absent — and the enable predicate
survives ``-O`` (it uses an explicit ``None`` check, not ``assert``).
"""

from __future__ import annotations

import pytest

from mousedroid.config.schema import Settings
from mousedroid.factory import build_orchestrator


@pytest.mark.asyncio
async def test_distill_loop_returns_immediately_without_coordinator() -> None:
    """The loop returns at once when the coordinator is ``None`` (guard branch)."""
    cfg = Settings.model_validate(
        {"mock_hardware": True, "vla": {"backend": "mock"}, "growth": {"enabled": True}}
    )
    orchestrator = build_orchestrator(cfg)
    orchestrator._growth_coordinator = None  # type: ignore[attr-defined]
    await orchestrator._growth_distill_loop()  # type: ignore[attr-defined]


def test_enable_predicate_false_when_block_absent() -> None:
    """``_growth_enabled`` is False when the block is absent."""
    cfg = Settings.model_validate({"mock_hardware": True})
    orchestrator = build_orchestrator(cfg)
    assert orchestrator._growth_enabled() is False  # type: ignore[attr-defined]


def test_enable_predicate_false_when_disabled() -> None:
    """``_growth_enabled`` is False when the block is present but disabled."""
    cfg = Settings.model_validate({"mock_hardware": True, "growth": {"enabled": False}})
    orchestrator = build_orchestrator(cfg)
    assert orchestrator._growth_enabled() is False  # type: ignore[attr-defined]


def test_enable_predicate_true_when_enabled() -> None:
    """``_growth_enabled`` is True when the block is enabled."""
    cfg = Settings.model_validate(
        {"mock_hardware": True, "vla": {"backend": "mock"}, "growth": {"enabled": True}}
    )
    orchestrator = build_orchestrator(cfg)
    assert orchestrator._growth_enabled() is True  # type: ignore[attr-defined]


def test_coordinator_wired_when_enabled_with_vla() -> None:
    """An enabled block with a VLA teacher wires a non-None coordinator."""
    cfg = Settings.model_validate(
        {"mock_hardware": True, "vla": {"backend": "mock"}, "growth": {"enabled": True}}
    )
    orchestrator = build_orchestrator(cfg)
    assert orchestrator._growth_coordinator is not None  # type: ignore[attr-defined]
    assert orchestrator._growth_task is None  # type: ignore[attr-defined]  # not spawned until start()
