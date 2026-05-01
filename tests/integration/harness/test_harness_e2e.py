"""End-to-end integration test for the agent harness wiring.

Drives a fully-wired :class:`MouseDroidOrchestrator` through several ticks
with the task tracker, hook registry, JSONL journal, AutoApproveGate, and
a single registered skill. Asserts that every harness phase fires in order
and that journal entries are persisted off the hot loop.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch

from mousedroid.config.schema import (
    HarnessJournalConfig,
    HarnessTrackerConfig,
    Settings,
)
from mousedroid.harness.approval.auto import AutoApproveGate
from mousedroid.harness.hooks import HookRegistry
from mousedroid.harness.journal.jsonl_journal import JSONLJournal
from mousedroid.harness.journal.protocol import JournalEntry
from mousedroid.harness.predicates import CallablePredicate, TickCountReached
from mousedroid.harness.protocol import (
    HookPhase,
    HookSpec,
    TaskSpec,
    TaskStatus,
    TickContext,
)
from mousedroid.harness.task_tracker import InMemoryTaskTracker
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext
from mousedroid.sensing.bundle import MouseDroidObservationBundle
from mousedroid.skills.delegator import SkillDelegator
from mousedroid.skills.protocol import SkillSpec
from mousedroid.skills.registry import SkillRegistry


def _observation(cfg: Settings) -> MouseDroidObservationBundle:
    return MouseDroidObservationBundle(
        _timestamp=0.0,
        _vision_features=np.zeros(cfg.camera.feature_dim, dtype=np.float32),
        _distance_m=1.5,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _audio_chunk=np.zeros(1024, dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )


def _build_orchestrator(
    *,
    hook_registry,
    task_tracker,
    journal,
    skill_delegator,
) -> MouseDroidOrchestrator:
    cfg = Settings(mock_hardware=True)
    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, cfg.model.hidden_dim),
        0.1,
    )
    agent = MagicMock()
    agent.name = "harness_test_agent"
    agent.act.return_value = torch.tensor([0.0, 0.0, 0.0])
    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = SafetyContext(is_emergency=False)
    esp32 = AsyncMock()
    sensor_manager = AsyncMock()
    sensor_manager.read_all.return_value = _observation(cfg)
    sensor_manager.recovery_attempt.return_value = 0
    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cfg=cfg,
        hook_registry=hook_registry,
        task_tracker=task_tracker,
        journal=journal,
        skill_delegator=skill_delegator,
    )


@pytest.mark.asyncio
async def test_harness_phases_fire_in_order_and_task_completes(tmp_path: Path) -> None:
    """Three-tick scenario:

    * Submit a task whose acceptance predicate flips True after one tick.
    * Verify pre/post hooks fire in order, post_tick sees the completed
      task, and the JSONL journal contains entries from JournalAppendHook.
    """
    journal = JSONLJournal(
        HarnessJournalConfig(backend="jsonl", path=tmp_path / "j.jsonl", queue_max=64)
    )
    tracker = InMemoryTaskTracker(HarnessTrackerConfig(enabled=True, history_size=8, max_active=4))
    hooks = HookRegistry()
    skill_registry = SkillRegistry()
    skill_registry.register(SkillSpec(name="diag", tool_names=frozenset()))
    delegator = SkillDelegator(skill_registry, AutoApproveGate(), journal, tracker)

    fired: list[str] = []

    async def record(ctx: TickContext, *, label: str) -> None:
        fired.append(label)
        await journal.append(JournalEntry(event=f"hook_{label}", payload={"tick": ctx.tick_index}))

    for phase, label in (
        (HookPhase.PRE_TICK, "pre_tick"),
        (HookPhase.PRE_ACTION, "pre_action"),
        (HookPhase.POST_ACTION, "post_action"),
        (HookPhase.POST_TICK, "post_tick"),
    ):
        hooks.register(
            HookSpec(
                name=label,
                phase=phase,
                handler=lambda c, _l=label: record(c, label=_l),
            )
        )

    orch = _build_orchestrator(
        hook_registry=hooks,
        task_tracker=tracker,
        journal=journal,
        skill_delegator=delegator,
    )
    await orch.start()
    try:
        # Submit a task whose predicate becomes True after the second tick.
        spec = TaskSpec(
            id="t-fast",
            goal="finish quickly",
            acceptance_predicate=CallablePredicate(lambda state, ctx: ctx.tick_index >= 2),
            timeout_s=10.0,
        )
        tracker.submit(spec)

        await orch.tick()
        await orch.tick()
        await orch.tick()
    finally:
        await orch.stop()

    # Hooks fired in registration order, three times each.
    assert fired.count("pre_tick") == 3
    assert fired.count("pre_action") == 3
    assert fired.count("post_action") == 3
    assert fired.count("post_tick") == 3
    # Order within each tick: pre_tick → pre_action → post_action → post_tick.
    expected_pattern = (
        "pre_tick",
        "pre_action",
        "post_action",
        "post_tick",
    ) * 3
    assert tuple(fired) == expected_pattern

    # Tracker terminal status reachable.
    state = tracker.get("t-fast")
    assert state is not None
    assert state.status == TaskStatus.COMPLETED

    # Journal flushed and contains hook entries.
    raw = (tmp_path / "j.jsonl").read_text(encoding="utf-8").splitlines()
    events = [line for line in raw if line]
    assert any("hook_pre_tick" in line for line in events)
    assert any("hook_post_tick" in line for line in events)


@pytest.mark.asyncio
async def test_harness_disabled_path_is_byte_identical_to_legacy(tmp_path: Path) -> None:
    """When ``hook_registry`` / ``journal`` / ``task_tracker`` are not
    supplied, the orchestrator must run exactly as it did before this
    branch — no exceptions, no journal files written, no hooks fired."""
    cfg = Settings(mock_hardware=True)
    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, cfg.model.hidden_dim),
        0.1,
    )
    agent = MagicMock()
    agent.name = "legacy_agent"
    agent.act.return_value = torch.tensor([0.0, 0.0, 0.0])
    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = SafetyContext(is_emergency=False)
    esp32 = AsyncMock()
    sensor_manager = AsyncMock()
    sensor_manager.read_all.return_value = _observation(cfg)

    orch = MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cfg=cfg,
    )
    await orch.start()
    try:
        await orch.tick()
        await orch.tick()
    finally:
        await orch.stop()

    # No journal file should have been created.
    assert list(tmp_path.iterdir()) == []
    # Tick count advanced normally.
    assert orch._tick_count == 2


@pytest.mark.asyncio
async def test_on_error_hook_fires_when_tick_raises(tmp_path: Path) -> None:
    """When a downstream subsystem raises, the harness's ``on_error`` hooks
    must fire with the error attached to the context, then re-raise."""
    cfg = Settings(mock_hardware=True)
    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, cfg.model.hidden_dim),
        0.1,
    )
    agent = MagicMock()
    agent.name = "kaboom"
    agent.act.side_effect = RuntimeError("agent exploded")
    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = SafetyContext(is_emergency=False)
    esp32 = AsyncMock()
    sensor_manager = AsyncMock()
    sensor_manager.read_all.return_value = _observation(cfg)

    hooks = HookRegistry()
    captured: list[BaseException | None] = []

    async def on_err(ctx: TickContext) -> None:
        captured.append(ctx.error)

    hooks.register(HookSpec(name="oe", phase=HookPhase.ON_ERROR, handler=on_err))

    orch = MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cfg=cfg,
        hook_registry=hooks,
    )
    with pytest.raises(RuntimeError, match="agent exploded"):
        await orch.tick()
    assert len(captured) == 1
    assert isinstance(captured[0], RuntimeError)
    assert str(captured[0]) == "agent exploded"


@pytest.mark.asyncio
async def test_task_tracker_runs_off_orchestrator(tmp_path: Path) -> None:
    """A task whose predicate uses ``TickCountReached`` must complete by
    consulting the orchestrator's tick index over successive ticks."""
    tracker = InMemoryTaskTracker(HarnessTrackerConfig(enabled=True, history_size=8, max_active=4))
    journal = JSONLJournal(
        HarnessJournalConfig(backend="jsonl", path=tmp_path / "j.jsonl", queue_max=32)
    )
    skill_registry = SkillRegistry()
    delegator = SkillDelegator(skill_registry, AutoApproveGate(), journal, tracker)

    orch = _build_orchestrator(
        hook_registry=HookRegistry(),
        task_tracker=tracker,
        journal=journal,
        skill_delegator=delegator,
    )
    await orch.start()
    try:
        spec = TaskSpec(
            id="three-ticks",
            goal="wait three ticks",
            acceptance_predicate=TickCountReached(n=3),
            metadata={"submitted_at_tick": 0},
        )
        tracker.submit(spec)
        for _ in range(4):
            await orch.tick()
    finally:
        await orch.stop()

    state = tracker.get("three-ticks")
    assert state is not None
    assert state.status == TaskStatus.COMPLETED
