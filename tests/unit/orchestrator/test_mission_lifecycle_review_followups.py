"""Regression tests for the Tier C2.2 review follow-ups (PR #98).

Covers four reviewer-flagged behaviours that the prior hardening commit
left open:

* Item #4: empty vision_features must invalidate the cached previous
  frame so the next non-empty observation isn't paired with a stale
  pre-dropout frame.
* Item #5: ``process_mission`` must NOT call ``lifecycle.start_mission``
  when no parser or LLM accepts the natural-language command — Stage 3
  (fallback to ``GoalVector()``) keeps the lifecycle idle.
* Item #6: ``MissionLifecycle.start_mission`` must submit a synthetic
  task to the wired ``TaskTrackerProtocol`` and the terminal
  ``_transition`` must forward SUCCEEDED → COMPLETED / FAILED → FAILED
  via ``tracker.update``.
* Item #7: the VLM observation tensor must use ``torch.tensor`` (single
  copy, contiguity-tolerant) rather than ``torch.from_numpy`` + clone.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch

from mousedroid.config.schema import MissionConfig, Settings
from mousedroid.harness.protocol import TaskState, TaskStatus
from mousedroid.orchestrator.mission_lifecycle import (
    MissionLifecycle,
    MissionLifecycleState,
)

# Reuse the wiring fixture from the existing wiring tests.
from tests.unit.orchestrator.test_mission_lifecycle_wiring import (
    _build_orch_with_lifecycle,
)

# ---------------------------------------------------------------------------
# Item #4: empty-vf invalidation of _prev_obs_for_vlm
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_vision_features_invalidates_cached_prev_obs() -> None:
    """vf.size == 0 must clear ``_prev_obs_for_vlm`` so no stale pairing.

    Sequence: tick 1 populates cache → tick 2 has empty vf (must clear
    cache) → tick 3 has non-empty vf. Without invalidation, tick 3 would
    pair its observation against tick 1's frame — violating the
    lifecycle's ``(obs_t, obs_tminus1)`` adjacency contract and silently
    corrupting VLM progress scoring across the dropout boundary.
    """
    from mousedroid.safety.context import SafetyContext

    cfg = Settings(mock_hardware=True)
    cfg.mission = MissionConfig(replan_enabled=True)
    lifecycle = MagicMock(spec=MissionLifecycle)
    lifecycle.tick = AsyncMock()

    # vf sequence: full → empty → full
    vf_sequence = [
        np.ones(8, dtype=np.float32),
        np.zeros(0, dtype=np.float32),
        np.full(8, 2.0, dtype=np.float32),
    ]
    obs_proxy = MagicMock()
    type(obs_proxy).vision_features = property(
        lambda _self: vf_sequence[min(_call_count[0], len(vf_sequence) - 1)]
    )
    _call_count = [0]

    async def _read_all() -> object:
        idx = _call_count[0]
        _call_count[0] += 1
        proxy = MagicMock()
        proxy.vision_features = vf_sequence[idx]
        return proxy

    sensor_manager = MagicMock()
    sensor_manager.read_all = AsyncMock(side_effect=_read_all)

    wm = MagicMock()
    combined = cfg.model.hidden_dim + cfg.model.cfc_hidden_dim
    wm.observe_step.return_value = (
        torch.zeros(1, combined),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, combined),
        0.1,
    )
    agent = MagicMock()
    agent.name = "mock"
    agent.act.return_value = torch.zeros(cfg.model.action_dim)
    sm = MagicMock()
    sm.evaluate.return_value = SafetyContext(is_emergency=False)

    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    orch = MouseDroidOrchestrator(
        world_model=wm,
        agents=[agent],
        safety_monitor=sm,
        esp32=AsyncMock(),
        sensor_manager=sensor_manager,
        cfg=cfg,
        mission_lifecycle=lifecycle,
    )

    # Tick 1: full vf — caches prev frame, no tick yet (needs prev frame).
    await orch.tick()
    assert lifecycle.tick.await_count == 0
    assert orch._prev_obs_for_vlm is not None  # cached

    # Tick 2: empty vf — must invalidate the cache (early-return BEFORE
    # tick AND clear cache so next non-empty frame isn't paired with stale).
    await orch.tick()
    assert lifecycle.tick.await_count == 0
    assert orch._prev_obs_for_vlm is None  # invalidated

    # Tick 3: full vf again — must populate cache fresh, NOT pair with
    # the old tick-1 frame. Since prev was cleared, no lifecycle.tick fires
    # (matches the lifecycle's first-frame-after-cache-miss contract).
    await orch.tick()
    assert lifecycle.tick.await_count == 0
    assert orch._prev_obs_for_vlm is not None


# ---------------------------------------------------------------------------
# Item #5: deferred start_mission — Stage 3 (unresolved) does NOT start
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_mission_does_not_start_lifecycle_on_unresolved_command() -> None:
    """Stage 3 fallback (no parser + no LLM) must NOT call start_mission.

    Previously, ``start_mission`` ran before any parser even saw the
    command, leaving the lifecycle RUNNING on the neutral fallback goal.
    The fix moves the call to inside the Stage 1 / Stage 2 acceptance
    branches.
    """
    cfg = Settings(mock_hardware=True)
    cfg.mission = MissionConfig(replan_enabled=True)
    lifecycle = MagicMock(spec=MissionLifecycle)
    lifecycle.start_mission = MagicMock()
    lifecycle.current_state = MissionLifecycleState.PENDING

    # Build orch WITHOUT a parser or LLM so every command falls through
    # to Stage 3.
    orch = _build_orch_with_lifecycle(cfg, lifecycle)
    # Strip parser / LLM to force Stage 3.
    orch._mission_parser = None  # type: ignore[assignment]
    orch._llm_gateway = None  # type: ignore[assignment]

    await orch.process_mission("totally nonsense command xyzzy")

    lifecycle.start_mission.assert_not_called()


@pytest.mark.asyncio
async def test_process_mission_starts_lifecycle_when_llm_accepts() -> None:
    """LLM-accepted Stage 2 path must call start_mission with the NL goal."""
    from mousedroid.llm_gateway.protocol import GoalVector

    cfg = Settings(mock_hardware=True)
    cfg.mission = MissionConfig(replan_enabled=True)
    lifecycle = MagicMock(spec=MissionLifecycle)
    lifecycle.start_mission = MagicMock()

    orch = _build_orch_with_lifecycle(cfg, lifecycle)
    # No parser → bypass Stage 1.
    orch._mission_parser = None  # type: ignore[assignment]

    fake_gateway = MagicMock()
    fake_gateway.translate_mission = AsyncMock(
        return_value=GoalVector(vx_target=0.5, vy_target=0.0, omega_target=0.0)
    )
    orch._llm_gateway = fake_gateway  # type: ignore[assignment]

    await orch.process_mission("explore the warehouse")

    lifecycle.start_mission.assert_called_once()
    _args, kwargs = lifecycle.start_mission.call_args
    assert kwargs["goal_text"] == "explore the warehouse"


# ---------------------------------------------------------------------------
# Item #6: TaskTracker wiring — submit on start, update on terminal
# ---------------------------------------------------------------------------


class _RecordingTracker:
    """Minimal TaskTrackerProtocol stub that records submit / update calls."""

    def __init__(self) -> None:
        self.submits: list[tuple[str, str]] = []
        self.updates: list[tuple[str, TaskStatus, str | None]] = []
        self._states: dict[str, TaskState] = {}

    def submit(self, spec: object) -> TaskState:
        from mousedroid.harness.protocol import TaskState as _TaskState

        sp = spec  # type: ignore[assignment]
        self.submits.append((sp.id, sp.goal))  # type: ignore[attr-defined]
        state = _TaskState(spec=sp)  # type: ignore[arg-type]
        self._states[sp.id] = state  # type: ignore[attr-defined]
        return state

    def get(self, task_id: str) -> TaskState | None:
        return self._states.get(task_id)

    def update(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        error: str | None = None,
    ) -> TaskState:
        self.updates.append((task_id, status, error))
        state = self._states[task_id]
        state.status = status
        if error is not None:
            state.last_error = error
        return state

    def evaluate(self, state: TaskState, ctx: object) -> TaskStatus:
        return state.status

    async def evaluate_active(self, ctx: object) -> tuple[TaskState, ...]:
        return tuple(s for s in self._states.values() if not s.is_terminal)

    def active(self) -> object:
        return [s for s in self._states.values() if not s.is_terminal]

    def history(self) -> object:
        return list(self._states.values())

    def cancel(self, task_id: str, *, reason: str | None = None) -> TaskState:
        return self.update(task_id, TaskStatus.CANCELLED, error=reason)


def _build_lifecycle_with_tracker(tracker: _RecordingTracker) -> MissionLifecycle:
    cfg = MissionConfig(
        replan_enabled=False,  # disable replan so terminal path is deterministic
        success_threshold=0.5,
        stall_threshold=0.1,
        stall_window_ticks=5,
    )
    return MissionLifecycle(cfg, task_tracker=tracker)


def test_lifecycle_submits_synthetic_task_on_start() -> None:
    """start_mission must submit a TaskSpec carrying the mission_id + goal."""
    tracker = _RecordingTracker()
    lifecycle = _build_lifecycle_with_tracker(tracker)
    lifecycle.start_mission("mission-000042", "navigate to charger")

    assert tracker.submits == [("mission-000042", "navigate to charger")]


def test_lifecycle_forwards_failed_terminal_to_tracker() -> None:
    """fail() must transition tracker task to FAILED with the lifecycle reason."""
    tracker = _RecordingTracker()
    lifecycle = _build_lifecycle_with_tracker(tracker)
    lifecycle.start_mission("mission-1", "stop the robot")
    lifecycle.fail(reason="user_requested_stop")

    assert tracker.updates == [("mission-1", TaskStatus.FAILED, "user_requested_stop")]
    assert tracker.get("mission-1").status == TaskStatus.FAILED  # type: ignore[union-attr]


def test_lifecycle_does_not_call_tracker_when_unwired() -> None:
    """No tracker wired → start / fail must not raise."""
    cfg = MissionConfig(replan_enabled=False)
    lifecycle = MissionLifecycle(cfg, task_tracker=None)
    lifecycle.start_mission("mission-2", "no-op")
    lifecycle.fail(reason="no-op")  # must not raise


# ---------------------------------------------------------------------------
# Item #7: torch.tensor (not torch.from_numpy) tolerates non-contiguous vf
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifecycle_handles_non_contiguous_vision_features() -> None:
    """A non-contiguous numpy slice must not raise inside the helper.

    ``torch.from_numpy`` raises ``ValueError`` on non-contiguous input;
    ``torch.tensor`` (the fix) copies and tolerates any strided layout.
    Sliced sensor buffers (e.g. ``buffer[::2]``) are realistic camera
    outputs after channel selection.
    """
    from mousedroid.safety.context import SafetyContext

    cfg = Settings(mock_hardware=True)
    cfg.mission = MissionConfig(replan_enabled=True)
    lifecycle = MagicMock(spec=MissionLifecycle)
    lifecycle.tick = AsyncMock()

    # Strided slice — non-contiguous by construction.
    underlying = np.arange(16, dtype=np.float32)
    non_contig = underlying[::2]
    assert not non_contig.flags["C_CONTIGUOUS"]

    obs = MagicMock()
    obs.vision_features = non_contig

    sensor_manager = MagicMock()
    sensor_manager.read_all = AsyncMock(return_value=obs)

    wm = MagicMock()
    combined = cfg.model.hidden_dim + cfg.model.cfc_hidden_dim
    wm.observe_step.return_value = (
        torch.zeros(1, combined),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, combined),
        0.1,
    )
    agent = MagicMock()
    agent.name = "mock"
    agent.act.return_value = torch.zeros(cfg.model.action_dim)
    sm = MagicMock()
    sm.evaluate.return_value = SafetyContext(is_emergency=False)

    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    orch = MouseDroidOrchestrator(
        world_model=wm,
        agents=[agent],
        safety_monitor=sm,
        esp32=AsyncMock(),
        sensor_manager=sensor_manager,
        cfg=cfg,
        mission_lifecycle=lifecycle,
    )

    # Must not raise on non-contiguous vf.
    await orch.tick()
    await orch.tick()
    # Second tick should have actually called the lifecycle tick (prev cached on tick 1).
    assert lifecycle.tick.await_count == 1
    # And the cached prev_obs is a contiguous torch tensor (copy semantics).
    cached = orch._prev_obs_for_vlm
    assert cached is not None
    assert cached.dtype == torch.float32
    assert cached.is_contiguous()
