"""Unit tests for the mission lifecycle state machine (Tier C2 / C2.2)."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from mousedroid.config.schema import MissionConfig
from mousedroid.llm_gateway.protocol import GoalVector
from mousedroid.orchestrator.mission_lifecycle import (
    MissionLifecycle,
    MissionLifecycleState,
)


class _StubVLM:
    """Test VLM that returns a queued sequence of scores."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)
        self.calls = 0

    def score(
        self,
        prev_obs: torch.Tensor,
        curr_obs: torch.Tensor,
        *,
        instruction: str | None = None,
    ) -> torch.Tensor:
        del prev_obs, curr_obs, instruction
        value = self._scores.pop(0) if self._scores else 0.0
        self.calls += 1
        return torch.tensor([[float(value)]], dtype=torch.float32)


class _StubReplanner:
    """Test replanner that returns a queued sequence of GoalVectors / Nones."""

    def __init__(self, responses: list[GoalVector | None]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def submit_replan_request(
        self,
        *,
        mission_id: str,
        goal_text: str,
        last_progress: float,
    ) -> GoalVector | None:
        self.calls.append(
            {
                "mission_id": mission_id,
                "goal_text": goal_text,
                "last_progress": last_progress,
            }
        )
        if self._responses:
            return self._responses.pop(0)
        return None


def _obs() -> torch.Tensor:
    return torch.zeros(1, 4)


def _cfg(**overrides: Any) -> MissionConfig:
    base: dict[str, Any] = {
        "replan_enabled": False,
        "success_threshold": 0.9,
        "stall_threshold": 0.1,
        "stall_window_ticks": 3,
        "max_replans_per_mission": 2,
    }
    base.update(overrides)
    return MissionConfig(**base)


# ---------------------------------------------------------------------------
# 10 TDD tests per plan §"Story C2.2 — Step 2.4: 10 TDD tests"
# ---------------------------------------------------------------------------


def test_lifecycle_starts_in_pending_no_mission_until_start_called() -> None:
    """A freshly-constructed lifecycle has no current mission."""
    lifecycle = MissionLifecycle(_cfg())
    assert lifecycle.current_state is None


def test_lifecycle_transitions_pending_to_running_on_start_mission() -> None:
    """``start_mission`` flips PENDING → RUNNING."""
    lifecycle = MissionLifecycle(_cfg())
    lifecycle.start_mission("mission-1", "explore the corridor")
    assert lifecycle.current_state == MissionLifecycleState.RUNNING


@pytest.mark.asyncio
async def test_lifecycle_transitions_running_to_succeeded_when_progress_crosses_threshold() -> None:
    """Progress >= success_threshold transitions RUNNING → SUCCEEDED."""
    vlm = _StubVLM([0.95])
    lifecycle = MissionLifecycle(_cfg(), vlm_progress=vlm)
    lifecycle.start_mission("mission-1", "goal")
    result = await lifecycle.tick(_obs(), _obs())
    assert result.state == MissionLifecycleState.SUCCEEDED
    assert result.transitioned is True


@pytest.mark.asyncio
async def test_lifecycle_stays_running_when_progress_below_success_and_above_stall() -> None:
    """Progress between stall and success keeps the lifecycle in RUNNING."""
    vlm = _StubVLM([0.5, 0.6, 0.7])
    lifecycle = MissionLifecycle(_cfg(), vlm_progress=vlm)
    lifecycle.start_mission("mission-1", "goal")
    for _ in range(3):
        result = await lifecycle.tick(_obs(), _obs())
    assert result.state == MissionLifecycleState.RUNNING
    assert result.transitioned is False


@pytest.mark.asyncio
async def test_lifecycle_replan_disabled_does_not_transition_on_stall() -> None:
    """When ``cfg.mission.replan_enabled=False`` stalls stay RUNNING — byte-identical pre-PR."""
    vlm = _StubVLM([0.0] * 10)
    lifecycle = MissionLifecycle(_cfg(replan_enabled=False), vlm_progress=vlm)
    lifecycle.start_mission("mission-1", "goal")
    for _ in range(10):
        result = await lifecycle.tick(_obs(), _obs())
    assert result.state == MissionLifecycleState.RUNNING


@pytest.mark.asyncio
async def test_lifecycle_transitions_to_replanning_after_stall_window() -> None:
    """``stall_window_ticks`` consecutive stall ticks trip REPLANNING."""
    vlm = _StubVLM([0.0, 0.0, 0.0])
    replanner = _StubReplanner([GoalVector(vx_target=0.5)])
    lifecycle = MissionLifecycle(
        _cfg(replan_enabled=True, stall_window_ticks=3),
        vlm_progress=vlm,
        replanner=replanner,
    )
    lifecycle.start_mission("mission-1", "goal")
    # First two ticks accumulate stall counter but don't transition.
    await lifecycle.tick(_obs(), _obs())
    await lifecycle.tick(_obs(), _obs())
    assert lifecycle.current_state == MissionLifecycleState.RUNNING
    # Third stall tick crosses the window — replanner returns a new
    # GoalVector so the lifecycle ends back in RUNNING.
    result = await lifecycle.tick(_obs(), _obs())
    assert result.state == MissionLifecycleState.RUNNING
    assert result.transitioned is True
    assert result.reason == "replan_succeeded"
    assert lifecycle.replan_count == 1
    assert len(replanner.calls) == 1


@pytest.mark.asyncio
async def test_lifecycle_replan_returns_none_transitions_to_failed() -> None:
    """LLM returning ``None`` from a replan request fails the mission."""
    vlm = _StubVLM([0.0, 0.0, 0.0])
    replanner = _StubReplanner([None])
    lifecycle = MissionLifecycle(
        _cfg(replan_enabled=True, stall_window_ticks=3),
        vlm_progress=vlm,
        replanner=replanner,
    )
    lifecycle.start_mission("mission-1", "goal")
    await lifecycle.tick(_obs(), _obs())
    await lifecycle.tick(_obs(), _obs())
    result = await lifecycle.tick(_obs(), _obs())
    assert result.state == MissionLifecycleState.FAILED
    assert result.reason == "llm_replan_unavailable"


@pytest.mark.asyncio
async def test_lifecycle_no_replanner_with_replan_enabled_fails_on_stall() -> None:
    """``replan_enabled=True`` with no replanner fails the mission on stall."""
    vlm = _StubVLM([0.0, 0.0, 0.0])
    lifecycle = MissionLifecycle(
        _cfg(replan_enabled=True, stall_window_ticks=3),
        vlm_progress=vlm,
        replanner=None,
    )
    lifecycle.start_mission("mission-1", "goal")
    await lifecycle.tick(_obs(), _obs())
    await lifecycle.tick(_obs(), _obs())
    result = await lifecycle.tick(_obs(), _obs())
    assert result.state == MissionLifecycleState.FAILED
    assert result.reason == "llm_replan_unavailable"


@pytest.mark.asyncio
async def test_lifecycle_replan_limit_caps_replan_count() -> None:
    """``max_replans_per_mission`` caps replans before failing."""
    # Stall sequence: each block of stall_window_ticks=2 triggers one replan.
    vlm = _StubVLM([0.0] * 10)
    replanner = _StubReplanner(
        [GoalVector(), GoalVector(), None]  # 3rd request never used
    )
    lifecycle = MissionLifecycle(
        _cfg(replan_enabled=True, stall_window_ticks=2, max_replans_per_mission=2),
        vlm_progress=vlm,
        replanner=replanner,
    )
    lifecycle.start_mission("mission-1", "goal")
    # 2 ticks → first replan (count=1)
    await lifecycle.tick(_obs(), _obs())
    await lifecycle.tick(_obs(), _obs())
    assert lifecycle.replan_count == 1
    # 2 more ticks → second replan (count=2)
    await lifecycle.tick(_obs(), _obs())
    await lifecycle.tick(_obs(), _obs())
    assert lifecycle.replan_count == 2
    # 2 more ticks → third attempted replan exceeds limit → FAILED
    await lifecycle.tick(_obs(), _obs())
    result = await lifecycle.tick(_obs(), _obs())
    assert result.state == MissionLifecycleState.FAILED
    assert result.reason == "replan_limit_exceeded"


@pytest.mark.asyncio
async def test_lifecycle_emits_state_transition_metrics() -> None:
    """Every state transition increments the labeled-pair counter."""
    from mousedroid.config.schema import MetricsConfig
    from mousedroid.telemetry.metrics import MetricsRegistry

    metrics = MetricsRegistry(MetricsConfig())
    vlm = _StubVLM([0.95])
    lifecycle = MissionLifecycle(_cfg(), vlm_progress=vlm, metrics=metrics)
    lifecycle.start_mission("mission-1", "goal")
    await lifecycle.tick(_obs(), _obs())
    rendered = metrics.render_prometheus()
    assert 'from_state="pending",to_state="running"' in rendered
    assert 'from_state="running",to_state="succeeded"' in rendered
    # Terminal SUCCEEDED also records the active duration histogram.
    assert "mission_active_duration_seconds_count 1" in rendered


# ---------------------------------------------------------------------------
# Factory wiring — build_mission_lifecycle gates on cfg.mission.replan_enabled
# ---------------------------------------------------------------------------


def test_build_mission_lifecycle_returns_none_when_disabled() -> None:
    """build_mission_lifecycle returns None when cfg.mission.replan_enabled=False."""
    from mousedroid.config.schema import Settings
    from mousedroid.factory import build_mission_lifecycle

    cfg = Settings(mock_hardware=True)
    assert cfg.mission.replan_enabled is False
    assert build_mission_lifecycle(cfg) is None


def test_build_mission_lifecycle_returns_lifecycle_when_enabled() -> None:
    """build_mission_lifecycle returns a MissionLifecycle when enabled."""
    from mousedroid.config.schema import Settings
    from mousedroid.factory import build_mission_lifecycle

    cfg = Settings(mock_hardware=True)
    cfg.mission.replan_enabled = True
    lifecycle = build_mission_lifecycle(cfg)
    assert lifecycle is not None
    assert isinstance(lifecycle, MissionLifecycle)


def test_build_mission_lifecycle_threads_dependencies() -> None:
    """build_mission_lifecycle wires task_tracker / vlm_progress / replanner / metrics."""
    from mousedroid.config.schema import MetricsConfig, Settings
    from mousedroid.factory import build_mission_lifecycle
    from mousedroid.telemetry.metrics import MetricsRegistry

    cfg = Settings(mock_hardware=True)
    cfg.mission.replan_enabled = True
    metrics = MetricsRegistry(MetricsConfig())
    vlm = _StubVLM([0.9])
    lifecycle = build_mission_lifecycle(
        cfg,
        task_tracker=None,
        vlm_progress=vlm,  # type: ignore[arg-type]
        replanner=None,
        metrics=metrics,
    )
    assert lifecycle is not None
    assert isinstance(lifecycle, MissionLifecycle)
