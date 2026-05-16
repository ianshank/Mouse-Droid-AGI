"""End-to-end integration test for the mission closed-loop (Tier C2 / C2.4).

Validates the full mission lifecycle against mock VLM + mock LLM
replanner: a multi-minute simulated mission with stalls + replans must
end in SUCCEEDED with all relevant Prometheus counters populated.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from mousedroid.config.schema import MetricsConfig, MissionConfig
from mousedroid.llm_gateway.protocol import GoalVector
from mousedroid.orchestrator.mission_lifecycle import (
    MissionLifecycle,
    MissionLifecycleState,
)
from mousedroid.telemetry.metrics import MetricsRegistry


class _ScriptedVLM:
    """VLM stub that returns a predetermined progress trajectory."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)

    def score(
        self,
        prev_obs: torch.Tensor,
        curr_obs: torch.Tensor,
        *,
        instruction: str | None = None,
    ) -> torch.Tensor:
        del prev_obs, curr_obs, instruction
        v = self._scores.pop(0) if self._scores else 0.0
        return torch.tensor([[float(v)]], dtype=torch.float32)


class _ScriptedReplanner:
    """LLM replanner stub returning a queued response per call."""

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
        return self._responses.pop(0) if self._responses else None


def _obs() -> torch.Tensor:
    return torch.zeros(1, 4)


@pytest.mark.asyncio
async def test_rising_progress_completes_mission_in_succeeded() -> None:
    """A multi-minute simulated mission with strictly rising progress ends SUCCEEDED."""
    # 600 ticks at 30 Hz ≈ 20 s, but a single 0.95 score finalises it
    # — keep the early progress middling so we exercise the RUNNING loop
    # for many ticks before crossing the success threshold.
    scores = [0.3] * 100 + [0.6] * 100 + [0.95]
    vlm = _ScriptedVLM(scores)
    cfg = MissionConfig(replan_enabled=False, success_threshold=0.9)
    metrics = MetricsRegistry(MetricsConfig())
    lifecycle = MissionLifecycle(cfg, vlm_progress=vlm, metrics=metrics)
    lifecycle.start_mission("mission-rising", "go to the kitchen")

    result = None
    for _ in range(250):
        result = await lifecycle.tick(_obs(), _obs())
        if result.state == MissionLifecycleState.SUCCEEDED:
            break

    assert result is not None
    assert result.state == MissionLifecycleState.SUCCEEDED
    rendered = metrics.render_prometheus()
    assert 'from_state="pending",to_state="running"' in rendered
    assert 'from_state="running",to_state="succeeded"' in rendered
    assert "mission_active_duration_seconds_count 1" in rendered


@pytest.mark.asyncio
async def test_stall_then_llm_replan_then_succeed() -> None:
    """Stall → REPLANNING → LLM replan → resume RUNNING → SUCCEEDED."""
    # Tick sequence:
    # ticks 0-9 → progress 0.0 (stall accumulator runs up)
    # On tick 9 (stall_window_ticks=10) replan fires.
    # After replan, progress reset to 0.5 for 10 ticks, then 0.95 to finish.
    stall_window = 10
    scores = [0.0] * stall_window + [0.5] * 10 + [0.95]
    vlm = _ScriptedVLM(scores)
    replanner = _ScriptedReplanner([GoalVector(vx_target=0.5, omega_target=0.1)])
    cfg = MissionConfig(
        replan_enabled=True,
        success_threshold=0.9,
        stall_threshold=0.1,
        stall_window_ticks=stall_window,
        max_replans_per_mission=3,
    )
    metrics = MetricsRegistry(MetricsConfig())
    lifecycle = MissionLifecycle(cfg, vlm_progress=vlm, replanner=replanner, metrics=metrics)
    lifecycle.start_mission("mission-stall-then-recover", "patrol the corridor")

    final_state = None
    saw_replanning = False
    for _ in range(100):
        result = await lifecycle.tick(_obs(), _obs())
        if result.state == MissionLifecycleState.REPLANNING:
            saw_replanning = True
        final_state = result.state
        if final_state in (MissionLifecycleState.SUCCEEDED, MissionLifecycleState.FAILED):
            break

    assert final_state == MissionLifecycleState.SUCCEEDED
    assert lifecycle.replan_count == 1
    assert len(replanner.calls) == 1

    rendered = metrics.render_prometheus()
    assert 'from_state="running",to_state="replanning"' in rendered
    assert 'from_state="replanning",to_state="running"' in rendered
    assert 'from_state="running",to_state="succeeded"' in rendered
    assert 'outcome="succeeded"' in rendered

    # The lifecycle may transition through REPLANNING in the same tick it
    # consumed the stall window — saw_replanning may be False because the
    # final state of the tick is RUNNING. The labeled-pair counter check
    # above is the authoritative signal that the transition occurred.
    del saw_replanning  # intentionally informational


@pytest.mark.asyncio
async def test_repeated_stalls_exceed_replan_limit_fails_mission() -> None:
    """Mission fails after ``max_replans_per_mission`` exhausted replans."""
    stall_window = 5
    # Constant 0.0 progress — every block of stall_window ticks triggers a replan.
    vlm = _ScriptedVLM([0.0] * 200)
    replanner = _ScriptedReplanner([GoalVector(), GoalVector()])  # exhausted after 2
    cfg = MissionConfig(
        replan_enabled=True,
        success_threshold=0.9,
        stall_threshold=0.1,
        stall_window_ticks=stall_window,
        max_replans_per_mission=2,
    )
    metrics = MetricsRegistry(MetricsConfig())
    lifecycle = MissionLifecycle(cfg, vlm_progress=vlm, replanner=replanner, metrics=metrics)
    lifecycle.start_mission("mission-limit", "explore endlessly")

    final_state = None
    for _ in range(100):
        result = await lifecycle.tick(_obs(), _obs())
        final_state = result.state
        if final_state == MissionLifecycleState.FAILED:
            break

    assert final_state == MissionLifecycleState.FAILED
    # The two successful replans incremented the counter; the third
    # attempt tripped the limit guard and recorded a "failed" replan.
    rendered = metrics.render_prometheus()
    assert 'outcome="succeeded"' in rendered
    assert 'outcome="failed"' in rendered
