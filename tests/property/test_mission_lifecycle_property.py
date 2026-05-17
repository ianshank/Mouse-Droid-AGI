"""Property: MissionLifecycle never makes illegal state transitions."""

from __future__ import annotations

import asyncio

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.config.schema import MissionConfig
from mousedroid.orchestrator.mission_lifecycle import (
    MissionLifecycle,
    MissionLifecycleState,
)

_VALID_TRANSITIONS: dict[MissionLifecycleState, set[MissionLifecycleState]] = {
    MissionLifecycleState.PENDING: {
        MissionLifecycleState.RUNNING,
        MissionLifecycleState.FAILED,
    },
    MissionLifecycleState.RUNNING: {
        MissionLifecycleState.RUNNING,
        MissionLifecycleState.SUCCEEDED,
        MissionLifecycleState.REPLANNING,
        MissionLifecycleState.FAILED,
    },
    MissionLifecycleState.REPLANNING: {
        MissionLifecycleState.RUNNING,
        MissionLifecycleState.FAILED,
    },
    MissionLifecycleState.SUCCEEDED: {MissionLifecycleState.SUCCEEDED},  # terminal
    MissionLifecycleState.FAILED: {MissionLifecycleState.FAILED},  # terminal
}


class _MutableVLM:
    """Stub VLM head whose returned score can be mutated between ticks."""

    def __init__(self) -> None:
        self.score_value: float = 0.0

    def score(self, *args: object, **kwargs: object) -> torch.Tensor:
        return torch.tensor([[self.score_value]])


async def _walk_transitions(scores: list[float]) -> None:
    """Drive ONE lifecycle through a sequence of scores; assert legal transitions."""
    cfg = MissionConfig(
        replan_enabled=False,
        success_threshold=0.9,
        stall_threshold=0.1,
        stall_window_ticks=3,
        max_replans_per_mission=0,
    )
    obs = torch.zeros(1, 4)
    # ONE lifecycle persists across all scores — state carries forward, so the
    # property test covers SUCCEEDED/FAILED stickiness and stall accumulation
    # across consecutive ticks (which a per-score fresh lifecycle would miss).
    vlm = _MutableVLM()
    lc = MissionLifecycle(cfg, vlm_progress=vlm)
    lc.start_mission("m1", "goal")
    prev_state = lc.current_state
    assert prev_state is not None
    for s in scores:
        vlm.score_value = s
        result = await lc.tick(obs, obs)
        assert (
            result.state in _VALID_TRANSITIONS[prev_state]
        ), f"Illegal transition {prev_state} -> {result.state} at score={s}"
        prev_state = result.state


@given(scores=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=50))
@settings(deadline=None, max_examples=100)
def test_lifecycle_never_makes_illegal_transitions(scores: list[float]) -> None:
    """Property: any sequence of VLM scores in [0,1] produces only legal state transitions."""
    asyncio.run(_walk_transitions(scores))
