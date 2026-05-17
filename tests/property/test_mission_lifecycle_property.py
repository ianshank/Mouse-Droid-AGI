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


# ---------------------------------------------------------------------------
# Tier C2.3 — absorbing terminal state under random LLM ready schedules
# ---------------------------------------------------------------------------


async def _walk_with_replanner(scores: list[float], ready_schedule: list[bool]) -> None:
    """Drive a lifecycle with a scripted VLM + a scripted LLM ready schedule.

    Asserts that once the lifecycle reaches SUCCEEDED or FAILED, no
    subsequent tick advances it. Mirrors the safety invariant the
    Tier C2.3 wiring depends on for correctness: a terminal mission
    must remain terminal regardless of follow-on VLM scores or
    transient LLM-gateway flapping.
    """
    from unittest.mock import AsyncMock

    from mousedroid.config.schema import MetricsConfig, MissionReplannerConfig
    from mousedroid.llm_gateway.protocol import GoalVector
    from mousedroid.orchestrator.llm_replanner import LLMGatewayMissionReplanner
    from mousedroid.telemetry.metrics import MetricsRegistry

    cfg = MissionConfig(
        replan_enabled=True,
        success_threshold=0.7,
        stall_threshold=0.2,
        stall_window_ticks=2,
        max_replans_per_mission=3,
    )

    class _ScriptedGateway:
        def __init__(self, ready_seq: list[bool]) -> None:
            self._ready_seq = list(ready_seq)
            self._idx = 0
            # ``translate_mission`` always returns a fresh non-zero goal so the
            # success path of the lifecycle's replan branch is exercised.
            self.translate_mission = AsyncMock(
                return_value=GoalVector(vx_target=0.5),
            )

        @property
        def is_ready(self) -> bool:
            r = self._ready_seq[min(self._idx, len(self._ready_seq) - 1)]
            self._idx += 1
            return r

    replanner = LLMGatewayMissionReplanner(
        gateway=_ScriptedGateway(ready_schedule),  # type: ignore[arg-type]
        cfg=MissionReplannerConfig(),
        metrics=MetricsRegistry(MetricsConfig()),
    )
    vlm = _MutableVLM()
    lc = MissionLifecycle(cfg, vlm_progress=vlm, replanner=replanner)
    lc.start_mission("p1", "explore")

    obs = torch.zeros(1, 8)
    for s in scores:
        vlm.score_value = s
        result = await lc.tick(obs, obs)
        if result.state in {
            MissionLifecycleState.SUCCEEDED,
            MissionLifecycleState.FAILED,
        }:
            terminal = result.state
            # Subsequent ticks must NOT leave the terminal state.
            for _ in range(3):
                next_result = await lc.tick(obs, obs)
                assert (
                    next_result.state == terminal
                ), f"Terminal state {terminal} leaked to {next_result.state}"
            return


@given(
    scores=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=20),
    ready_schedule=st.lists(st.booleans(), min_size=1, max_size=20),
)
@settings(deadline=None, max_examples=50)
def test_terminal_state_is_absorbing_under_any_score_and_ready_schedule(
    scores: list[float],
    ready_schedule: list[bool],
) -> None:
    """Tier C2.3: once the lifecycle reaches SUCCEEDED or FAILED, no tick moves it."""
    asyncio.run(_walk_with_replanner(scores, ready_schedule))
