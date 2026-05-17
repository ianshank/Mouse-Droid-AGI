"""Tier C2.3: LLMGatewayMissionReplanner adapter unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from mousedroid.config.schema import MetricsConfig, MissionReplannerConfig
from mousedroid.llm_gateway.protocol import GoalVector
from mousedroid.orchestrator.llm_replanner import LLMGatewayMissionReplanner
from mousedroid.telemetry.metrics import MetricsRegistry


def _make_gateway(
    *,
    ready: bool,
    goal: GoalVector | None = None,
    raises: Exception | None = None,
) -> MagicMock:
    gw = MagicMock()
    type(gw).is_ready = PropertyMock(return_value=ready)
    if raises is not None:
        gw.translate_mission = AsyncMock(side_effect=raises)
    else:
        gw.translate_mission = AsyncMock(return_value=goal or GoalVector())
    return gw


@pytest.mark.asyncio
async def test_submit_replan_request_success_returns_goal_vector() -> None:
    metrics = MetricsRegistry(MetricsConfig())
    expected = GoalVector(vx_target=0.4, vy_target=0.0, omega_target=0.1)
    gw = _make_gateway(ready=True, goal=expected)
    adapter = LLMGatewayMissionReplanner(
        gateway=gw,
        cfg=MissionReplannerConfig(),
        metrics=metrics,
    )
    result = await adapter.submit_replan_request(
        mission_id="m-1",
        goal_text="navigate to charger",
        last_progress=0.1,
    )
    assert result == expected
    gw.translate_mission.assert_awaited_once()
    forwarded = gw.translate_mission.await_args.args[0]
    assert "navigate to charger" in forwarded
    assert "last_progress=0.10" in forwarded
    assert 'mission_replan_llm_calls_total{outcome="ok"} 1' in metrics.render_prometheus()


@pytest.mark.asyncio
async def test_submit_replan_request_returns_none_when_gateway_degraded() -> None:
    metrics = MetricsRegistry(MetricsConfig())
    gw = _make_gateway(ready=False)
    adapter = LLMGatewayMissionReplanner(
        gateway=gw,
        cfg=MissionReplannerConfig(),
        metrics=metrics,
    )
    result = await adapter.submit_replan_request(
        mission_id="m-2",
        goal_text="go to kitchen",
        last_progress=0.05,
    )
    assert result is None
    gw.translate_mission.assert_not_awaited()
    assert 'mission_replan_llm_calls_total{outcome="degraded"} 1' in metrics.render_prometheus()


@pytest.mark.asyncio
async def test_submit_replan_request_swallows_exception_and_returns_none() -> None:
    metrics = MetricsRegistry(MetricsConfig())
    gw = _make_gateway(ready=True, raises=RuntimeError("LLM timeout"))
    adapter = LLMGatewayMissionReplanner(
        gateway=gw,
        cfg=MissionReplannerConfig(),
        metrics=metrics,
    )
    result = await adapter.submit_replan_request(
        mission_id="m-3",
        goal_text="patrol perimeter",
        last_progress=0.2,
    )
    assert result is None
    assert 'mission_replan_llm_calls_total{outcome="exception"} 1' in metrics.render_prometheus()


@pytest.mark.asyncio
async def test_prompt_clipped_at_max_prompt_chars() -> None:
    metrics = MetricsRegistry(MetricsConfig())
    gw = _make_gateway(ready=True)
    adapter = LLMGatewayMissionReplanner(
        gateway=gw,
        cfg=MissionReplannerConfig(max_prompt_chars=32),
        metrics=metrics,
    )
    await adapter.submit_replan_request(
        mission_id="m-4",
        goal_text="x" * 500,
        last_progress=0.1,
    )
    forwarded = gw.translate_mission.await_args.args[0]
    assert len(forwarded) <= 32


@pytest.mark.asyncio
async def test_include_progress_in_prompt_false_omits_progress_hint() -> None:
    metrics = MetricsRegistry(MetricsConfig())
    gw = _make_gateway(ready=True)
    adapter = LLMGatewayMissionReplanner(
        gateway=gw,
        cfg=MissionReplannerConfig(include_progress_in_prompt=False),
        metrics=metrics,
    )
    await adapter.submit_replan_request(
        mission_id="m-5",
        goal_text="explore",
        last_progress=0.05,
    )
    forwarded = gw.translate_mission.await_args.args[0]
    assert "last_progress" not in forwarded
    assert forwarded.startswith("explore")


@pytest.mark.asyncio
async def test_metrics_optional_does_not_crash_when_omitted() -> None:
    gw = _make_gateway(ready=True)
    adapter = LLMGatewayMissionReplanner(
        gateway=gw,
        cfg=MissionReplannerConfig(),
    )
    result = await adapter.submit_replan_request(
        mission_id="m-6",
        goal_text="explore",
        last_progress=0.1,
    )
    assert result is not None
