"""Tier C2.3: mission_replan_llm_calls counter wiring."""

from __future__ import annotations

from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry.metrics import MetricsRegistry


def test_counter_increments_per_outcome() -> None:
    registry = MetricsRegistry(MetricsConfig())
    registry.inc_mission_replan_llm("ok")
    registry.inc_mission_replan_llm("ok")
    registry.inc_mission_replan_llm("degraded")
    registry.inc_mission_replan_llm("exception")

    payload = registry.render_prometheus()
    assert 'mission_replan_llm_calls_total{outcome="ok"} 2' in payload
    assert 'mission_replan_llm_calls_total{outcome="degraded"} 1' in payload
    assert 'mission_replan_llm_calls_total{outcome="exception"} 1' in payload


def test_counter_zero_or_negative_amount_is_noop() -> None:
    """``amount<=0`` mirrors the existing ``inc_*`` helper contract."""
    registry = MetricsRegistry(MetricsConfig())
    registry.inc_mission_replan_llm("ok", amount=0)
    registry.inc_mission_replan_llm("ok", amount=-3)
    payload = registry.render_prometheus()
    # No writes occurred → the counter family doesn't surface yet.
    assert "mission_replan_llm_calls_total" not in payload


def test_counter_is_namespaced_consistently_with_existing_metrics() -> None:
    """The counter respects ``cfg.namespace`` like every other metric."""
    cfg = MetricsConfig()
    registry = MetricsRegistry(cfg)
    registry.inc_mission_replan_llm("ok")
    payload = registry.render_prometheus()
    assert f"{cfg.namespace}_mission_replan_llm_calls_total" in payload
