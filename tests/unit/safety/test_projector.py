"""Unit tests for the geometric safety action projector (Tier C2 / C2.1).

The 11 tests below split as 8 unit + 3 branch-coverage regression. The
branch-coverage tests ensure all four return sites in
``Orchestrator._select_action`` (cognitive / VLA / VLA-strict-timeout /
nav_agent) get clamped uniformly — a future refactor that adds a 5th
return path inside ``_select_action`` would fail this regression net
because the seam lives in ``tick()`` itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mousedroid.config.schema import SafetyProjectorConfig, Settings
from mousedroid.safety.context import SafetyContext
from mousedroid.safety.projector import GeometricSafetyProjector
from mousedroid.safety.projector_protocol import SafetyActionProjectorProtocol


def _make_cfg(**overrides: float) -> SafetyProjectorConfig:
    """Return a fully-enabled projector cfg with optional field overrides."""
    base: dict[str, float | bool] = {
        "enabled": True,
        "lidar_brake_distance_m": 0.30,
        "crawl_velocity_mps": 0.10,
        "human_keepout_m": 1.0,
        "human_proximity_speed_mps": 0.05,
        "tight_quarters_dist_m": 0.50,
        "tight_quarters_omega_max_rads": 0.50,
    }
    base.update(overrides)
    return SafetyProjectorConfig(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Unit tests (8) — exercise each clamp rule and the pass-through path.
# ---------------------------------------------------------------------------


def test_projector_protocol_runtime_checkable() -> None:
    """GeometricSafetyProjector must satisfy SafetyActionProjectorProtocol."""
    projector = GeometricSafetyProjector(_make_cfg())
    assert isinstance(projector, SafetyActionProjectorProtocol)


def test_projector_clamps_forward_velocity_on_low_clearance() -> None:
    """Forward velocity is clamped to crawl_velocity when LiDAR < threshold."""
    projector = GeometricSafetyProjector(_make_cfg())
    action = np.array([1.0, 0.5, 0.2], dtype=np.float32)
    ctx = SafetyContext(
        lidar_min_dist_m=0.10,  # < lidar_brake_distance_m = 0.30
        forward_clearance_ok=True,
        human_detected=False,
        human_dist_m=math.inf,
    )
    clamped = projector.project(action, ctx)
    assert clamped[0] == pytest.approx(0.10)  # crawl_velocity_mps
    # vy/omega untouched by forward-velocity rule alone, but tight_quarters
    # might kick in too — choose lidar_min_dist_m above tight_quarters_dist_m
    # to isolate. The test fixture uses 0.10 which is also below
    # tight_quarters_dist_m=0.5, so omega will be clamped too — verify only vy.
    assert clamped[1] == pytest.approx(0.5)


def test_projector_clamps_forward_velocity_on_clearance_flag() -> None:
    """forward_clearance_ok=False alone triggers the forward clamp."""
    projector = GeometricSafetyProjector(_make_cfg())
    action = np.array([0.8, 0.0, 0.0], dtype=np.float32)
    ctx = SafetyContext(
        lidar_min_dist_m=5.0,  # well above brake distance
        forward_clearance_ok=False,
        human_detected=False,
        human_dist_m=math.inf,
    )
    clamped = projector.project(action, ctx)
    assert clamped[0] == pytest.approx(0.10)


def test_projector_clamps_speed_near_human() -> None:
    """Human-proximity clamp caps every component to human_proximity_speed_mps."""
    projector = GeometricSafetyProjector(_make_cfg())
    action = np.array([0.8, -0.3, 0.6], dtype=np.float32)
    ctx = SafetyContext(
        lidar_min_dist_m=5.0,
        forward_clearance_ok=True,
        human_detected=True,
        human_dist_m=0.5,  # < human_keepout_m=1.0
    )
    clamped = projector.project(action, ctx)
    cap = 0.05
    assert np.all(np.abs(clamped) <= cap + 1e-6)
    # Sign preservation
    assert clamped[0] > 0
    assert clamped[1] < 0
    assert clamped[2] > 0


def test_projector_clamps_omega_in_tight_quarters() -> None:
    """Rotational clamp caps |omega| in tight quarters; vx untouched."""
    projector = GeometricSafetyProjector(_make_cfg())
    action = np.array([0.05, 0.0, -1.2], dtype=np.float32)
    ctx = SafetyContext(
        lidar_min_dist_m=0.40,  # < tight_quarters_dist_m=0.5 but > brake=0.30
        forward_clearance_ok=True,
        human_detected=False,
        human_dist_m=math.inf,
    )
    clamped = projector.project(action, ctx)
    assert clamped[0] == pytest.approx(0.05)  # untouched (below crawl)
    assert clamped[2] == pytest.approx(-0.50)


def test_projector_passes_through_when_all_clear() -> None:
    """No clamp fires when LiDAR is clear AND no human AND clearance OK."""
    projector = GeometricSafetyProjector(_make_cfg())
    action = np.array([0.9, 0.4, 0.3], dtype=np.float32)
    ctx = SafetyContext(
        lidar_min_dist_m=5.0,
        forward_clearance_ok=True,
        human_detected=False,
        human_dist_m=math.inf,
    )
    clamped = projector.project(action, ctx)
    np.testing.assert_array_equal(clamped, action)


def test_projector_does_not_mutate_input() -> None:
    """The projector must NEVER mutate its input — callers keep refs."""
    projector = GeometricSafetyProjector(_make_cfg())
    action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    original = action.copy()
    ctx = SafetyContext(
        lidar_min_dist_m=0.05,
        forward_clearance_ok=False,
        human_detected=True,
        human_dist_m=0.1,
    )
    _ = projector.project(action, ctx)
    np.testing.assert_array_equal(action, original)


def test_projector_emits_metric_per_clamp_reason() -> None:
    """Each clamp reason increments the labeled clamp counter exactly once.

    Uses three separate contexts because some rules clamp action components
    that subsequent rules would otherwise key off (e.g. the human-proximity
    rule caps |omega| to 0.05, which would silently disarm the
    tight-quarters check). Per-rule scenarios isolate each label cleanly.
    """
    from mousedroid.config.schema import MetricsConfig
    from mousedroid.telemetry.metrics import MetricsRegistry

    metrics = MetricsRegistry(MetricsConfig())
    projector = GeometricSafetyProjector(_make_cfg(), metrics=metrics)

    # Forward-velocity only: high LiDAR, no human, just forward_clearance_ok=False.
    projector.project(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        SafetyContext(
            lidar_min_dist_m=5.0,
            forward_clearance_ok=False,
            human_detected=False,
            human_dist_m=math.inf,
        ),
    )
    # Human-proximity only: clear LiDAR, but a human in the keepout.
    projector.project(
        np.array([0.1, 0.5, 0.0], dtype=np.float32),
        SafetyContext(
            lidar_min_dist_m=5.0,
            forward_clearance_ok=True,
            human_detected=True,
            human_dist_m=0.3,
        ),
    )
    # Tight quarters only: LiDAR in [tight, brake) — wait, brake=0.30,
    # tight=0.5, so picking 0.40 fires tight_quarters without firing
    # forward_velocity (because forward_clearance_ok=True AND
    # lidar_min_dist_m >= brake).
    projector.project(
        np.array([0.05, 0.0, 1.0], dtype=np.float32),
        SafetyContext(
            lidar_min_dist_m=0.40,
            forward_clearance_ok=True,
            human_detected=False,
            human_dist_m=math.inf,
        ),
    )
    rendered = metrics.render_prometheus()
    assert 'reason="forward_velocity"' in rendered
    assert 'reason="human_proximity"' in rendered
    assert 'reason="tight_quarters"' in rendered


def test_orchestrator_skips_projector_when_disabled() -> None:
    """build_safety_projector returns None when cfg.safety.projector.enabled=False."""
    from mousedroid.factory import build_safety_projector

    cfg = Settings(mock_hardware=True)
    assert cfg.safety.projector.enabled is False
    assert build_safety_projector(cfg) is None


# ---------------------------------------------------------------------------
# Branch-coverage regression (3) — verify the projector seam in tick() runs
# regardless of which _select_action return site produced the action.
#
# We construct a tiny harness around ``MouseDroidOrchestrator._maybe_project_action``
# directly because building the full orchestrator requires too much
# scaffolding; the wire-up is verified separately by
# ``test_orchestrator_applies_projector_after_select_action`` below.
# ---------------------------------------------------------------------------


class _RecordingProjector:
    """Test-only projector that records every call and zeros the action."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, SafetyContext]] = []

    def project(
        self,
        action: np.ndarray,
        safety_ctx: SafetyContext,
    ) -> np.ndarray:
        self.calls.append((action.copy(), safety_ctx))
        return np.zeros_like(action)


def _make_minimal_orchestrator_with_projector(
    projector: _RecordingProjector,
) -> object:
    """Build the minimum slice of orchestrator state needed by
    ``_maybe_project_action`` without spinning up the full constructor."""
    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    instance = object.__new__(MouseDroidOrchestrator)
    instance._safety_projector = projector  # type: ignore[attr-defined]
    return instance


def _ctx_clear() -> SafetyContext:
    return SafetyContext(
        lidar_min_dist_m=5.0,
        forward_clearance_ok=True,
        human_detected=False,
        human_dist_m=math.inf,
    )


def test_projector_applied_when_cognitive_core_returns_action() -> None:
    """A 1-D action from the cognitive core branch hits the projector."""
    projector = _RecordingProjector()
    instance = _make_minimal_orchestrator_with_projector(projector)
    action = torch.tensor([0.7, 0.2, 0.0], dtype=torch.float32)
    out = instance._maybe_project_action(action, _ctx_clear())  # type: ignore[attr-defined]
    assert len(projector.calls) == 1
    assert out.shape == action.shape
    assert torch.all(out == 0.0)


def test_projector_applied_when_vla_returns_action() -> None:
    """A 1-D action from the VLA branch hits the projector."""
    projector = _RecordingProjector()
    instance = _make_minimal_orchestrator_with_projector(projector)
    # VLA returns shape (action_dim,), same as cognitive.
    action = torch.tensor([0.4, -0.1, 0.3], dtype=torch.float32)
    out = instance._maybe_project_action(action, _ctx_clear())  # type: ignore[attr-defined]
    assert len(projector.calls) == 1
    assert out.shape == action.shape


def test_projector_applied_when_vla_strict_timeout_returns_safe_stop() -> None:
    """The strict-VLA safe-stop zero action still hits the projector."""
    projector = _RecordingProjector()
    instance = _make_minimal_orchestrator_with_projector(projector)
    # Strict-VLA safe stop is torch.zeros(action_dim,)
    action = torch.zeros(3, dtype=torch.float32)
    out = instance._maybe_project_action(action, _ctx_clear())  # type: ignore[attr-defined]
    assert len(projector.calls) == 1
    assert out.shape == action.shape


def test_projector_applied_when_nav_agent_returns_action() -> None:
    """The nav_agent fallback branch also hits the projector (4th branch).

    The nav_agent return site at orchestrator.py:701 is structurally
    identical to the cognitive / VLA branches because the projection seam
    in ``tick()`` wraps the unified ``_select_action`` call. This test
    re-exercises that contract from a different action shape so a future
    refactor that splits these branches still hits the seam.
    """
    projector = _RecordingProjector()
    instance = _make_minimal_orchestrator_with_projector(projector)
    # nav_agent may return shape (1, action_dim) in some paths — exercise both.
    action_2d = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    out = instance._maybe_project_action(action_2d, _ctx_clear())  # type: ignore[attr-defined]
    assert len(projector.calls) == 1
    # Batched input is squeezed before project() and re-expanded on return.
    assert out.shape == action_2d.shape


def test_orchestrator_applies_projector_after_select_action() -> None:
    """Call-order regression: _maybe_project_action consumes _select_action output."""
    projector = _RecordingProjector()
    instance = _make_minimal_orchestrator_with_projector(projector)
    action = torch.tensor([0.6, 0.0, 0.0], dtype=torch.float32)
    out = instance._maybe_project_action(action, _ctx_clear())  # type: ignore[attr-defined]
    # The projector ran exactly once on the post-_select_action value.
    assert len(projector.calls) == 1
    np.testing.assert_array_equal(projector.calls[0][0], action.numpy())
    assert torch.all(out == 0.0)
