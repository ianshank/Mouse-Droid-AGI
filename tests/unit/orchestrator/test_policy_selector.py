"""Phase 3a: orchestrator policy selector / VLA branch tests.

These tests exercise ``MouseDroidOrchestrator._select_action`` directly
across all valid ``cfg.loop.policy_selector`` values and confirm:

* Default ``"nav_agent"`` preserves the legacy nav-agent action even
  when a VLA policy is wired in.
* ``"vla"`` and ``"auto"`` route through the VLA policy.
* Inference timeouts trigger the configured fallback strategy.
* Predict-time exceptions never escape the loop.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import torch

from mousedroid.config.schema import Settings, VLAConfig
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext
from mousedroid.sensing.bundle import MouseDroidObservationBundle
from mousedroid.vla.policy import MockVLA, VLAAction, VLAObservation


def _make_observation(cfg: Settings) -> MouseDroidObservationBundle:
    return MouseDroidObservationBundle(
        _timestamp=0.0,
        _vision_features=np.zeros(cfg.camera.feature_dim, dtype=np.float32),
        _distance_m=1.5,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _audio_chunk=np.zeros(1024, dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )


def _make_orchestrator(
    *,
    cfg: Settings,
    nav_action: torch.Tensor,
    vla_policy: object | None = None,
) -> MouseDroidOrchestrator:
    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, cfg.model.hidden_dim),
        0.1,
    )
    agent = MagicMock()
    agent.name = "nav_agent_under_test"
    agent.act.return_value = nav_action

    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = SafetyContext(is_emergency=False)

    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=AsyncMock(),
        sensor_manager=AsyncMock(),
        cfg=cfg,
        vla_policy=vla_policy,  # type: ignore[arg-type]
    )


def _ctx() -> SafetyContext:
    return SafetyContext(is_emergency=False)


class TestPolicySelectorDefault:
    def test_default_selector_is_nav_agent(self) -> None:
        cfg = Settings(mock_hardware=True)
        assert cfg.loop.policy_selector == "nav_agent"

    def test_default_inference_timeout_is_none(self) -> None:
        cfg = Settings(mock_hardware=True)
        assert cfg.loop.inference_timeout_s is None

    def test_default_vla_backend_is_none(self) -> None:
        cfg = Settings(mock_hardware=True)
        assert cfg.vla.backend == "none"


class TestNavAgentPathPreserved:
    def test_no_vla_policy_uses_nav_agent(self) -> None:
        cfg = Settings(mock_hardware=True)
        nav = torch.tensor([0.5, -0.25, 0.1])
        orch = _make_orchestrator(cfg=cfg, nav_action=nav, vla_policy=None)
        out = orch._select_action(_ctx(), _make_observation(cfg), 0.0)
        assert torch.equal(out, nav)

    def test_nav_agent_selector_ignores_vla_policy(self) -> None:
        # Even with a wired VLA policy returning a distinct action, the
        # default selector must not consult it.
        cfg = Settings(mock_hardware=True)
        cfg.vla = VLAConfig(backend="mock")
        nav = torch.tensor([0.5, -0.25, 0.1])
        vla = MockVLA(
            action_dim=cfg.model.action_dim,
            canned_action=torch.tensor([0.9, 0.9, 0.9]),
        )
        orch = _make_orchestrator(cfg=cfg, nav_action=nav, vla_policy=vla)
        out = orch._select_action(_ctx(), _make_observation(cfg), 0.0)
        assert torch.equal(out, nav)


class TestVLASelector:
    def test_vla_selector_uses_vla_action(self) -> None:
        cfg = Settings(mock_hardware=True)
        cfg.loop.policy_selector = "vla"
        canned = torch.tensor([0.1, 0.2, 0.3])
        vla = MockVLA(action_dim=cfg.model.action_dim, canned_action=canned)
        orch = _make_orchestrator(cfg=cfg, nav_action=torch.zeros(3), vla_policy=vla)
        out = orch._select_action(_ctx(), _make_observation(cfg), 0.0)
        assert torch.allclose(out, canned)

    def test_auto_selector_uses_vla_when_present(self) -> None:
        cfg = Settings(mock_hardware=True)
        cfg.loop.policy_selector = "auto"
        canned = torch.tensor([0.4, 0.5, 0.6])
        vla = MockVLA(action_dim=cfg.model.action_dim, canned_action=canned)
        orch = _make_orchestrator(cfg=cfg, nav_action=torch.zeros(3), vla_policy=vla)
        out = orch._select_action(_ctx(), _make_observation(cfg), 0.0)
        assert torch.allclose(out, canned)

    def test_auto_selector_without_vla_falls_back(self) -> None:
        cfg = Settings(mock_hardware=True)
        cfg.loop.policy_selector = "auto"
        nav = torch.tensor([0.7, 0.0, 0.0])
        orch = _make_orchestrator(cfg=cfg, nav_action=nav, vla_policy=None)
        out = orch._select_action(_ctx(), _make_observation(cfg), 0.0)
        assert torch.equal(out, nav)


class _SlowVLA:
    """Test double that sleeps longer than the configured timeout."""

    name = "slow"

    def __init__(self, action_dim: int, sleep_s: float) -> None:
        self._dim = action_dim
        self._sleep_s = sleep_s

    def predict(self, observation: VLAObservation) -> VLAAction:
        del observation
        time.sleep(self._sleep_s)
        return VLAAction(action=torch.ones(self._dim))


class _ExplodingVLA:
    name = "boom"

    def predict(self, observation: VLAObservation) -> VLAAction:
        raise RuntimeError("boom")


class TestVLATimeoutAndFallback:
    def test_auto_falls_back_to_nav_on_timeout(self) -> None:
        cfg = Settings(mock_hardware=True)
        cfg.loop.policy_selector = "auto"
        cfg.loop.inference_timeout_s = 0.001
        nav = torch.tensor([0.2, 0.0, 0.0])
        slow = _SlowVLA(action_dim=cfg.model.action_dim, sleep_s=0.05)
        orch = _make_orchestrator(cfg=cfg, nav_action=nav, vla_policy=slow)
        out = orch._select_action(_ctx(), _make_observation(cfg), 0.0)
        assert torch.equal(out, nav)

    def test_strict_vla_safe_stop_on_timeout(self) -> None:
        cfg = Settings(mock_hardware=True)
        cfg.loop.policy_selector = "vla"
        cfg.loop.inference_timeout_s = 0.001
        cfg.vla = VLAConfig(backend="mock", fallback_on_timeout=False)
        slow = _SlowVLA(action_dim=cfg.model.action_dim, sleep_s=0.05)
        orch = _make_orchestrator(
            cfg=cfg,
            nav_action=torch.tensor([0.9, 0.9, 0.9]),
            vla_policy=slow,
        )
        out = orch._select_action(_ctx(), _make_observation(cfg), 0.0)
        assert torch.equal(out, torch.zeros(cfg.model.action_dim))

    def test_strict_vla_with_fallback_uses_nav_on_timeout(self) -> None:
        cfg = Settings(mock_hardware=True)
        cfg.loop.policy_selector = "vla"
        cfg.loop.inference_timeout_s = 0.001
        cfg.vla = VLAConfig(backend="mock", fallback_on_timeout=True)
        nav = torch.tensor([0.3, 0.3, 0.3])
        slow = _SlowVLA(action_dim=cfg.model.action_dim, sleep_s=0.05)
        orch = _make_orchestrator(cfg=cfg, nav_action=nav, vla_policy=slow)
        out = orch._select_action(_ctx(), _make_observation(cfg), 0.0)
        assert torch.equal(out, nav)

    def test_predict_exception_falls_back(self) -> None:
        cfg = Settings(mock_hardware=True)
        cfg.loop.policy_selector = "auto"
        nav = torch.tensor([0.4, 0.0, 0.0])
        orch = _make_orchestrator(cfg=cfg, nav_action=nav, vla_policy=_ExplodingVLA())
        out = orch._select_action(_ctx(), _make_observation(cfg), 0.0)
        assert torch.equal(out, nav)

    def test_action_shape_mismatch_falls_back(self) -> None:
        cfg = Settings(mock_hardware=True)
        cfg.loop.policy_selector = "auto"
        nav = torch.tensor([0.6, 0.0, 0.0])
        # MockVLA configured for a different action_dim than cfg.model.action_dim
        wrong = MockVLA(action_dim=cfg.model.action_dim + 1)
        orch = _make_orchestrator(cfg=cfg, nav_action=nav, vla_policy=wrong)
        out = orch._select_action(_ctx(), _make_observation(cfg), 0.0)
        assert torch.equal(out, nav)

    def test_default_budget_uses_one_over_control_hz(self) -> None:
        # control_hz=30 → ~33ms budget. A 1ms predict comfortably succeeds.
        cfg = Settings(mock_hardware=True)
        cfg.loop.policy_selector = "vla"
        cfg.loop.inference_timeout_s = None
        canned = torch.tensor([0.05, 0.0, 0.0])
        vla = MockVLA(action_dim=cfg.model.action_dim, canned_action=canned)
        orch = _make_orchestrator(cfg=cfg, nav_action=torch.zeros(3), vla_policy=vla)
        out = orch._select_action(_ctx(), _make_observation(cfg), 0.0)
        assert torch.allclose(out, canned)
