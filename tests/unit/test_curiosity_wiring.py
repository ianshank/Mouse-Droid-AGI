"""Tests for curiosity wiring in orchestrator obs_dict."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch

from mousedroid.config.schema import Settings
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext
from mousedroid.sensing.bundle import MouseDroidObservationBundle


def _make_observation(cfg: Settings) -> MouseDroidObservationBundle:
    """Create a default observation bundle for testing."""
    return MouseDroidObservationBundle(
        _timestamp=0.0,
        _vision_features=np.zeros(cfg.camera.feature_dim, dtype=np.float32),
        _distance_m=1.5,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _audio_chunk=np.zeros(1024, dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )


def _make_orchestrator_with_curiosity(
    *,
    curiosity_module: object | None = None,
    memory_tier: object | None = None,
) -> MouseDroidOrchestrator:
    """Create orchestrator with optional curiosity module."""
    cfg = Settings(mock_hardware=True)

    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, cfg.model.hidden_dim),
        0.1,
    )

    agent = MagicMock()
    agent.name = "test_agent"
    agent.act.return_value = torch.tensor([0.1, 0.0, 0.0])

    safety_ctx = SafetyContext()
    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = safety_ctx

    esp32 = AsyncMock()
    sensor_manager = AsyncMock()
    sensor_manager.read_all.return_value = _make_observation(cfg)
    sensor_manager.recovery_attempt.return_value = 0

    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cfg=cfg,
        curiosity_module=curiosity_module,
        memory_tier=memory_tier,
    )


# ---------------------------------------------------------------------------
# Curiosity scores in obs_dict
# ---------------------------------------------------------------------------


def test_compute_curiosity_scores_no_module() -> None:
    """_compute_curiosity_scores returns zeros without curiosity module."""
    orch = _make_orchestrator_with_curiosity()
    scores = orch._compute_curiosity_scores()
    assert "intrinsic" in scores
    assert "epistemic" in scores
    assert scores["intrinsic"] == 0.0
    assert scores["epistemic"] == 0.0


def test_compute_curiosity_scores_with_module() -> None:
    """_compute_curiosity_scores returns intrinsic reward from ICM."""
    curiosity = MagicMock()
    curiosity.intrinsic_reward.return_value = torch.tensor([0.42])

    orch = _make_orchestrator_with_curiosity(curiosity_module=curiosity)
    scores = orch._compute_curiosity_scores()
    assert scores["intrinsic"] == pytest.approx(0.42, abs=1e-5)
    curiosity.intrinsic_reward.assert_called_once()


def test_compute_curiosity_scores_epistemic_from_semantic() -> None:
    """_compute_curiosity_scores computes epistemic from semantic index distance."""
    semantic = MagicMock()
    semantic.size = 5
    semantic.retrieve.return_value = [("key0", 1.5)]

    memory_tier = MagicMock()
    memory_tier.semantic = semantic

    orch = _make_orchestrator_with_curiosity(memory_tier=memory_tier)
    scores = orch._compute_curiosity_scores()
    assert scores["epistemic"] == pytest.approx(1.5, abs=1e-5)
    semantic.retrieve.assert_called_once()


def test_compute_curiosity_scores_empty_semantic() -> None:
    """_compute_curiosity_scores returns 0 epistemic when semantic index is empty."""
    semantic = MagicMock()
    semantic.size = 0

    memory_tier = MagicMock()
    memory_tier.semantic = semantic

    orch = _make_orchestrator_with_curiosity(memory_tier=memory_tier)
    scores = orch._compute_curiosity_scores()
    assert scores["epistemic"] == 0.0
    semantic.retrieve.assert_not_called()


async def test_curiosity_key_in_cognitive_obs_dict() -> None:
    """Cognitive core obs_dict contains 'curiosity' key with channel scores."""
    orch = _make_orchestrator_with_curiosity()

    # Set up cognitive core to capture obs_dict
    cognitive_core = MagicMock()
    cognitive_core.tick_fast = MagicMock(return_value=(np.array([0.1, 0.0, 0.0]), []))
    orch._cognitive_core = cognitive_core

    obs = _make_observation(orch._cfg)
    orch._try_cognitive_action(obs, 10.0)

    # Verify curiosity key is present
    call_args = cognitive_core.tick_fast.call_args[0][0]
    assert "curiosity" in call_args
    curiosity_scores = call_args["curiosity"]
    assert "intrinsic" in curiosity_scores
    assert "epistemic" in curiosity_scores


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_orchestrator_requires_at_least_one_agent() -> None:
    """Orchestrator raises ValueError when agents list is empty."""
    import pytest

    cfg = Settings(mock_hardware=True)
    with pytest.raises(ValueError, match="At least one agent"):
        MouseDroidOrchestrator(
            world_model=MagicMock(),
            agents=[],
            safety_monitor=MagicMock(),
            esp32=AsyncMock(),
            sensor_manager=AsyncMock(),
            cfg=cfg,
        )


def test_orchestrator_accepts_none_optional_components() -> None:
    """Orchestrator works with all optional components as None."""
    orch = _make_orchestrator_with_curiosity()
    assert orch._memory_tier is None
    assert orch._experience_logger is None
    assert orch._curiosity_module is None
    assert orch._consolidation_task is None
