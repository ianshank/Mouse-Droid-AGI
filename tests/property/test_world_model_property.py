from __future__ import annotations

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.config.schema import ModelConfig
from mousedroid.sensing.bundle import MouseDroidObservationBundle
from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM
from mousedroid.world_model.encoder import MultimodalEncoder
from mousedroid.world_model.rssm import RSSM


def _default_cfg() -> ModelConfig:
    return ModelConfig()


@given(
    scale=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20, deadline=1000)
def test_encoder_output_finite_for_finite_input(scale: float) -> None:
    cfg = _default_cfg()
    enc = MultimodalEncoder(cfg)
    vision = torch.full((1, cfg.vision_dim), scale)
    ultrasonic = torch.full((1, cfg.ultrasonic_dim), scale)
    motor = torch.full((1, cfg.motor_state_dim), scale)
    mask = torch.ones(1, 4)
    out = enc(vision, ultrasonic, motor, mask)
    assert torch.isfinite(out).all()


@given(
    dist=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=10)
def test_rssm_observe_step_returns_finite(dist: float) -> None:
    cfg = _default_cfg()
    rssm = RSSM(cfg)
    rssm.eval()
    obs = MouseDroidObservationBundle(
        _vision_features=np.zeros(cfg.vision_dim, dtype=np.float32),
        _distance_m=dist,
        _motor_state=np.zeros(cfg.motor_state_dim, dtype=np.float32),
        _valid_mask=np.ones(4, dtype=np.float32),
    )
    h = torch.zeros(1, cfg.hidden_dim)
    z = torch.zeros(1, cfg.latent_dim)
    prev_action = torch.zeros(1, cfg.action_dim)
    new_h, new_z, obs_embed, surprise = rssm.observe_step(obs, prev_action, h, z)
    assert torch.isfinite(new_h).all()
    assert torch.isfinite(new_z).all()
    assert torch.isfinite(obs_embed).all()
    assert np.isfinite(surprise)


def test_rssm_imagine_step_returns_finite() -> None:
    cfg = _default_cfg()
    rssm = RSSM(cfg)
    rssm.eval()
    h = torch.zeros(1, cfg.hidden_dim)
    z = torch.zeros(1, cfg.latent_dim)
    action = torch.zeros(1, cfg.action_dim)
    new_h, new_z, reward = rssm.imagine_step(action, h, z)
    assert torch.isfinite(new_h).all()
    assert torch.isfinite(new_z).all()
    assert torch.isfinite(reward).all()


def test_agent_actions_bounded() -> None:
    from unittest.mock import MagicMock

    from mousedroid.agents.navigation import MouseDroidNavigationAgent
    from mousedroid.config.schema import Settings
    from mousedroid.safety.context import SafetyContext

    s = Settings(mock_hardware=True)
    cfg = s.model
    mock_planner = MagicMock()
    mock_planner.plan.return_value = torch.tensor([[0.1, 0.0, 0.0]])
    agent = MouseDroidNavigationAgent(mock_planner, s)
    h = torch.zeros(1, cfg.hidden_dim)
    z = torch.zeros(1, cfg.latent_dim)
    ctx = SafetyContext()
    action = agent.act(h, z, ctx)
    assert (action >= -1.0).all()
    assert (action <= 1.0).all()


def test_encoder_output_shape() -> None:
    cfg = _default_cfg()
    enc = MultimodalEncoder(cfg)
    vision = torch.randn(1, cfg.vision_dim)
    ultrasonic = torch.randn(1, cfg.ultrasonic_dim)
    motor = torch.randn(1, cfg.motor_state_dim)
    mask = torch.ones(1, 4)
    out = enc(vision, ultrasonic, motor, mask)
    assert out.shape == (1, cfg.obs_dim)


# ---------------------------------------------------------------------------
# Dual-Stream RSSM property tests
# ---------------------------------------------------------------------------


def _dual_stream_cfg() -> ModelConfig:
    """Small dual-stream config for fast property tests."""
    return ModelConfig(
        vision_dim=16,
        ultrasonic_dim=1,
        motor_state_dim=4,
        hidden_dim=32,
        latent_dim=8,
        action_dim=2,
        obs_dim=16,
        vision_proj_dim=8,
        ultrasonic_proj_dim=4,
        motor_proj_dim=4,
        cfc_hidden_dim=16,
        cfc_backbone_units=32,
        cfc_backbone_layers=1,
    )


@given(
    dist=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=10, deadline=2000)
def test_dual_stream_observe_step_returns_finite(dist: float) -> None:
    cfg = _dual_stream_cfg()
    model = DualStreamRSSM(cfg)
    model.eval()
    combined_dim = cfg.hidden_dim + cfg.cfc_hidden_dim

    obs = MouseDroidObservationBundle(
        _vision_features=np.zeros(cfg.vision_dim, dtype=np.float32),
        _distance_m=dist,
        _motor_state=np.zeros(cfg.motor_state_dim, dtype=np.float32),
        _valid_mask=np.ones(4, dtype=np.float32),
    )
    h = torch.zeros(1, combined_dim)
    z = torch.zeros(1, cfg.latent_dim)
    prev_action = torch.zeros(1, cfg.action_dim)

    new_h, new_z, obs_embed, surprise = model.observe_step(obs, prev_action, h, z)
    assert torch.isfinite(new_h).all()
    assert torch.isfinite(new_z).all()
    assert torch.isfinite(obs_embed).all()
    assert np.isfinite(surprise)


@given(
    gru_dim=st.integers(min_value=16, max_value=128),
    cfc_dim=st.integers(min_value=8, max_value=64),
)
@settings(max_examples=10, deadline=2000)
def test_dual_stream_output_dim_equals_sum(gru_dim: int, cfc_dim: int) -> None:
    """Combined hidden dim always equals gru_dim + cfc_dim."""
    cfg = ModelConfig(
        vision_dim=16,
        ultrasonic_dim=1,
        motor_state_dim=4,
        hidden_dim=gru_dim,
        latent_dim=8,
        action_dim=2,
        obs_dim=16,
        vision_proj_dim=8,
        ultrasonic_proj_dim=4,
        motor_proj_dim=4,
        cfc_hidden_dim=cfc_dim,
        cfc_backbone_units=32,
        cfc_backbone_layers=1,
    )
    model = DualStreamRSSM(cfg)
    model.eval()
    expected_combined = gru_dim + cfc_dim

    h = torch.zeros(1, expected_combined)
    z = torch.zeros(1, cfg.latent_dim)
    action = torch.zeros(1, cfg.action_dim)

    new_h, _new_z, _reward = model.imagine_step(action, h, z)
    assert new_h.shape == (1, expected_combined)
