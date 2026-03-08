from __future__ import annotations

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.config.schema import ModelConfig
from mousedroid.sensing.bundle import MouseDroidObservationBundle
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
    mask = torch.ones(1, 3)
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
        _valid_mask=np.ones(3, dtype=np.float32),
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
    mock_wm = MagicMock()
    mock_wm.imagine_step.return_value = (
        torch.zeros(1, cfg.hidden_dim),
        torch.zeros(1, cfg.latent_dim),
        torch.tensor([[0.1]]),
    )
    agent = MouseDroidNavigationAgent(mock_wm, s)
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
    mask = torch.ones(1, 3)
    out = enc(vision, ultrasonic, motor, mask)
    assert out.shape == (1, cfg.obs_dim)
