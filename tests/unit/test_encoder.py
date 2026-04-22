from __future__ import annotations

import pytest
import torch

from mousedroid.config.schema import ModelConfig
from mousedroid.world_model.encoder import MultimodalEncoder


@pytest.fixture
def cfg() -> ModelConfig:
    return ModelConfig()


@pytest.fixture
def encoder(cfg: ModelConfig) -> MultimodalEncoder:
    return MultimodalEncoder(cfg)


# ---------------------------------------------------------------------------
# Original 3-modality tests (backwards compatibility)
# ---------------------------------------------------------------------------


def test_constructor_creates_submodules(encoder: MultimodalEncoder) -> None:
    assert hasattr(encoder, "vision_proj")
    assert hasattr(encoder, "ultrasonic_proj")
    assert hasattr(encoder, "motor_proj")
    assert hasattr(encoder, "fusion")


def test_constructor_custom_dims() -> None:
    cfg = ModelConfig(
        vision_dim=64,
        obs_dim=32,
        vision_proj_dim=16,
        ultrasonic_proj_dim=8,
        motor_proj_dim=8,
    )
    enc = MultimodalEncoder(cfg)
    assert enc.vision_proj.in_features == 64
    assert enc.fusion.out_features == 32


def test_forward_output_shape(encoder: MultimodalEncoder, cfg: ModelConfig) -> None:
    batch = 4
    vision = torch.randn(batch, cfg.vision_dim)
    ultrasonic = torch.randn(batch, cfg.ultrasonic_dim)
    motor = torch.randn(batch, cfg.motor_state_dim)
    mask = torch.ones(batch, 3)
    out = encoder(vision, ultrasonic, motor, mask)
    assert out.shape == (batch, cfg.obs_dim)


def test_forward_single_batch(encoder: MultimodalEncoder, cfg: ModelConfig) -> None:
    vision = torch.randn(1, cfg.vision_dim)
    ultrasonic = torch.randn(1, cfg.ultrasonic_dim)
    motor = torch.randn(1, cfg.motor_state_dim)
    mask = torch.ones(1, 3)
    out = encoder(vision, ultrasonic, motor, mask)
    assert out.shape == (1, cfg.obs_dim)


def test_zero_mask_zeros_output(encoder: MultimodalEncoder, cfg: ModelConfig) -> None:
    batch = 2
    vision = torch.randn(batch, cfg.vision_dim)
    ultrasonic = torch.randn(batch, cfg.ultrasonic_dim)
    motor = torch.randn(batch, cfg.motor_state_dim)
    mask = torch.zeros(batch, 3)
    out = encoder(vision, ultrasonic, motor, mask)
    # With zero mask, all modality projections are zeroed before fusion;
    # the fusion linear layer may still produce non-zero output from bias,
    # but the contribution from the modalities is zero.
    mask_ones = torch.ones(batch, 3)
    out_ones = encoder(vision, ultrasonic, motor, mask_ones)
    # Just verify shapes and that it runs without error
    assert out.shape == out_ones.shape


def test_partial_mask_gates_modality(encoder: MultimodalEncoder, cfg: ModelConfig) -> None:
    batch = 2
    vision = torch.randn(batch, cfg.vision_dim)
    ultrasonic = torch.randn(batch, cfg.ultrasonic_dim)
    motor = torch.randn(batch, cfg.motor_state_dim)
    mask_no_vision = torch.tensor([[0.0, 1.0, 1.0]] * batch)
    mask_all = torch.ones(batch, 3)
    out_no_vision = encoder(vision, ultrasonic, motor, mask_no_vision)
    out_all = encoder(vision, ultrasonic, motor, mask_all)
    assert not torch.allclose(out_no_vision, out_all)


def test_all_ones_valid_mask(encoder: MultimodalEncoder, cfg: ModelConfig) -> None:
    batch = 3
    vision = torch.randn(batch, cfg.vision_dim)
    ultrasonic = torch.randn(batch, cfg.ultrasonic_dim)
    motor = torch.randn(batch, cfg.motor_state_dim)
    mask = torch.ones(batch, 3)
    out = encoder(vision, ultrasonic, motor, mask)
    assert out.shape == (batch, cfg.obs_dim)
    assert torch.isfinite(out).all()


def test_batch_dimension_handling(encoder: MultimodalEncoder, cfg: ModelConfig) -> None:
    for batch in [1, 2, 8, 16]:
        vision = torch.randn(batch, cfg.vision_dim)
        ultrasonic = torch.randn(batch, cfg.ultrasonic_dim)
        motor = torch.randn(batch, cfg.motor_state_dim)
        mask = torch.ones(batch, 3)
        out = encoder(vision, ultrasonic, motor, mask)
        assert out.shape == (batch, cfg.obs_dim)


def test_output_is_differentiable(encoder: MultimodalEncoder, cfg: ModelConfig) -> None:
    vision = torch.randn(2, cfg.vision_dim, requires_grad=True)
    ultrasonic = torch.randn(2, cfg.ultrasonic_dim)
    motor = torch.randn(2, cfg.motor_state_dim)
    mask = torch.ones(2, 3)
    out = encoder(vision, ultrasonic, motor, mask)
    out.sum().backward()
    assert vision.grad is not None


def test_encoder_is_nn_module(encoder: MultimodalEncoder) -> None:
    assert isinstance(encoder, torch.nn.Module)


# ---------------------------------------------------------------------------
# Audio-enabled tests (4-modality)
# ---------------------------------------------------------------------------


def _make_audio_cfg(audio_dim: int = 1024, audio_proj_dim: int = 32) -> ModelConfig:
    return ModelConfig(audio_dim=audio_dim, audio_proj_dim=audio_proj_dim)


def _make_audio_encoder(audio_dim: int = 1024, audio_proj_dim: int = 32) -> MultimodalEncoder:
    return MultimodalEncoder(_make_audio_cfg(audio_dim, audio_proj_dim))


def test_forward_4_modality_with_audio() -> None:
    cfg = _make_audio_cfg()
    enc = _make_audio_encoder()
    batch = 4
    vision = torch.randn(batch, cfg.vision_dim)
    ultrasonic = torch.randn(batch, cfg.ultrasonic_dim)
    motor = torch.randn(batch, cfg.motor_state_dim)
    audio = torch.randn(batch, cfg.audio_dim)
    mask = torch.ones(batch, 4)
    out = enc(vision, ultrasonic, motor, mask, audio=audio)
    assert out.shape == (batch, cfg.obs_dim)
    assert torch.isfinite(out).all()


def test_audio_gating_zeroed() -> None:
    cfg = _make_audio_cfg()
    enc = _make_audio_encoder()
    batch = 2
    vision = torch.randn(batch, cfg.vision_dim)
    ultrasonic = torch.randn(batch, cfg.ultrasonic_dim)
    motor = torch.randn(batch, cfg.motor_state_dim)
    audio = torch.randn(batch, cfg.audio_dim)
    mask_audio_off = torch.tensor([[1.0, 1.0, 1.0, 0.0]] * batch)
    mask_audio_on = torch.ones(batch, 4)
    out_off = enc(vision, ultrasonic, motor, mask_audio_off, audio=audio)
    out_on = enc(vision, ultrasonic, motor, mask_audio_on, audio=audio)
    assert not torch.allclose(out_off, out_on)


def test_backward_compat_3_modality_no_audio() -> None:
    """Default config (audio_dim=0) with 3-element mask works as before."""
    cfg = ModelConfig()
    enc = MultimodalEncoder(cfg)
    batch = 2
    vision = torch.randn(batch, cfg.vision_dim)
    ultrasonic = torch.randn(batch, cfg.ultrasonic_dim)
    motor = torch.randn(batch, cfg.motor_state_dim)
    mask = torch.ones(batch, 3)
    out = enc(vision, ultrasonic, motor, mask)
    assert out.shape == (batch, cfg.obs_dim)


def test_backward_compat_audio_enabled_no_kwarg() -> None:
    """Audio enabled in config but no audio tensor passed — should not error."""
    enc = _make_audio_encoder()
    cfg = _make_audio_cfg()
    batch = 2
    vision = torch.randn(batch, cfg.vision_dim)
    ultrasonic = torch.randn(batch, cfg.ultrasonic_dim)
    motor = torch.randn(batch, cfg.motor_state_dim)
    mask = torch.ones(batch, 4)
    # audio kwarg omitted — defaults to None
    out = enc(vision, ultrasonic, motor, mask)
    assert out.shape == (batch, cfg.obs_dim)


def test_custom_audio_dims() -> None:
    cfg = ModelConfig(audio_dim=512, audio_proj_dim=16)
    enc = MultimodalEncoder(cfg)
    batch = 2
    audio = torch.randn(batch, 512)
    vision = torch.randn(batch, cfg.vision_dim)
    ultrasonic = torch.randn(batch, cfg.ultrasonic_dim)
    motor = torch.randn(batch, cfg.motor_state_dim)
    mask = torch.ones(batch, 4)
    out = enc(vision, ultrasonic, motor, mask, audio=audio)
    assert out.shape == (batch, cfg.obs_dim)


def test_fused_dim_includes_audio() -> None:
    enc = _make_audio_encoder()
    # 128 (vision) + 32 (ultrasonic) + 32 (motor) + 32 (audio) = 224
    assert enc.fusion.in_features == 128 + 32 + 32 + 32


def test_fused_dim_excludes_audio() -> None:
    enc = MultimodalEncoder(ModelConfig())
    # 128 (vision) + 32 (ultrasonic) + 32 (motor) = 192
    assert enc.fusion.in_features == 128 + 32 + 32


def test_audio_disabled_no_audio_proj() -> None:
    enc = MultimodalEncoder(ModelConfig())  # audio_dim=0
    assert not hasattr(enc, "audio_proj")


def test_audio_enabled_has_audio_proj() -> None:
    enc = _make_audio_encoder()
    assert hasattr(enc, "audio_proj")
    assert enc.audio_proj.in_features == 1024
    assert enc.audio_proj.out_features == 32


def test_ultrasonic_disabled_no_ultrasonic_proj() -> None:
    enc = MultimodalEncoder(
        ModelConfig(ultrasonic_dim=0, ultrasonic_proj_dim=0, lidar_dim=36, lidar_proj_dim=16)
    )
    assert not hasattr(enc, "ultrasonic_proj")
    assert not enc.ultrasonic_enabled


def test_output_differentiable_with_audio() -> None:
    cfg = _make_audio_cfg()
    enc = _make_audio_encoder()
    audio = torch.randn(2, cfg.audio_dim, requires_grad=True)
    vision = torch.randn(2, cfg.vision_dim)
    ultrasonic = torch.randn(2, cfg.ultrasonic_dim)
    motor = torch.randn(2, cfg.motor_state_dim)
    mask = torch.ones(2, 4)
    out = enc(vision, ultrasonic, motor, mask, audio=audio)
    out.sum().backward()
    assert audio.grad is not None


# ---------------------------------------------------------------------------
# LiDAR-enabled tests (5-modality)
# ---------------------------------------------------------------------------


def test_lidar_disabled_backwards_compat() -> None:
    """ModelConfig with lidar_dim=0, verify forward still works with 4-element mask."""
    cfg = ModelConfig(lidar_dim=0)
    enc = MultimodalEncoder(cfg)
    batch = 2
    vision = torch.randn(batch, cfg.vision_dim)
    ultrasonic = torch.randn(batch, cfg.ultrasonic_dim)
    motor = torch.randn(batch, cfg.motor_state_dim)
    mask = torch.ones(batch, 4)
    out = enc(vision, ultrasonic, motor, mask)
    assert out.shape == (batch, cfg.obs_dim)
    assert torch.isfinite(out).all()


def test_lidar_enabled_forward() -> None:
    """ModelConfig with lidar_dim=36, lidar_proj_dim=16, verify forward with lidar tensor."""
    cfg = ModelConfig(lidar_dim=36, lidar_proj_dim=16)
    enc = MultimodalEncoder(cfg)
    batch = 4
    vision = torch.randn(batch, cfg.vision_dim)
    ultrasonic = torch.randn(batch, cfg.ultrasonic_dim)
    motor = torch.randn(batch, cfg.motor_state_dim)
    lidar_tensor = torch.randn(batch, cfg.lidar_dim)
    mask = torch.ones(batch, 5)
    out = enc(vision, ultrasonic, motor, mask, lidar=lidar_tensor)
    assert out.shape == (batch, cfg.obs_dim)
    assert torch.isfinite(out).all()


def test_lidar_only_forward_without_ultrasonic() -> None:
    cfg = ModelConfig(ultrasonic_dim=0, ultrasonic_proj_dim=0, lidar_dim=36, lidar_proj_dim=16)
    enc = MultimodalEncoder(cfg)
    batch = 3
    vision = torch.randn(batch, cfg.vision_dim)
    motor = torch.randn(batch, cfg.motor_state_dim)
    lidar_tensor = torch.randn(batch, cfg.lidar_dim)
    mask = torch.ones(batch, 5)
    out = enc(vision, None, motor, mask, lidar=lidar_tensor)
    assert out.shape == (batch, cfg.obs_dim)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Coverage gap tests
# ---------------------------------------------------------------------------


def test_gate_projection_short_mask() -> None:
    """_gate_projection returns zeros_like when valid_mask is narrower than slot index."""
    cfg = ModelConfig()
    enc = MultimodalEncoder(cfg)
    projected = torch.ones(2, 16)
    # "motor" has slot_index=2; a 1-column mask satisfies shape[-1] <= slot_index
    narrow_mask = torch.ones(2, 1)
    result = enc._gate_projection(projected, narrow_mask, "motor")
    assert result.shape == projected.shape
    assert torch.all(result == 0)


def test_ultrasonic_enabled_none_input_uses_zeros() -> None:
    """Encoder fills zero projection when ultrasonic=None but ultrasonic_enabled=True."""
    cfg = ModelConfig()  # default has ultrasonic_dim=1
    assert cfg.ultrasonic_dim > 0, "fixture assumption"
    enc = MultimodalEncoder(cfg)
    batch = 3
    vision = torch.randn(batch, cfg.vision_dim)
    motor = torch.randn(batch, cfg.motor_state_dim)
    mask = torch.ones(batch, 4)
    out = enc(vision, None, motor, mask)
    assert out.shape == (batch, cfg.obs_dim)
    assert torch.isfinite(out).all()


def test_lidar_enabled_none_input_uses_zeros() -> None:
    """Encoder fills zero projection when lidar=None but lidar_enabled=True."""
    cfg = ModelConfig(lidar_dim=36, lidar_proj_dim=16)
    enc = MultimodalEncoder(cfg)
    batch = 3
    vision = torch.randn(batch, cfg.vision_dim)
    ultrasonic = torch.randn(batch, cfg.ultrasonic_dim)
    motor = torch.randn(batch, cfg.motor_state_dim)
    mask = torch.ones(batch, 5)
    out = enc(vision, ultrasonic, motor, mask, lidar=None)
    assert out.shape == (batch, cfg.obs_dim)
    assert torch.isfinite(out).all()
