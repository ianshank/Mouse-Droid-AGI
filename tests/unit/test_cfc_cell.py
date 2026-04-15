"""Unit tests for CfCWrapper — ncps CfC cell wrapper."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("ncps")


from mousedroid.config.schema import ModelConfig
from mousedroid.world_model.cfc_cell import CfCWrapper


def _make_cfg(cfc_dim: int = 32) -> ModelConfig:
    return ModelConfig(
        cfc_hidden_dim=cfc_dim,
        cfc_backbone_units=64,
        cfc_backbone_layers=1,
        cfc_mode="default",
        hidden_dim=32,
        latent_dim=8,
        action_dim=2,
        obs_dim=16,
    )


class TestCfCWrapperConstruction:
    """Test CfCWrapper initialization."""

    def test_hidden_size_matches_config(self) -> None:
        cfg = _make_cfg(cfc_dim=32)
        wrapper = CfCWrapper(input_dim=10, cfg=cfg)
        assert wrapper.hidden_size == 32

    def test_hidden_size_various_dims(self) -> None:
        for dim in [16, 32, 64]:
            cfg = _make_cfg(cfc_dim=dim)
            wrapper = CfCWrapper(input_dim=10, cfg=cfg)
            assert wrapper.hidden_size == dim

    def test_is_nn_module(self) -> None:
        cfg = _make_cfg()
        wrapper = CfCWrapper(input_dim=10, cfg=cfg)
        assert isinstance(wrapper, torch.nn.Module)

    def test_has_parameters(self) -> None:
        cfg = _make_cfg()
        wrapper = CfCWrapper(input_dim=10, cfg=cfg)
        params = list(wrapper.parameters())
        assert len(params) > 0


class TestCfCWrapperForward:
    """Test CfCWrapper forward pass."""

    def test_forward_shape_batch_1(self) -> None:
        cfg = _make_cfg(cfc_dim=32)
        wrapper = CfCWrapper(input_dim=10, cfg=cfg)
        x = torch.randn(1, 10)
        h = wrapper.initial_state(1)
        h_new = wrapper.forward(x, h)
        assert h_new.shape == (1, 32)

    def test_forward_shape_batch_4(self) -> None:
        cfg = _make_cfg(cfc_dim=32)
        wrapper = CfCWrapper(input_dim=10, cfg=cfg)
        x = torch.randn(4, 10)
        h = wrapper.initial_state(4)
        h_new = wrapper.forward(x, h)
        assert h_new.shape == (4, 32)

    def test_forward_output_finite(self) -> None:
        cfg = _make_cfg(cfc_dim=16)
        wrapper = CfCWrapper(input_dim=10, cfg=cfg)
        x = torch.randn(4, 10)
        h = wrapper.initial_state(4)
        h_new = wrapper.forward(x, h)
        assert torch.isfinite(h_new).all()

    def test_forward_with_dt(self) -> None:
        cfg = _make_cfg(cfc_dim=32)
        wrapper = CfCWrapper(input_dim=10, cfg=cfg)
        x = torch.randn(4, 10)
        h = wrapper.initial_state(4)
        dt = torch.ones(4, 1) * 0.033
        h_new = wrapper.forward(x, h, dt=dt)
        assert h_new.shape == (4, 32)
        assert torch.isfinite(h_new).all()

    def test_dt_affects_output(self) -> None:
        """Different dt values produce different hidden states."""
        cfg = _make_cfg(cfc_dim=32)
        wrapper = CfCWrapper(input_dim=10, cfg=cfg)
        wrapper.eval()

        torch.manual_seed(42)
        x = torch.randn(4, 10)
        h = torch.randn(4, 32) * 0.1

        dt_fast = torch.ones(4, 1) * 0.01
        dt_slow = torch.ones(4, 1) * 1.0

        h_fast = wrapper.forward(x, h, dt=dt_fast)
        h_slow = wrapper.forward(x, h, dt=dt_slow)

        # Different time deltas should produce different outputs
        assert not torch.allclose(h_fast, h_slow, atol=1e-5)

    def test_gradient_flows(self) -> None:
        cfg = _make_cfg(cfc_dim=16)
        wrapper = CfCWrapper(input_dim=10, cfg=cfg)
        x = torch.randn(2, 10, requires_grad=True)
        h = wrapper.initial_state(2)
        h_new = wrapper.forward(x, h)
        loss = h_new.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_different_input_dims(self) -> None:
        """Wrapper works with various input dimensions."""
        for input_dim in [5, 10, 67, 128]:
            cfg = _make_cfg(cfc_dim=16)
            wrapper = CfCWrapper(input_dim=input_dim, cfg=cfg)
            x = torch.randn(2, input_dim)
            h = wrapper.initial_state(2)
            h_new = wrapper.forward(x, h)
            assert h_new.shape == (2, 16)


class TestCfCWrapperInitialState:
    """Test initial state creation."""

    def test_initial_state_shape(self) -> None:
        cfg = _make_cfg(cfc_dim=32)
        wrapper = CfCWrapper(input_dim=10, cfg=cfg)
        h = wrapper.initial_state(4)
        assert h.shape == (4, 32)

    def test_initial_state_is_zeros(self) -> None:
        cfg = _make_cfg(cfc_dim=32)
        wrapper = CfCWrapper(input_dim=10, cfg=cfg)
        h = wrapper.initial_state(4)
        assert torch.allclose(h, torch.zeros(4, 32))

    def test_initial_state_respects_device(self) -> None:
        cfg = _make_cfg(cfc_dim=16)
        wrapper = CfCWrapper(input_dim=10, cfg=cfg)
        h = wrapper.initial_state(2, device=torch.device("cpu"))
        assert h.device == torch.device("cpu")
