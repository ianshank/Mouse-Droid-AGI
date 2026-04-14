"""Unit tests for StreamFusion — concat fusion + state extraction."""

from __future__ import annotations

import torch

from mousedroid.world_model.stream_fusion import StreamFusion


class TestStreamFusionConstruction:
    """Test StreamFusion initialization."""

    def test_combined_dim(self) -> None:
        fusion = StreamFusion(gru_dim=256, cfc_dim=32)
        assert fusion.combined_dim == 288

    def test_gru_dim_property(self) -> None:
        fusion = StreamFusion(gru_dim=128, cfc_dim=64)
        assert fusion.gru_dim == 128

    def test_cfc_dim_property(self) -> None:
        fusion = StreamFusion(gru_dim=128, cfc_dim=64)
        assert fusion.cfc_dim == 64

    def test_is_nn_module(self) -> None:
        fusion = StreamFusion(gru_dim=256, cfc_dim=32)
        assert isinstance(fusion, torch.nn.Module)


class TestStreamFusionFuse:
    """Test fuse() concatenation."""

    def test_fuse_shape(self) -> None:
        fusion = StreamFusion(gru_dim=256, cfc_dim=32)
        h_slow = torch.randn(4, 256)
        h_fast = torch.randn(4, 32)
        h_combined = fusion.fuse(h_slow, h_fast)
        assert h_combined.shape == (4, 288)

    def test_fuse_single_batch(self) -> None:
        fusion = StreamFusion(gru_dim=64, cfc_dim=16)
        h_slow = torch.randn(1, 64)
        h_fast = torch.randn(1, 16)
        h_combined = fusion.fuse(h_slow, h_fast)
        assert h_combined.shape == (1, 80)

    def test_fuse_preserves_values(self) -> None:
        fusion = StreamFusion(gru_dim=4, cfc_dim=2)
        h_slow = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        h_fast = torch.tensor([[5.0, 6.0]])
        h_combined = fusion.fuse(h_slow, h_fast)
        expected = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        assert torch.allclose(h_combined, expected)

    def test_forward_is_alias_for_fuse(self) -> None:
        fusion = StreamFusion(gru_dim=32, cfc_dim=8)
        h_slow = torch.randn(2, 32)
        h_fast = torch.randn(2, 8)
        assert torch.allclose(fusion(h_slow, h_fast), fusion.fuse(h_slow, h_fast))


class TestStreamFusionExtraction:
    """Test state extraction via slicing."""

    def test_extract_gru_state_shape(self) -> None:
        fusion = StreamFusion(gru_dim=256, cfc_dim=32)
        h_combined = torch.randn(4, 288)
        h_gru = fusion.extract_gru_state(h_combined)
        assert h_gru.shape == (4, 256)

    def test_extract_cfc_state_shape(self) -> None:
        fusion = StreamFusion(gru_dim=256, cfc_dim=32)
        h_combined = torch.randn(4, 288)
        h_cfc = fusion.extract_cfc_state(h_combined)
        assert h_cfc.shape == (4, 32)

    def test_roundtrip_fuse_extract(self) -> None:
        """fuse(extract_gru(h), extract_cfc(h)) == h."""
        fusion = StreamFusion(gru_dim=64, cfc_dim=16)
        h_slow = torch.randn(4, 64)
        h_fast = torch.randn(4, 16)
        h_combined = fusion.fuse(h_slow, h_fast)
        h_slow_back = fusion.extract_gru_state(h_combined)
        h_fast_back = fusion.extract_cfc_state(h_combined)
        assert torch.allclose(h_slow_back, h_slow)
        assert torch.allclose(h_fast_back, h_fast)

    def test_extract_gru_values_correct(self) -> None:
        fusion = StreamFusion(gru_dim=3, cfc_dim=2)
        h_combined = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        h_gru = fusion.extract_gru_state(h_combined)
        assert torch.allclose(h_gru, torch.tensor([[1.0, 2.0, 3.0]]))

    def test_extract_cfc_values_correct(self) -> None:
        fusion = StreamFusion(gru_dim=3, cfc_dim=2)
        h_combined = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        h_cfc = fusion.extract_cfc_state(h_combined)
        assert torch.allclose(h_cfc, torch.tensor([[4.0, 5.0]]))

    def test_gradient_flows_through_extraction(self) -> None:
        fusion = StreamFusion(gru_dim=8, cfc_dim=4)
        h = torch.randn(2, 12, requires_grad=True)
        h_gru = fusion.extract_gru_state(h)
        loss = h_gru.sum()
        loss.backward()
        assert h.grad is not None
        # Gradient should only flow through GRU slice
        assert h.grad[:, :8].abs().sum() > 0
        assert torch.allclose(h.grad[:, 8:], torch.zeros(2, 4))
