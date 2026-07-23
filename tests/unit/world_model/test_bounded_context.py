"""Unit tests for the F-023 bounded-context latent memory.

Pins the D1 design contracts (ADR-015): constant-size storage, cold-start
identity (uncaptured sink / never-folded EMA excluded; empty key set ⇒ exact
identity), the NaN contract (non-finite inputs dropped; non-finite blend falls
back to identity), sink capture/rearm/reset lifecycle, determinism, and
protocol conformance.
"""

from __future__ import annotations

import pytest
import torch

from mousedroid.config.schema import WorldModelMemoryConfig
from mousedroid.world_model.bounded_context import BoundedContextMemory
from mousedroid.world_model.protocol import LatentContextProtocol

_H_DIM = 8
_Z_DIM = 4


def _mem(**overrides: object) -> BoundedContextMemory:
    cfg = WorldModelMemoryConfig.model_validate({"enabled": True, **overrides})
    return BoundedContextMemory(cfg, h_dim=_H_DIM, z_dim=_Z_DIM)


def _state(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return (
        torch.randn(1, _H_DIM, generator=gen),
        torch.randn(1, _Z_DIM, generator=gen),
    )


class TestBoundedness:
    def test_footprint_constant_over_long_rollout(self) -> None:
        """S1a: 10k observes never exceed recent_size + 2 stored vectors."""
        mem = _mem(recent_size=16, sink_warmup_ticks=0, stride=8)
        cap = 16 + 2
        for i in range(10_000):
            h, z = _state(i)
            mem.observe(h, z)
            assert len(mem) <= cap
        assert len(mem) == cap

    def test_invalid_dims_rejected(self) -> None:
        cfg = WorldModelMemoryConfig()
        with pytest.raises(ValueError, match="must be positive"):
            BoundedContextMemory(cfg, h_dim=0, z_dim=_Z_DIM)


class TestColdStartIdentity:
    def test_empty_memory_is_exact_identity(self) -> None:
        """No zero-vector damping: empty key set returns the inputs untouched."""
        mem = _mem(blend_weight=0.5, sink_warmup_ticks=10)
        h, z = _state(0)
        h_out, z_out = mem.contextualize(h, z)
        assert h_out is h
        assert z_out is z

    def test_zero_blend_weight_is_exact_identity(self) -> None:
        mem = _mem(blend_weight=0.0, sink_warmup_ticks=0)
        for i in range(20):
            mem.observe(*_state(i))
        h, z = _state(99)
        h_out, z_out = mem.contextualize(h, z)
        assert h_out is h
        assert z_out is z

    def test_uncaptured_sink_and_unfolded_ema_excluded(self) -> None:
        """Before warmup/fold, only the ring participates in retrieval."""
        mem = _mem(sink_warmup_ticks=100, stride=100, recent_size=4, blend_weight=0.5)
        mem.observe(*_state(0))
        # ring=1 entry, no sink, no EMA
        assert len(mem) == 1


class TestNaNContract:
    def test_nonfinite_observe_dropped(self) -> None:
        mem = _mem(sink_warmup_ticks=0)
        h = torch.full((1, _H_DIM), float("nan"))
        z = torch.zeros(1, _Z_DIM)
        mem.observe(h, z)
        assert len(mem) == 0  # nothing stored — sink not captured either

    def test_inf_observe_dropped(self) -> None:
        mem = _mem(sink_warmup_ticks=0)
        h = torch.full((1, _H_DIM), float("inf"))
        z = torch.zeros(1, _Z_DIM)
        mem.observe(h, z)
        assert len(mem) == 0

    def test_nonfinite_query_blend_falls_back_to_identity(self) -> None:
        """A NaN query would make the blend NaN — identity fallback fires."""
        mem = _mem(sink_warmup_ticks=0, blend_weight=0.5)
        for i in range(4):
            mem.observe(*_state(i))
        h_nan = torch.full((1, _H_DIM), float("nan"))
        z = torch.zeros(1, _Z_DIM)
        h_out, z_out = mem.contextualize(h_nan, z)
        assert h_out is h_nan
        assert z_out is z


class TestSinkLifecycle:
    def test_sink_captured_after_warmup(self) -> None:
        mem = _mem(sink_warmup_ticks=3, recent_size=2)
        for i in range(3):
            mem.observe(*_state(i))
        assert len(mem) == 2  # ring only (capacity 2), sink not yet captured
        mem.observe(*_state(3))  # 4th validated observe: warmup satisfied
        assert len(mem) == 3  # ring(2) + sink

    def test_sink_frozen_beyond_compressed_window(self) -> None:
        """S1b (accessibility): the sink persists after the ring fully turns over."""
        mem = _mem(sink_warmup_ticks=0, recent_size=4, stride=1000, blend_weight=0.5)
        h0, z0 = _state(0)
        mem.observe(h0, z0)  # captured as sink
        sink_before = mem._sink
        assert sink_before is not None
        for i in range(1, 4 * 20 + 1):
            mem.observe(*_state(i))
        assert mem._sink is not None
        assert torch.equal(mem._sink, sink_before)

    def test_sink_shifts_retrieval_output(self) -> None:
        """S1b (incorporation): sink-present output differs from sink-ablated."""
        with_sink = _mem(sink_warmup_ticks=0, recent_size=4, stride=1000, blend_weight=0.3)
        no_sink = _mem(sink_warmup_ticks=0, recent_size=4, stride=1000, blend_weight=0.3)
        # Distinguishable first state becomes with_sink's anchor.
        h0 = torch.full((1, _H_DIM), 5.0)
        z0 = torch.full((1, _Z_DIM), -5.0)
        with_sink.observe(h0, z0)
        no_sink.observe(h0, z0)
        no_sink.rearm_sink()  # ablate the anchor; ring content stays identical
        # Roll far beyond the compressed window with identical inputs.
        for i in range(1, 41):
            h, z = _state(i)
            with_sink.observe(h, z)
            no_sink.observe(h, z)
        no_sink._sink = None  # rearm may have recaptured; force pure ablation
        q_h, q_z = _state(500)
        h_a, z_a = with_sink.contextualize(q_h, q_z)
        h_b, z_b = no_sink.contextualize(q_h, q_z)
        assert not torch.allclose(h_a, h_b)
        assert not torch.allclose(z_a, z_b)

    def test_rearm_clears_sink_keeps_ring_and_ema(self) -> None:
        mem = _mem(sink_warmup_ticks=0, recent_size=4, stride=2)
        for i in range(8):
            mem.observe(*_state(i))
        assert len(mem) == 4 + 2  # ring + sink + EMA
        mem.rearm_sink()
        assert len(mem) == 4 + 1  # sink gone; ring + EMA retained

    def test_reset_clears_everything_and_rearms(self) -> None:
        mem = _mem(sink_warmup_ticks=0, recent_size=4, stride=2)
        for i in range(8):
            mem.observe(*_state(i))
        mem.reset()
        assert len(mem) == 0
        # Warmup restarted: the next observe recaptures a fresh sink.
        mem.observe(*_state(100))
        assert len(mem) == 2  # ring(1) + fresh sink


class TestDeterminism:
    def test_same_inputs_same_outputs(self) -> None:
        """Pure tensor ops — identical histories yield identical blends."""

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            mem = _mem(sink_warmup_ticks=0, recent_size=8, stride=4, blend_weight=0.2)
            for i in range(30):
                mem.observe(*_state(i))
            return mem.contextualize(*_state(999))

        h1, z1 = run()
        h2, z2 = run()
        assert torch.equal(h1, h2)
        assert torch.equal(z1, z2)

    def test_blend_output_shapes(self) -> None:
        mem = _mem(sink_warmup_ticks=0, blend_weight=0.2)
        mem.observe(*_state(0))
        h, z = _state(1)
        h_out, z_out = mem.contextualize(h, z)
        assert h_out.shape == (1, _H_DIM)
        assert z_out.shape == (1, _Z_DIM)
        assert bool(torch.isfinite(h_out).all())
        assert bool(torch.isfinite(z_out).all())


def test_protocol_conformance() -> None:
    mem = _mem()
    assert isinstance(mem, LatentContextProtocol)
