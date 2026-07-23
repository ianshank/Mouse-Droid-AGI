"""Hypothesis property tests for the F-023 surfaces.

The repo test-tier mirror (CLAUDE.md) lists the property tier as part of the
mandatory mirror for world-model/learning features. These fuzz the
load-bearing invariants over generated dims/lengths that the example-based unit
tests only spot-check:

- ``BoundedContextMemory`` constant footprint (``len <= recent_size + 2``) and
  cold-start identity (empty key set ⇒ exact identity) hold for any dims.
- ``RSSM.train_sequence_corrupted`` forced-k=0 equals ``train_sequence`` for
  any (small) dims/lengths.
"""

from __future__ import annotations

import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mousedroid.config.schema import ModelConfig, WorldModelMemoryConfig
from mousedroid.constants import SENSOR_SLOT_MAP
from mousedroid.world_model.bounded_context import BoundedContextMemory
from mousedroid.world_model.rssm import RSSM, RawModalityDecoders

# max_prefix_frac small enough that floor(frac * T) == 0 for the T range below.
_FORCE_K0_FRAC = 0.05


@given(
    h_dim=st.integers(min_value=1, max_value=12),
    z_dim=st.integers(min_value=1, max_value=8),
    recent_size=st.integers(min_value=1, max_value=8),
    n_observes=st.integers(min_value=0, max_value=60),
)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_footprint_constant_for_any_dims(
    h_dim: int, z_dim: int, recent_size: int, n_observes: int
) -> None:
    cfg = WorldModelMemoryConfig.model_validate(
        {"enabled": True, "recent_size": recent_size, "sink_warmup_ticks": 0, "stride": 1}
    )
    mem = BoundedContextMemory(cfg, h_dim=h_dim, z_dim=z_dim)
    gen = torch.Generator().manual_seed(h_dim * 100 + z_dim * 10 + recent_size)
    for _ in range(n_observes):
        mem.observe(torch.randn(1, h_dim, generator=gen), torch.randn(1, z_dim, generator=gen))
    assert len(mem) <= recent_size + 2


@given(
    h_dim=st.integers(min_value=1, max_value=12),
    z_dim=st.integers(min_value=1, max_value=8),
    blend_weight=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=40, deadline=None)
def test_empty_memory_is_identity_for_any_dims(h_dim: int, z_dim: int, blend_weight: float) -> None:
    cfg = WorldModelMemoryConfig.model_validate(
        {"enabled": True, "blend_weight": blend_weight, "sink_warmup_ticks": 5}
    )
    mem = BoundedContextMemory(cfg, h_dim=h_dim, z_dim=z_dim)
    h = torch.randn(1, h_dim)
    z = torch.randn(1, z_dim)
    h_out, z_out = mem.contextualize(h, z)
    # Empty key set (or blend_weight 0) ⇒ the exact input references back.
    assert h_out is h
    assert z_out is z


def _cfg(hidden: int, latent: int, action: int) -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "vision_dim": 0,
            "vision_proj_dim": 0,
            "ultrasonic_dim": 1,
            "motor_state_dim": 4,
            "hidden_dim": hidden,
            "latent_dim": latent,
            "action_dim": action,
            "obs_dim": hidden,
            "ultrasonic_proj_dim": 4,
            "motor_proj_dim": 8,
        }
    )


@given(
    hidden=st.integers(min_value=4, max_value=16),
    latent=st.integers(min_value=2, max_value=8),
    action=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=2, max_value=6),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_k0_equals_train_sequence_for_any_dims(
    hidden: int, latent: int, action: int, seq_len: int, seed: int
) -> None:
    mcfg = _cfg(hidden, latent, action)
    torch.manual_seed(0)
    model = RSSM(mcfg)
    torch.manual_seed(0)
    decoders = RawModalityDecoders(mcfg)
    b = 2
    gen = torch.Generator().manual_seed(seed)
    valid = torch.zeros(b, seq_len, len(SENSOR_SLOT_MAP))
    valid[..., SENSOR_SLOT_MAP["motor"]] = 1.0
    valid[..., SENSOR_SLOT_MAP["ultrasonic"]] = 1.0
    batch = {
        "motor": torch.randn(b, seq_len, mcfg.motor_state_dim, generator=gen),
        "ultrasonic": torch.randn(b, seq_len, mcfg.ultrasonic_dim, generator=gen),
        "valid_mask": valid,
        "action": torch.randn(b, seq_len, mcfg.action_dim, generator=gen),
    }
    torch.manual_seed(seed)
    base = model.train_sequence(batch, decoders)
    torch.manual_seed(seed)
    corrupted = model.train_sequence_corrupted(batch, decoders, max_prefix_frac=_FORCE_K0_FRAC)
    assert int(corrupted["prefix_len"]) == 0
    assert torch.allclose(base["loss"], corrupted["loss"])
