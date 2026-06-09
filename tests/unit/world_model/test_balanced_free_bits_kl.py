"""Balanced, free-bits, fp32-stable KL for RSSM training."""

from __future__ import annotations

import torch

from mousedroid.world_model.latent_utils import balanced_free_bits_kl


def _g(mean: float, logvar: float, n: int = 4, d: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.full((n, d), mean), torch.full((n, d), logvar)


def test_identical_distributions_clamped_to_free_nats() -> None:
    pm, pl = _g(0.0, 0.0)
    km, kl = _g(0.0, 0.0)
    out = balanced_free_bits_kl(pm, pl, km, kl, alpha=0.8, free_nats=1.0)
    # zero KL is clamped up to free_nats
    assert torch.isfinite(out)
    assert float(out) == 1.0


def test_free_nats_zero_recovers_plain_kl_scale() -> None:
    pm, pl = _g(2.0, 0.0)
    rm, rl = _g(0.0, 0.0)
    out = balanced_free_bits_kl(pm, pl, rm, rl, alpha=0.5, free_nats=0.0)
    assert float(out) > 0.0


def test_fp16_inputs_do_not_overflow() -> None:
    pm = torch.full((2, 4), 0.0, dtype=torch.float16)
    pl = torch.full((2, 4), 30.0, dtype=torch.float16)  # exp(30) overflows fp16
    rm = torch.zeros(2, 4, dtype=torch.float16)
    rl = torch.zeros(2, 4, dtype=torch.float16)
    out = balanced_free_bits_kl(pm, pl, rm, rl, alpha=0.8, free_nats=1.0)
    assert torch.isfinite(out)


def test_balancing_is_convex_combination() -> None:
    pm, pl = _g(1.0, 0.5)
    rm, rl = _g(0.0, 0.0)
    a0 = balanced_free_bits_kl(pm, pl, rm, rl, alpha=0.0, free_nats=0.0)
    a1 = balanced_free_bits_kl(pm, pl, rm, rl, alpha=1.0, free_nats=0.0)
    amid = balanced_free_bits_kl(pm, pl, rm, rl, alpha=0.5, free_nats=0.0)
    lo, hi = min(float(a0), float(a1)), max(float(a0), float(a1))
    assert lo - 1e-4 <= float(amid) <= hi + 1e-4


def test_returns_scalar() -> None:
    pm, pl = _g(1.0, 0.5)
    rm, rl = _g(0.0, 0.0)
    out = balanced_free_bits_kl(pm, pl, rm, rl, alpha=0.8, free_nats=0.0)
    assert out.ndim == 0
