"""Latency budget for the F-023 bounded-context blend.

The existing ``test_observe_step_budget.py`` times ``world_model.observe_step``
directly — the orchestrator-seam blend is OUTSIDE its measured path, so this
file owns the blend's budget. The budget is env-var-tunable
(``MOUSEDROID_LATENT_CONTEXT_BUDGET_MS``, default 5 ms — generous against the
33 ms tick at 30 Hz; the blend is one ``(K+2) x D`` matmul + softmax).
"""

from __future__ import annotations

import os
import time

import pytest
import torch

from mousedroid.config.schema import ModelConfig, WorldModelMemoryConfig
from mousedroid.world_model.bounded_context import BoundedContextMemory

_ITERATIONS = 200
_DEFAULT_BUDGET_MS = 5.0


def _resolve_budget_ms() -> float:
    raw = os.environ.get("MOUSEDROID_LATENT_CONTEXT_BUDGET_MS", "")
    if not raw:
        return _DEFAULT_BUDGET_MS
    budget = float(raw)
    if budget <= 0:
        msg = f"MOUSEDROID_LATENT_CONTEXT_BUDGET_MS must be positive; got {raw!r}"
        raise ValueError(msg)
    return budget


@pytest.mark.slow
def test_contextualize_mean_latency_under_budget() -> None:
    """Full memory (worst case: ring at capacity + sink + EMA), default dims."""
    model_cfg = ModelConfig()
    memory_cfg = WorldModelMemoryConfig.model_validate({"enabled": True, "sink_warmup_ticks": 0})
    h_dim = model_cfg.hidden_dim + model_cfg.cfc_hidden_dim
    mem = BoundedContextMemory(memory_cfg, h_dim=h_dim, z_dim=model_cfg.latent_dim)
    gen = torch.Generator().manual_seed(0)
    for _ in range(memory_cfg.recent_size * 2):
        mem.observe(
            torch.randn(1, h_dim, generator=gen),
            torch.randn(1, model_cfg.latent_dim, generator=gen),
        )
    h = torch.randn(1, h_dim, generator=gen)
    z = torch.randn(1, model_cfg.latent_dim, generator=gen)
    # Warmup (allocator, kernel caches).
    for _ in range(10):
        mem.contextualize(h, z)
    start = time.perf_counter()
    for _ in range(_ITERATIONS):
        mem.contextualize(h, z)
    mean_ms = (time.perf_counter() - start) / _ITERATIONS * 1000.0
    budget = _resolve_budget_ms()
    assert mean_ms < budget, f"contextualize mean {mean_ms:.3f} ms >= budget {budget} ms"
