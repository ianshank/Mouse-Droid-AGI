"""Unit tests for ``build_latent_context`` (F-023).

Pins the factory None-gate (absent / disabled ⇒ ``None``, byte-identical
orchestrator path) and the engine-agnostic dimension contract
(``h_dim = hidden_dim + cfc_hidden_dim``).
"""

from __future__ import annotations

from mousedroid.config.schema import Settings
from mousedroid.factory import build_latent_context
from mousedroid.world_model.bounded_context import BoundedContextMemory
from mousedroid.world_model.protocol import LatentContextProtocol


def test_returns_none_when_block_absent() -> None:
    cfg = Settings.model_validate({"mock_hardware": True})
    assert build_latent_context(cfg) is None


def test_returns_none_when_disabled() -> None:
    cfg = Settings.model_validate({"mock_hardware": True, "world_model_memory": {"enabled": False}})
    assert build_latent_context(cfg) is None


def test_returns_memory_when_enabled() -> None:
    cfg = Settings.model_validate({"mock_hardware": True, "world_model_memory": {"enabled": True}})
    ctx = build_latent_context(cfg)
    assert isinstance(ctx, BoundedContextMemory)
    assert isinstance(ctx, LatentContextProtocol)


def test_dims_match_combined_carried_state() -> None:
    """h_dim must include cfc_hidden_dim — the orchestrator carries the
    combined vector for the dual-stream engine."""
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "world_model_memory": {"enabled": True},
            "model": {"hidden_dim": 32, "cfc_hidden_dim": 16, "latent_dim": 8},
        }
    )
    ctx = build_latent_context(cfg)
    assert isinstance(ctx, BoundedContextMemory)
    assert ctx._h_dim == 32 + 16
    assert ctx._z_dim == 8
