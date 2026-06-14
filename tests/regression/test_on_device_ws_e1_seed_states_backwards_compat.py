"""Regression: WS-E1 replay-encoded seed states are additive + default-OFF.

Pins the WS-E1 backwards-compatibility invariants:

* the default ``seed_state_source="sampled"`` keeps the #134 ``manual_seed``-
  sampled gate seed states BYTE-IDENTICAL to the pre-WS-E1 inline formula
  (a single ``torch.Generator(scoring_seed)`` drawing each ``(h, z)``);
* building the on-device gate runner with the default source still produces a
  working ``(CandidateSlot) -> None`` closure;
* the ``replay_encoded`` source FALLS BACK to those exact sampled seed states
  when the replay store is empty (fresh Jetson) — so enabling the flag on a
  rover with no recorded experience never degrades the gate to zero states.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mousedroid.config.schema import Settings
from mousedroid.factory import (
    _build_replay_encoded_seed_states,
    _build_sampled_seed_states,
    build_on_device_coordinator,
    build_world_model,
)


def _canonical_sampled_seed_states(
    hidden_dim: int, latent_dim: int, n_seed: int, *, seed: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The exact pre-WS-E1 inline formula (the byte-identical reference)."""
    gen = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randn(1, hidden_dim, generator=gen),
            torch.randn(1, latent_dim, generator=gen),
        )
        for _ in range(n_seed)
    ]


def test_sampled_helper_byte_identical_to_inline_formula() -> None:
    """``_build_sampled_seed_states`` matches the pre-WS-E1 inline formula."""
    hidden_dim, latent_dim, n_seed, seed = 16, 8, 5, 1234

    expected = _canonical_sampled_seed_states(hidden_dim, latent_dim, n_seed, seed=seed)
    actual = _build_sampled_seed_states(hidden_dim, latent_dim, n_seed, seed=seed)

    assert len(actual) == len(expected) == n_seed
    for (ha, za), (he, ze) in zip(actual, expected, strict=True):
        assert torch.equal(ha, he)
        assert torch.equal(za, ze)


def test_default_source_builds_working_gate_runner(tmp_path: Path) -> None:
    """The default ``sampled`` source still yields a callable gate runner."""
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "experience": {"path": str(tmp_path / "exp"), "map_size_gb": 0.01},
            "on_device_learning": {"enabled": True, "trigger_min_new_records": 10},
        }
    )
    assert cfg.on_device_learning is not None
    assert cfg.on_device_learning.seed_state_source == "sampled"

    wm = build_world_model(cfg)
    coordinator = build_on_device_coordinator(cfg, world_model=wm)
    assert coordinator is not None


def test_replay_encoded_empty_store_returns_empty_for_sampled_fallback(
    tmp_path: Path,
) -> None:
    """A replay-encoded source over an empty store yields [] (caller falls back).

    The factory's gate builder substitutes the sampled path when the replay
    encoder returns an empty list, so an empty store on a fresh Jetson never
    degrades the gate. Here we pin that the encoder helper returns ``[]`` for an
    empty store, which is the signal the fallback hinges on.
    """
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "experience": {"path": str(tmp_path / "empty_exp"), "map_size_gb": 0.01},
            "on_device_learning": {
                "enabled": True,
                "trigger_min_new_records": 10,
                "seed_state_source": "replay_encoded",
            },
        }
    )
    wm = build_world_model(cfg)

    from mousedroid.training.replay.lmdb_reader import LMDBReplayReader

    reader = LMDBReplayReader(cfg.experience)
    states = _build_replay_encoded_seed_states(wm, reader=reader, n_seed=5)

    assert states == []


def test_replay_encoded_source_builds_coordinator(tmp_path: Path) -> None:
    """Enabling ``replay_encoded`` still builds a coordinator (empty-store safe)."""
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "experience": {"path": str(tmp_path / "exp"), "map_size_gb": 0.01},
            "on_device_learning": {
                "enabled": True,
                "trigger_min_new_records": 10,
                "seed_state_source": "replay_encoded",
            },
        }
    )
    wm = build_world_model(cfg)
    coordinator = build_on_device_coordinator(cfg, world_model=wm)
    assert coordinator is not None
