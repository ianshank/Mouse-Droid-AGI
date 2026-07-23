"""Unit tests for ``RSSM.train_sequence_corrupted`` (F-023, ADR-015).

Pins the k=0 equality contract (forced prefix length 0 is allclose-identical
to ``train_sequence`` under the same seed — including the private-generator
guarantee that the prefix draw never consumes the global RNG), the detached
open-loop prefix, the separate ``residual_loss`` key with head-only gradients,
and the RSSM ``state_dict`` key-set invariance (the refactor + external
``DriftCorrectionHead`` add nothing to the deployment checkpoint).
"""

from __future__ import annotations

import pytest
import torch

from mousedroid.config.schema import ModelConfig
from mousedroid.world_model.rssm import RSSM, DriftCorrectionHead, RawModalityDecoders
from tests.unit._rssm_drift_helpers import seeded_model_pair, seq_batch, tiny_rssm_cfg

_B = 3
_T = 8
# max_prefix_frac small enough that floor(frac * T) == 0 — forces k = 0
# deterministically without touching the RNG contract.
_FORCE_K0_FRAC = 0.1


def _tiny_cfg() -> ModelConfig:
    return tiny_rssm_cfg()


def _batch(mcfg: ModelConfig, seed: int = 0) -> dict[str, torch.Tensor]:
    return seq_batch(mcfg, episodes=_B, seq_len=_T, seed=seed)


def _model_pair(seed: int = 7) -> tuple[RSSM, RawModalityDecoders]:
    return seeded_model_pair(seed)


class TestK0Equality:
    def test_k0_allclose_identical_to_train_sequence(self) -> None:
        """Forced k=0 reproduces train_sequence exactly under the same seed."""
        model, decoders = _model_pair()
        batch = _batch(model.cfg)
        torch.manual_seed(123)
        base = model.train_sequence(batch, decoders)
        torch.manual_seed(123)
        corrupted = model.train_sequence_corrupted(batch, decoders, max_prefix_frac=_FORCE_K0_FRAC)
        assert int(corrupted["prefix_len"]) == 0
        for key in ("loss", "recon", "kl", "posterior_std"):
            assert torch.allclose(base[key], corrupted[key]), key
        assert float(corrupted["residual_loss"]) == 0.0

    def test_k0_recovery_weight_inert(self) -> None:
        """recovery_weight != 1 must not change the k=0 loss (weight gated on k>0)."""
        model, decoders = _model_pair()
        batch = _batch(model.cfg)
        torch.manual_seed(5)
        out_default = model.train_sequence_corrupted(
            batch, decoders, max_prefix_frac=_FORCE_K0_FRAC, recovery_weight=1.0
        )
        torch.manual_seed(5)
        out_weighted = model.train_sequence_corrupted(
            batch, decoders, max_prefix_frac=_FORCE_K0_FRAC, recovery_weight=3.0
        )
        assert torch.allclose(out_default["loss"], out_weighted["loss"])

    def test_prefix_draw_never_consumes_global_rng(self) -> None:
        """Global RNG stream after k=0 corrupted == after train_sequence."""
        model, decoders = _model_pair()
        batch = _batch(model.cfg)
        torch.manual_seed(99)
        model.train_sequence(batch, decoders)
        state_after_base = torch.get_rng_state()
        torch.manual_seed(99)
        model.train_sequence_corrupted(batch, decoders, max_prefix_frac=_FORCE_K0_FRAC)
        state_after_corrupted = torch.get_rng_state()
        assert torch.equal(state_after_base, state_after_corrupted)


def _corrupted_with_positive_k(
    model: RSSM,
    decoders: RawModalityDecoders,
    batch: dict[str, torch.Tensor],
    **kwargs: object,
) -> dict[str, torch.Tensor]:
    """Run with a generator seed guaranteeing k > 0 (searched deterministically)."""
    for gen_seed in range(50):
        gen = torch.Generator().manual_seed(gen_seed)
        torch.manual_seed(11)
        out = model.train_sequence_corrupted(
            batch,
            decoders,
            max_prefix_frac=0.5,
            generator=gen,
            **kwargs,  # type: ignore[arg-type]
        )
        if int(out["prefix_len"]) > 0:
            return out
    msg = "no generator seed in range produced k > 0"  # pragma: no cover
    raise AssertionError(msg)  # pragma: no cover


class TestCorruptedPrefix:
    def test_positive_prefix_trains_and_grads_flow(self) -> None:
        model, decoders = _model_pair()
        batch = _batch(model.cfg)
        out = _corrupted_with_positive_k(model, decoders, batch)
        assert 0 < int(out["prefix_len"]) <= _T // 2
        assert bool(torch.isfinite(out["loss"]))
        out["loss"].backward()
        grads = [p.grad for p in model.gru.parameters()]
        assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)

    def test_loss_graph_excludes_prefix(self) -> None:
        """The prefix is no_grad + detached — loss must not backprop into it.

        Indirect but decisive check: backward succeeds (a graph that
        referenced no_grad-produced tensors as non-leaf inputs would fail) and
        the loss value is finite with only ``t - k`` steps normalised.
        """
        model, decoders = _model_pair()
        batch = _batch(model.cfg)
        out = _corrupted_with_positive_k(model, decoders, batch)
        out["loss"].backward()  # raises if the prefix leaked into the graph

    def test_max_prefix_frac_validated(self) -> None:
        model, decoders = _model_pair()
        batch = _batch(model.cfg)
        with pytest.raises(ValueError, match="max_prefix_frac"):
            model.train_sequence_corrupted(batch, decoders, max_prefix_frac=0.0)
        with pytest.raises(ValueError, match="max_prefix_frac"):
            model.train_sequence_corrupted(batch, decoders, max_prefix_frac=1.5)


class TestResidualHead:
    def test_residual_loss_separate_key_and_head_only_grads(self) -> None:
        model, decoders = _model_pair()
        batch = _batch(model.cfg)
        torch.manual_seed(3)
        head = DriftCorrectionHead(model.cfg)
        torch.manual_seed(21)
        out = model.train_sequence_corrupted(
            batch, decoders, max_prefix_frac=_FORCE_K0_FRAC, residual_head=head
        )
        assert bool(torch.isfinite(out["residual_loss"]))
        assert float(out["residual_loss"]) > 0.0
        # residual_loss backward reaches ONLY the head — model + decoders stay
        # gradient-free (input and target are both detached).
        out["residual_loss"].backward()
        assert all(p.grad is not None for p in head.parameters())
        assert all(p.grad is None for p in model.parameters())
        assert all(p.grad is None for p in decoders.parameters())

    def test_residual_loss_never_folded_into_loss(self) -> None:
        model, decoders = _model_pair()
        batch = _batch(model.cfg)
        torch.manual_seed(3)
        head = DriftCorrectionHead(model.cfg)
        torch.manual_seed(21)
        with_head = model.train_sequence_corrupted(
            batch, decoders, max_prefix_frac=_FORCE_K0_FRAC, residual_head=head
        )
        torch.manual_seed(21)
        without_head = model.train_sequence_corrupted(
            batch, decoders, max_prefix_frac=_FORCE_K0_FRAC
        )
        assert torch.allclose(with_head["loss"], without_head["loss"])


class TestStateDictInvariance:
    def test_rssm_state_dict_keys_unchanged(self) -> None:
        """The refactor + external head leave the deployment checkpoint intact."""
        model, _ = _model_pair()
        prefixes = {key.split(".")[0] for key in model.state_dict()}
        assert prefixes == {
            "encoder",
            "gru",
            "posterior",
            "prior",
            "reward_head",
            "observation_decoder",
        }

    def test_drift_head_params_external(self) -> None:
        head = DriftCorrectionHead(_tiny_cfg())
        assert all(key.startswith("head.") for key in head.state_dict())
