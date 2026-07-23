"""Unit tests for ``training/drift_metrics.py`` (F-023).

Pins determinism (same seed ⇒ byte-identical report; global RNG restored),
the RSSM capability narrow, metric honesty (per-modality separation, range
headline, zero-fill exclusion, ``valid_mask`` threading), the residual-head
consumption channel, and the B=1 contract for the ``latent_context`` hook.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from mousedroid.config.schema import ModelConfig, WorldModelMemoryConfig
from mousedroid.constants import SENSOR_SLOT_MAP
from mousedroid.training.drift_metrics import measure_drift
from mousedroid.world_model.bounded_context import BoundedContextMemory
from mousedroid.world_model.rssm import RSSM, DriftCorrectionHead, RawModalityDecoders
from tests.unit._rssm_drift_helpers import seeded_model_pair, seq_batch, tiny_rssm_cfg

_B = 3
_T = 16
_CONTEXT = 4
_HORIZON = 8


def _tiny_cfg() -> ModelConfig:
    return tiny_rssm_cfg()


def _batch(mcfg: ModelConfig, *, episodes: int = _B, seed: int = 0) -> dict[str, torch.Tensor]:
    return seq_batch(mcfg, episodes=episodes, seq_len=_T, seed=seed)


def _model_pair(seed: int = 7) -> tuple[RSSM, RawModalityDecoders]:
    return seeded_model_pair(seed)


def _measure(**overrides: object) -> object:
    model, decoders = _model_pair()
    batch = _batch(model.cfg)
    kwargs: dict[str, object] = {
        "context_steps": _CONTEXT,
        "horizon": _HORIZON,
        "seed": 42,
    }
    kwargs.update(overrides)
    return measure_drift(model, batch, decoders, **kwargs)  # type: ignore[arg-type]


class TestDeterminism:
    def test_same_seed_byte_identical(self) -> None:
        model, decoders = _model_pair()
        batch = _batch(model.cfg)
        r1 = measure_drift(
            model, batch, decoders, context_steps=_CONTEXT, horizon=_HORIZON, seed=42
        )
        r2 = measure_drift(
            model, batch, decoders, context_steps=_CONTEXT, horizon=_HORIZON, seed=42
        )
        assert r1.per_step_mse == r2.per_step_mse

    def test_global_rng_restored(self) -> None:
        model, decoders = _model_pair()
        batch = _batch(model.cfg)
        torch.manual_seed(777)
        before = torch.get_rng_state()
        measure_drift(model, batch, decoders, context_steps=_CONTEXT, horizon=_HORIZON, seed=42)
        assert torch.equal(before, torch.get_rng_state())

    def test_train_mode_restored(self) -> None:
        model, decoders = _model_pair()
        batch = _batch(model.cfg)
        model.train()
        decoders.train()
        measure_drift(model, batch, decoders, context_steps=_CONTEXT, horizon=_HORIZON, seed=42)
        assert model.training
        assert decoders.training


class TestContracts:
    def test_non_rssm_rejected(self) -> None:
        _, decoders = _model_pair()
        batch = _batch(_tiny_cfg())
        with pytest.raises(TypeError, match="concrete RSSM"):
            measure_drift(
                MagicMock(name="not_an_rssm"),
                batch,
                decoders,
                context_steps=_CONTEXT,
                horizon=_HORIZON,
                seed=42,
            )

    def test_short_batch_rejected(self) -> None:
        model, decoders = _model_pair()
        batch = _batch(model.cfg)
        with pytest.raises(ValueError, match="context_steps"):
            measure_drift(model, batch, decoders, context_steps=_T, horizon=_HORIZON, seed=42)

    def test_channels_and_headline(self) -> None:
        report = _measure()
        assert set(report.channels()) == {"motor", "range", "latent_h", "latent_z"}  # type: ignore[attr-defined]
        assert report.headline_channel == "range"  # type: ignore[attr-defined]
        for channel in report.channels():  # type: ignore[attr-defined]
            curve = report.per_step_mse[channel]  # type: ignore[attr-defined]
            assert len(curve) == _HORIZON
            assert all(v >= 0.0 for v in curve)

    def test_valid_mask_threaded(self) -> None:
        """Zeroing the range mask must change the range score (masked samples drop)."""
        model, decoders = _model_pair()
        batch = _batch(model.cfg)
        masked = {k: v.clone() for k, v in batch.items()}
        # Invalidate range for all but the first episode.
        masked["valid_mask"][1:, :, SENSOR_SLOT_MAP["ultrasonic"]] = 0.0
        full = measure_drift(
            model, batch, decoders, context_steps=_CONTEXT, horizon=_HORIZON, seed=42
        )
        partial = measure_drift(
            model, masked, decoders, context_steps=_CONTEXT, horizon=_HORIZON, seed=42
        )
        assert full.per_step_mse["range"] != partial.per_step_mse["range"]

    def test_residual_head_adds_corrected_channel(self) -> None:
        torch.manual_seed(1)
        head = DriftCorrectionHead(_tiny_cfg())
        report = _measure(residual_head=head)
        assert "motor_corrected" in report.channels()  # type: ignore[attr-defined]


class TestLatentContextHook:
    def _memory(self) -> BoundedContextMemory:
        cfg = WorldModelMemoryConfig.model_validate(
            {"enabled": True, "sink_warmup_ticks": 0, "blend_weight": 0.3}
        )
        mcfg = _tiny_cfg()
        return BoundedContextMemory(cfg, h_dim=mcfg.hidden_dim, z_dim=mcfg.latent_dim)

    def test_requires_batch_size_one(self) -> None:
        model, decoders = _model_pair()
        batch = _batch(model.cfg)  # B = 3
        with pytest.raises(ValueError, match="batch size 1"):
            measure_drift(
                model,
                batch,
                decoders,
                context_steps=_CONTEXT,
                horizon=_HORIZON,
                seed=42,
                latent_context=self._memory(),
            )

    def test_memory_changes_report_on_single_episode(self) -> None:
        model, decoders = _model_pair()
        batch = _batch(model.cfg, episodes=1)
        off = measure_drift(
            model, batch, decoders, context_steps=_CONTEXT, horizon=_HORIZON, seed=42
        )
        on = measure_drift(
            model,
            batch,
            decoders,
            context_steps=_CONTEXT,
            horizon=_HORIZON,
            seed=42,
            latent_context=self._memory(),
        )
        assert off.per_step_mse["latent_h"] != on.per_step_mse["latent_h"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only harmonization path")
def test_cuda_model_scores_cpu_batch() -> None:
    """measure_drift harmonises a CPU-built batch to the model's device."""
    model, decoders = _model_pair()
    model = model.cuda()
    decoders = decoders.cuda()
    batch = _batch(model.cfg)  # CPU tensors
    report = measure_drift(
        model, batch, decoders, context_steps=_CONTEXT, horizon=_HORIZON, seed=42
    )
    assert all(v >= 0.0 for curve in report.per_step_mse.values() for v in curve)
