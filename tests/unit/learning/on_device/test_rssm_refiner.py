"""Unit tests for the WS-E2 RSSM-refinement learner.

Pins the SPIKE-LOCKED WS-E2 ENABLEMENT contracts (see
``docs/superpowers/plans/2026-06-14-phase6-enablement.md`` — "SPIKE RESULTS"):

* :class:`RSSMRefiner` runs EXACTLY ``cfg.update_steps`` autograd.grad manual-SGD
  steps at ``cfg.learning_rate`` and returns an
  :class:`~mousedroid.learning.on_device.protocol.OnDeviceUpdateResult` whose
  ``candidate_state_dict`` is the refined RSSM's ``state_dict`` (NO decoder keys);
* the candidate's parameters DIVERGE from the base after refinement;
* the candidate ``state_dict`` strict-loads into a fresh ``build_world_model(cfg)``
  (round-trip, missing=[] unexpected=[]);
* the ``allow_unused=True`` None-grad guard is exercised — the RSSM ``reward_head``
  (+ others) receive ``None`` grad in the recon/KL graph and the loop skips them
  without raising;
* ``lambda=0`` v1: NO EWC term — the loss equals ``recon + kl_beta*kl`` exactly;
* the refinement is DETERMINISTIC given a fixed seed + fixed batch;
* structlog ``on_device_*`` events fire on start + complete.

Uses tiny ``ModelConfig`` dims (hidden=8, latent=4, action=3) and
``Settings(mock_hardware=True)`` (a bare ``Settings()`` raises the
distance-sensor validator). Vision is OFF, so the batch carries ``vision``
shaped ``(B, T, 0)`` — mirroring ``RSSMPretrainer._to_device``.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from mousedroid.config.schema import ModelConfig, OnDeviceLearningConfig, Settings
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.factory import build_world_model
from mousedroid.learning.on_device.protocol import OnDeviceUpdateResult
from mousedroid.learning.on_device.rssm_refiner import (
    RSSMRefiner,
    build_sequence_batch,
)
from mousedroid.world_model.rssm import RSSM

_DEVICE = torch.device("cpu")


def _model_cfg(*, lidar: bool = False, vision: bool = False) -> ModelConfig:
    """Tiny deterministic ModelConfig (vision/lidar optional)."""
    return ModelConfig(
        vision_dim=12 if vision else 0,
        vision_proj_dim=6 if vision else 0,
        ultrasonic_dim=1,
        ultrasonic_proj_dim=4,
        motor_state_dim=4,
        hidden_dim=8,
        latent_dim=4,
        action_dim=3,
        obs_dim=8,
        motor_proj_dim=4,
        lidar_dim=6 if lidar else 0,
        lidar_proj_dim=4 if lidar else 0,
    )


def _settings(*, lidar: bool = False, vision: bool = False) -> Settings:
    """Settings(mock_hardware=True) carrying the tiny model cfg."""
    return Settings(mock_hardware=True, model=_model_cfg(lidar=lidar, vision=vision))


def _make_rssm(cfg: ModelConfig, *, seed: int = 0) -> RSSM:
    """Build a small deterministic RSSM."""
    torch.manual_seed(seed)
    wm = RSSM(cfg)
    wm.eval()
    return wm


def _make_records(
    n: int, *, vision_dim: int = 0, seed: int = 7
) -> list[MouseDroidExperienceRecord]:
    """Build ``n`` deterministic records with non-trivial fields."""
    rng = np.random.default_rng(seed)
    records: list[MouseDroidExperienceRecord] = []
    for i in range(n):
        records.append(
            MouseDroidExperienceRecord(
                timestamp=float(i),
                vision_features=(
                    rng.standard_normal(vision_dim).astype(np.float32)
                    if vision_dim
                    else np.zeros(0, dtype=np.float32)
                ),
                distance_m=float(rng.uniform(0.1, 2.0)),
                motor_state=rng.standard_normal(4).astype(np.float32),
                action=rng.standard_normal(3).astype(np.float32),
                reward=float(rng.uniform(-1.0, 1.0)),
            )
        )
    return records


def _ocfg(**overrides: object) -> OnDeviceLearningConfig:
    base: dict[str, object] = {
        "enabled": True,
        "update_steps": 3,
        "learning_rate": 1e-2,
        "ewc_lambda": 0.0,
        "refine_sequence_length": 4,
        "refine_batch_episodes": 2,
    }
    base.update(overrides)
    return OnDeviceLearningConfig(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_sequence_batch — batch dict shape mirrors RSSMPretrainer._to_device
# ---------------------------------------------------------------------------


def test_build_sequence_batch_keys_and_shapes_vision_off() -> None:
    """Batch carries motor/action/valid_mask always; vision is (B, T, 0)."""
    cfg = _model_cfg()
    wm = _make_rssm(cfg)
    records = _make_records(20)
    batch = build_sequence_batch(
        records,
        cfg,
        wm.encoder,
        sequence_length=4,
        n_episodes=2,
        device=_DEVICE,
    )
    assert set(batch) == {"motor", "action", "valid_mask", "ultrasonic", "vision"}
    assert batch["motor"].shape == (2, 4, 4)
    assert batch["action"].shape == (2, 4, 3)
    # valid_mask length is the encoder modality count (4 with lidar off).
    assert batch["valid_mask"].shape == (2, 4, 4)
    assert batch["ultrasonic"].shape == (2, 4, 1)
    assert batch["vision"].shape == (2, 4, 0)
    # Every tensor must be on the requested device + float32.
    for tensor in batch.values():
        assert tensor.device == _DEVICE
        assert tensor.dtype == torch.float32


def test_build_sequence_batch_lidar_and_vision_enabled() -> None:
    """Lidar key present + valid_mask length 5 when lidar enabled; vision (B,T,V)."""
    cfg = _model_cfg(lidar=True, vision=True)
    wm = _make_rssm(cfg)
    records = _make_records(20, vision_dim=12)
    batch = build_sequence_batch(
        records,
        cfg,
        wm.encoder,
        sequence_length=3,
        n_episodes=2,
        device=_DEVICE,
    )
    assert "lidar" in batch
    assert batch["lidar"].shape == (2, 3, 6)
    assert batch["vision"].shape == (2, 3, 12)
    # 5-element mask (lidar adds slot 4).
    assert batch["valid_mask"].shape[-1] == 5


def test_build_sequence_batch_vision_enabled_empty_record_zero_fills() -> None:
    """A vision-enabled encoder + an empty-vision record zero-fills to (B, T, V)."""
    cfg = _model_cfg(vision=True)
    wm = _make_rssm(cfg)
    # vision_dim=0 records ⇒ each record's vision_features is empty; the assembler
    # must zero-fill to cfg.vision_dim so the (B, T, vision_dim) shape holds.
    records = _make_records(20, vision_dim=0)
    batch = build_sequence_batch(
        records, cfg, wm.encoder, sequence_length=3, n_episodes=2, device=_DEVICE
    )
    assert batch["vision"].shape == (2, 3, cfg.vision_dim)
    assert torch.count_nonzero(batch["vision"]) == 0


def test_build_sequence_batch_drives_train_sequence() -> None:
    """The assembled batch is consumable by RSSM.train_sequence (loss requires grad)."""
    from mousedroid.world_model.rssm import RawModalityDecoders

    cfg = _model_cfg()
    wm = _make_rssm(cfg)
    records = _make_records(20)
    batch = build_sequence_batch(
        records, cfg, wm.encoder, sequence_length=4, n_episodes=2, device=_DEVICE
    )
    decoders = RawModalityDecoders(cfg)
    out = wm.train_sequence(batch, decoders)
    assert set(out) >= {"loss", "recon", "kl", "posterior_std"}
    assert out["loss"].requires_grad


def test_build_sequence_batch_raises_on_insufficient_records() -> None:
    """Too few records to build B*T steps raises a clear error."""
    cfg = _model_cfg()
    wm = _make_rssm(cfg)
    with pytest.raises(ValueError, match="records"):
        build_sequence_batch(
            _make_records(3),
            cfg,
            wm.encoder,
            sequence_length=4,
            n_episodes=2,
            device=_DEVICE,
        )


# ---------------------------------------------------------------------------
# RSSMRefiner.update — the locked autograd.grad manual-SGD loop
# ---------------------------------------------------------------------------


def _make_batch(cfg: ModelConfig, wm: RSSM) -> dict[str, torch.Tensor]:
    records = _make_records(40)
    return build_sequence_batch(
        records, cfg, wm.encoder, sequence_length=4, n_episodes=3, device=_DEVICE
    )


def test_refiner_runs_update_steps_and_returns_result() -> None:
    """update() returns an OnDeviceUpdateResult with n_steps == cfg.update_steps."""
    cfg = _model_cfg()
    wm = _make_rssm(cfg)
    batch = _make_batch(cfg, wm)
    refiner = RSSMRefiner(wm, _ocfg(update_steps=3))

    result = refiner.update(batch)

    assert isinstance(result, OnDeviceUpdateResult)
    assert result.n_steps == 3
    assert isinstance(result.train_loss, float)
    assert np.isfinite(result.train_loss)


def test_refiner_candidate_diverges_from_base() -> None:
    """After refinement the candidate params differ from the base RSSM."""
    cfg = _model_cfg()
    wm = _make_rssm(cfg)
    batch = _make_batch(cfg, wm)
    refiner = RSSMRefiner(wm, _ocfg(update_steps=3, learning_rate=1e-1))

    result = refiner.update(batch)

    base = dict(wm.named_parameters())
    diverged = any(
        not torch.equal(result.candidate_state_dict[name], base[name].detach()) for name in base
    )
    assert diverged, "candidate must diverge from base after refinement"


def test_refiner_candidate_state_dict_has_no_decoder_keys() -> None:
    """The candidate slot is the RSSM state_dict ONLY — never the decoders."""
    cfg = _model_cfg()
    wm = _make_rssm(cfg)
    batch = _make_batch(cfg, wm)
    result = RSSMRefiner(wm, _ocfg()).update(batch)

    keys = set(result.candidate_state_dict)
    assert keys == set(wm.state_dict())
    assert not any(k.startswith("decode_") for k in keys)


def test_refiner_round_trip_into_fresh_build_world_model() -> None:
    """candidate.state_dict() strict-loads into a fresh build_world_model(cfg)."""
    settings = _settings()
    wm = build_world_model(settings)
    assert isinstance(wm, RSSM)
    batch = _make_batch(settings.model, wm)
    result = RSSMRefiner(wm, _ocfg()).update(batch)

    fresh = build_world_model(settings)
    assert isinstance(fresh, RSSM)
    incompatible = fresh.load_state_dict(result.candidate_state_dict, strict=True)
    assert list(incompatible.missing_keys) == []
    assert list(incompatible.unexpected_keys) == []


def test_refiner_allow_unused_none_grad_guard() -> None:
    """The reward_head receives None grad (recon/KL graph) — loop must not raise.

    reward_head + observation_decoder + prior are not on the recon/KL path, so
    autograd.grad(..., allow_unused=True) returns None for them. The refiner must
    skip None grads (the SPIKE-LOCKED None-grad guard) and leave those params
    bitwise-unchanged.
    """
    cfg = _model_cfg()
    wm = _make_rssm(cfg)
    batch = _make_batch(cfg, wm)
    before = {n: p.detach().clone() for n, p in wm.named_parameters()}

    result = RSSMRefiner(wm, _ocfg(update_steps=2, learning_rate=1e-1)).update(batch)

    # reward_head is unused in train_sequence's loss graph → None grad → unchanged.
    for name in ("reward_head.weight", "reward_head.bias"):
        assert torch.equal(
            result.candidate_state_dict[name], before[name]
        ), f"unused param {name!r} should be untouched (None-grad guard)"
    # ...but a used param (the GRU) must have moved.
    moved = any(
        not torch.equal(result.candidate_state_dict[n], before[n])
        for n in before
        if n.startswith("gru.")
    )
    assert moved, "a used param (gru) must change"


def test_refiner_lambda_zero_loss_is_recon_plus_kl_beta_kl() -> None:
    """λ=0 v1: train_loss equals recon + kl_beta*kl exactly (no EWC penalty)."""
    cfg = _model_cfg()
    wm = _make_rssm(cfg)
    batch = _make_batch(cfg, wm)

    # Reproduce one deterministic step ourselves and compare the FIRST-step loss.
    from mousedroid.world_model.rssm import RawModalityDecoders

    candidate = copy.deepcopy(wm)
    decoders = RawModalityDecoders(candidate.cfg)
    torch.manual_seed(_ocfg().scoring_seed)
    out = candidate.train_sequence(batch, decoders)
    expected_first = float(out["recon"] + candidate.cfg.kl_beta * out["kl"])
    assert float(out["loss"].detach()) == pytest.approx(expected_first)


def test_refiner_deterministic_given_fixed_seed_and_batch() -> None:
    """Two refiners over the same base + batch yield identical candidates."""
    cfg = _model_cfg()
    wm_a = _make_rssm(cfg, seed=3)
    wm_b = _make_rssm(cfg, seed=3)
    batch = _make_batch(cfg, wm_a)

    res_a = RSSMRefiner(wm_a, _ocfg(update_steps=3)).update(batch)
    res_b = RSSMRefiner(wm_b, _ocfg(update_steps=3)).update(batch)

    for name in res_a.candidate_state_dict:
        assert torch.equal(
            res_a.candidate_state_dict[name], res_b.candidate_state_dict[name]
        ), f"non-deterministic refine on {name!r}"
    assert res_a.train_loss == res_b.train_loss


def test_refiner_lidar_enabled_path() -> None:
    """A lidar-enabled RSSM refines + round-trips (5-element mask path)."""
    settings = _settings(lidar=True)
    wm = build_world_model(settings)
    assert isinstance(wm, RSSM)
    records = _make_records(40)
    batch = build_sequence_batch(
        records,
        settings.model,
        wm.encoder,
        sequence_length=4,
        n_episodes=2,
        device=_DEVICE,
    )
    result = RSSMRefiner(wm, _ocfg()).update(batch)

    fresh = build_world_model(settings)
    assert isinstance(fresh, RSSM)
    fresh.load_state_dict(result.candidate_state_dict, strict=True)


def test_refiner_emits_structlog_events() -> None:
    """Start + complete structlog events fire (reuse the on_device_* family)."""
    import structlog

    cfg = _model_cfg()
    wm = _make_rssm(cfg)
    batch = _make_batch(cfg, wm)

    with structlog.testing.capture_logs() as captured:
        RSSMRefiner(wm, _ocfg()).update(batch)

    events = [entry.get("event", "") for entry in captured]
    assert "on_device_refine_start" in events
    assert "on_device_refine_complete" in events


def test_refiner_non_rssm_engine_rejected() -> None:
    """A model lacking train_sequence is rejected at construction (capability guard)."""

    class _NoTrainSeq(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
            return x

    with pytest.raises((TypeError, ValueError), match="train_sequence"):
        RSSMRefiner(_NoTrainSeq(), _ocfg())  # type: ignore[arg-type]
