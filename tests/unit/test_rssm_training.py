"""Tests for Phase 2.1 — RSSM pretraining components."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mousedroid.config.schema import ModelConfig, Settings, TrainingConfig
from mousedroid.world_model.rssm import RSSM


def _make_model_cfg() -> ModelConfig:
    return ModelConfig(
        vision_dim=16,
        ultrasonic_dim=1,
        motor_state_dim=4,
        hidden_dim=32,
        latent_dim=8,
        action_dim=3,
        obs_dim=16,
        vision_proj_dim=8,
        ultrasonic_proj_dim=4,
        motor_proj_dim=4,
    )


class TestRSSMDecoder:
    """Test the new observation_decoder head on RSSM."""

    def test_decoder_exists(self) -> None:
        cfg = _make_model_cfg()
        rssm = RSSM(cfg)
        assert hasattr(rssm, "observation_decoder")

    def test_decode_output_shape(self) -> None:
        cfg = _make_model_cfg()
        rssm = RSSM(cfg)
        h = torch.randn(1, cfg.hidden_dim)
        z = torch.randn(1, cfg.latent_dim)
        recon = rssm.decode(h, z)
        assert recon.shape == (1, cfg.obs_dim)

    def test_decode_batch(self) -> None:
        cfg = _make_model_cfg()
        rssm = RSSM(cfg)
        h = torch.randn(4, cfg.hidden_dim)
        z = torch.randn(4, cfg.latent_dim)
        recon = rssm.decode(h, z)
        assert recon.shape == (4, cfg.obs_dim)

    def test_decode_gradient_flows(self) -> None:
        cfg = _make_model_cfg()
        rssm = RSSM(cfg)
        h = torch.randn(1, cfg.hidden_dim, requires_grad=True)
        z = torch.randn(1, cfg.latent_dim, requires_grad=True)
        recon = rssm.decode(h, z)
        loss = recon.sum()
        loss.backward()
        assert h.grad is not None
        assert z.grad is not None


class TestRSSMTrainingLoop:
    """Test RSSM training on tiny synthetic data."""

    def test_loss_decreases(self) -> None:
        cfg = _make_model_cfg()
        rssm = RSSM(cfg)
        optimizer = torch.optim.Adam(rssm.parameters(), lr=1e-3)

        # Generate tiny data: 2 sequences of length 5
        batch_size = 2
        seq_len = 5
        vision = torch.randn(batch_size, seq_len, cfg.vision_dim)
        ultrasonic = torch.randn(batch_size, seq_len, 1)
        motor_state = torch.randn(batch_size, seq_len, cfg.motor_state_dim)
        valid_mask = torch.ones(batch_size, seq_len, 3)
        actions = torch.randn(batch_size, seq_len, cfg.action_dim)

        losses = []
        for _epoch in range(5):
            h = torch.zeros(batch_size, cfg.hidden_dim)
            z = torch.zeros(batch_size, cfg.latent_dim)
            total_loss = torch.tensor(0.0)

            for t in range(seq_len):
                obs_embed = rssm.encoder(
                    vision[:, t], ultrasonic[:, t], motor_state[:, t], valid_mask[:, t],
                )
                prev_action = actions[:, max(0, t - 1)]
                gru_input = torch.cat([z, prev_action], dim=-1)
                h = rssm.gru(gru_input, h)

                post_params = rssm.posterior(torch.cat([h, obs_embed], dim=-1))
                z, pm, plv = rssm._sample_gaussian(post_params)

                prior_params = rssm.prior(h)
                _, prm, prlv = rssm._sample_gaussian(prior_params)

                recon = rssm.decode(h, z)
                total_loss = total_loss + torch.nn.functional.mse_loss(recon, obs_embed)
                total_loss = total_loss + rssm._kl_divergence(pm, plv, prm, prlv)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())

        assert losses[-1] < losses[0], "Loss should decrease during training"

    def test_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        cfg = _make_model_cfg()
        rssm = RSSM(cfg)

        ckpt = tmp_path / "rssm.pt"
        torch.save(rssm.state_dict(), ckpt)

        rssm2 = RSSM(cfg)
        rssm2.load_state_dict(torch.load(ckpt, weights_only=True))

        # Compare weights
        for (k1, v1), (k2, v2) in zip(
            rssm.state_dict().items(), rssm2.state_dict().items(),
        ):
            assert k1 == k2
            assert torch.allclose(v1, v2)


class TestTrainingConfig:
    """Test extended TrainingConfig fields."""

    def test_new_fields_have_defaults(self) -> None:
        cfg = TrainingConfig()
        assert cfg.kl_beta == 1.0
        assert cfg.sequence_length == 50
        assert cfg.n_episodes == 1000
        assert cfg.data_dir == "training/data"
        assert cfg.weights_dir == "weights"

    def test_settings_includes_ppo(self) -> None:
        cfg = Settings(mock_hardware=True)
        assert hasattr(cfg, "ppo")
        assert cfg.ppo.clip_epsilon == 0.2
