"""Phase 2.1 — RSSM pretraining on synthetic observation sequences.

Usage:
    python -m training.train_rssm --config config/mock_hardware.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mousedroid.config.schema import Settings
from mousedroid.world_model.rssm import RSSM
from training.rssm_dataset import RSSMSequenceDataset

_log = structlog.get_logger(__name__)


def train_rssm(
    cfg: Settings,
    data_path: Path,
    device: torch.device | None = None,
) -> Path:
    """Train RSSM encoder + dynamics on synthetic data.

    Args:
        cfg: Root settings with training and model configs.
        data_path: Path to ``sequences.pt`` file.
        device: Torch device (defaults to CPU).

    Returns:
        Path to the final checkpoint.
    """
    device = device or torch.device("cpu")
    tcfg = cfg.training
    mcfg = cfg.model

    # Build model
    rssm = RSSM(mcfg).to(device)
    optimizer = torch.optim.Adam(rssm.parameters(), lr=tcfg.learning_rate)

    # Build dataset
    dataset = RSSMSequenceDataset(data_path, seq_len=tcfg.sequence_length)
    loader = DataLoader(
        dataset,
        batch_size=tcfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Checkpoint directory
    weights_dir = Path(tcfg.weights_dir) / "rssm"
    weights_dir.mkdir(parents=True, exist_ok=True)

    mse_loss_fn = nn.MSELoss()

    for epoch in range(1, tcfg.epochs + 1):
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for vision, ultrasonic, motor_state, valid_mask, actions in loader:
            vision = vision.to(device)
            ultrasonic = ultrasonic.to(device)
            motor_state = motor_state.to(device)
            valid_mask = valid_mask.to(device)
            actions = actions.to(device)

            batch_size = vision.shape[0]
            seq_len = vision.shape[1]

            # Init latent state
            h = torch.zeros(batch_size, mcfg.hidden_dim, device=device)
            z = torch.zeros(batch_size, mcfg.latent_dim, device=device)

            total_recon = torch.tensor(0.0, device=device)
            total_kl = torch.tensor(0.0, device=device)

            for t in range(seq_len):
                # Encode observation
                obs_embed = rssm.encoder(
                    vision[:, t],
                    ultrasonic[:, t],
                    motor_state[:, t],
                    valid_mask[:, t],
                )

                # GRU step
                prev_action = actions[:, max(0, t - 1)]
                gru_input = torch.cat([z, prev_action], dim=-1)
                h = rssm.gru(gru_input, h)

                # Posterior
                post_params = rssm.posterior(torch.cat([h, obs_embed], dim=-1))
                z, post_mean, post_logvar = rssm._sample_gaussian(post_params)

                # Prior
                prior_params = rssm.prior(h)
                _, prior_mean, prior_logvar = rssm._sample_gaussian(prior_params)

                # Reconstruction loss
                obs_recon = rssm.decode(h, z)
                total_recon = total_recon + mse_loss_fn(obs_recon, obs_embed)

                # KL loss
                kl = rssm._kl_divergence(
                    post_mean,
                    post_logvar,
                    prior_mean,
                    prior_logvar,
                )
                total_kl = total_kl + kl

            # Average over sequence length
            total_recon = total_recon / seq_len
            total_kl = total_kl / seq_len

            loss = total_recon + tcfg.kl_beta * total_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_recon += total_recon.item()
            epoch_kl += total_kl.item()
            n_batches += 1

        avg_recon = epoch_recon / max(n_batches, 1)
        avg_kl = epoch_kl / max(n_batches, 1)

        _log.info(
            "rssm_epoch",
            epoch=epoch,
            recon_loss=round(avg_recon, 6),
            kl_loss=round(avg_kl, 6),
        )

        # Checkpoint
        if epoch % tcfg.checkpoint_every_n == 0:
            ckpt_path = weights_dir / f"epoch_{epoch}.pt"
            torch.save(rssm.state_dict(), ckpt_path)
            _log.info("checkpoint_saved", path=str(ckpt_path), epoch=epoch)

    # Save final
    final_path = weights_dir / "final.pt"
    torch.save(rssm.state_dict(), final_path)
    _log.info("training_complete", final_checkpoint=str(final_path))
    return final_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="RSSM pretraining")
    parser.add_argument(
        "--config",
        type=str,
        default="config/mock_hardware.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to sequences.pt (default: cfg.training.data_dir/sequences.pt)",
    )
    args = parser.parse_args()

    # Load config
    import yaml

    with open(args.config) as f:
        overrides = yaml.safe_load(f) or {}
    cfg = Settings(**overrides)

    data_path = Path(args.data) if args.data else Path(cfg.training.data_dir) / "sequences.pt"

    train_rssm(cfg, data_path)


if __name__ == "__main__":
    main()
