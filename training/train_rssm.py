"""Phase 2.1 — RSSM pretraining on synthetic observation sequences.

GPU-accelerated with AMP support and checkpoint resume.

Usage:
    python -m training.train_rssm --config config/mock_hardware.yaml
    python -m training.train_rssm --config config/mock_hardware.yaml \
        --device cuda --resume weights/rssm/epoch_50.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mousedroid.config.schema import Settings
from mousedroid.world_model.rssm import RSSM
from training.gpu_utils import check_memory_budget, log_gpu_info, resolve_device
from training.rssm_dataset import RSSMSequenceDataset

_log = structlog.get_logger(__name__)


@dataclass
class CheckpointState:
    """Serialisable training checkpoint for resume support."""

    epoch: int
    best_loss: float
    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, torch.Tensor]
    scaler_state_dict: dict[str, torch.Tensor] | None
    rng_state: torch.Tensor


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: RSSM,
    optimizer: torch.optim.Optimizer,
    best_loss: float,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """Save a training checkpoint with full state."""
    state = CheckpointState(
        epoch=epoch,
        best_loss=best_loss,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scaler_state_dict=scaler.state_dict() if scaler else None,
        rng_state=torch.get_rng_state(),
    )
    checkpoint = {
        "epoch": state.epoch,
        "best_loss": state.best_loss,
        "model_state_dict": state.model_state_dict,
        "optimizer_state_dict": state.optimizer_state_dict,
        "scaler_state_dict": state.scaler_state_dict,
        "rng_state": state.rng_state,
    }
    torch.save(checkpoint, path)
    _log.info("checkpoint_saved", path=str(path), epoch=epoch)


def _load_checkpoint(
    path: Path,
    model: RSSM,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, float]:
    """Load a training checkpoint and return the starting epoch and best loss."""
    data = torch.load(path, map_location=device, weights_only=False)

    # Detect checkpoint format:
    # 1) Full training checkpoint dict with "model_state_dict" and friends
    # 2) Raw model state_dict (e.g. final weights only)
    if not isinstance(data, dict):
        raise TypeError(
            f"Unsupported checkpoint format at {path!s}: expected a dict, "
            f"got {type(data).__name__}"
        )

    if "model_state_dict" in data:
        # Full training checkpoint
        model.load_state_dict(data["model_state_dict"])

        optimizer_state = data.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        scaler_state = data.get("scaler_state_dict")
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)

        rng_state = data.get("rng_state")
        if rng_state is not None:
            torch.set_rng_state(rng_state)

        start_epoch = int(data.get("epoch", -1)) + 1
        best_loss = float(data.get("best_loss", float("inf")))
        _log.info(
            "checkpoint_loaded",
            path=str(path),
            resume_epoch=start_epoch,
            best_loss=best_loss,
            format="full",
        )
        return start_epoch, best_loss

    # Fallback: treat `data` as a raw model state_dict (no optimizer/epoch info)
    try:
        model.load_state_dict(data)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Unrecognized checkpoint format at {path!s}: expected either a full "
            f"training checkpoint dict with 'model_state_dict' or a raw model "
            f"state_dict compatible with RSSM."
        ) from exc

    # When only model weights are available, start from epoch 0 with unknown best_loss
    start_epoch = 0
    best_loss = float("inf")
    _log.info(
        "checkpoint_loaded",
        path=str(path),
        resume_epoch=start_epoch,
        best_loss=best_loss,
        format="state_dict_only",
    )
    return start_epoch, best_loss


def train_rssm(
    cfg: Settings,
    data_path: Path,
    device: torch.device | None = None,
    *,
    resume_from: Path | None = None,
) -> Path:
    """Train RSSM encoder + dynamics on synthetic data.

    Supports GPU acceleration with AMP and checkpoint resume.

    Args:
        cfg: Root settings with training and model configs.
        data_path: Path to ``sequences.pt`` file.
        device: Torch device (None = auto-detect).
        resume_from: Optional checkpoint path to resume from.

    Returns:
        Path to the final checkpoint.
    """
    device = device or resolve_device(cfg.training.gpu.device)
    log_gpu_info(device)

    tcfg = cfg.training
    mcfg = cfg.model

    # Build model
    rssm = RSSM(mcfg).to(device)
    optimizer = torch.optim.Adam(rssm.parameters(), lr=tcfg.learning_rate)

    # AMP setup (CUDA only)
    use_amp = tcfg.gpu.enable_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    _log.info("amp_status", enabled=use_amp, device=str(device))

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

    # Resume
    start_epoch = 1
    best_loss = float("inf")
    resume_path = resume_from or (Path(tcfg.resume_from) if tcfg.resume_from else None)
    if resume_path and resume_path.exists():
        start_epoch, best_loss = _load_checkpoint(resume_path, rssm, optimizer, device, scaler)

    mse_loss_fn = nn.MSELoss()

    for epoch in range(start_epoch, tcfg.epochs + 1):
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

            optimizer.zero_grad()

            # AMP context
            with torch.amp.autocast("cuda", enabled=use_amp):
                total_recon = torch.tensor(0.0, device=device)
                total_kl = torch.tensor(0.0, device=device)

                for t in range(seq_len):
                    obs_embed = rssm.encoder(
                        vision[:, t],
                        ultrasonic[:, t],
                        motor_state[:, t],
                        valid_mask[:, t],
                    )

                    prev_action = actions[:, max(0, t - 1)]
                    gru_input = torch.cat([z, prev_action], dim=-1)
                    h = rssm.gru(gru_input, h)

                    post_params = rssm.posterior(torch.cat([h, obs_embed], dim=-1))
                    z, post_mean, post_logvar = rssm._sample_gaussian(post_params)

                    prior_params = rssm.prior(h)
                    _, prior_mean, prior_logvar = rssm._sample_gaussian(prior_params)

                    obs_recon = rssm.decode(h, z)
                    total_recon = total_recon + mse_loss_fn(obs_recon, obs_embed)

                    kl = rssm._kl_divergence(
                        post_mean,
                        post_logvar,
                        prior_mean,
                        prior_logvar,
                    )
                    total_kl = total_kl + kl

                total_recon = total_recon / seq_len
                total_kl = total_kl / seq_len
                loss = total_recon + tcfg.kl_beta * total_kl

            # Backward with AMP scaling
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_recon += total_recon.item()
            epoch_kl += total_kl.item()
            n_batches += 1

        avg_recon = epoch_recon / max(n_batches, 1)
        avg_kl = epoch_kl / max(n_batches, 1)
        avg_loss = avg_recon + tcfg.kl_beta * avg_kl

        _log.info(
            "rssm_epoch",
            epoch=epoch,
            recon_loss=round(avg_recon, 6),
            kl_loss=round(avg_kl, 6),
        )

        # Track best
        if avg_loss < best_loss:
            best_loss = avg_loss

        # Checkpoint
        if epoch % tcfg.checkpoint_every_n == 0:
            ckpt_path = weights_dir / f"epoch_{epoch}.pt"
            _save_checkpoint(ckpt_path, epoch, rssm, optimizer, best_loss, scaler)

        # Memory check
        if device.type == "cuda":
            check_memory_budget(tcfg.gpu.memory_limit_gb, device)

    # Save final
    final_path = weights_dir / "final.pt"
    torch.save(rssm.state_dict(), final_path)
    _log.info("training_complete", final_checkpoint=str(final_path))
    return final_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="RSSM pretraining (GPU-accelerated)")
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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force torch device (cuda:0, cpu). Default: auto-detect",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    # Load config
    import yaml

    with open(args.config) as f:
        overrides = yaml.safe_load(f) or {}
    cfg = Settings(**overrides)

    data_path = Path(args.data) if args.data else Path(cfg.training.data_dir) / "sequences.pt"

    device = resolve_device(args.device) if args.device else None
    resume = Path(args.resume) if args.resume else None

    train_rssm(cfg, data_path, device=device, resume_from=resume)


if __name__ == "__main__":
    main()
