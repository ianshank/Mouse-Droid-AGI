r"""Phase 2.2 — Dual-Stream CfC/GRU RSSM pretraining on synthetic observation sequences.

Extends the baseline RSSM training pattern (train_rssm.py) with:
  - Dual optimizers: separate Adam for GRU+heads and CfC cell.
  - Separate gradient clipping thresholds per stream.
  - Linear CfC loss weight warmup from ``cfc_loss_weight_initial`` to
    ``cfc_loss_weight_final`` over ``cfc_loss_warmup_steps`` global steps.
  - Periodic fallback monitoring to log CfC contribution quality.
  - Full AMP support and checkpoint resume.

Usage:
    python -m training.train_dual_stream_rssm --config config/mock_hardware.yaml
    python -m training.train_dual_stream_rssm \
        --config config/mock_hardware.yaml \
        --device cuda \
        --resume weights/dual_stream_rssm/epoch_50.pt
    python -m training.train_dual_stream_rssm \
        --config config/mock_hardware.yaml \
        --validate-only
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
from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM
from training.gpu_utils import check_memory_budget, log_gpu_info, resolve_device
from training.rssm_dataset import RSSMSequenceDataset

_log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint data structure
# ---------------------------------------------------------------------------


@dataclass
class DualStreamCheckpointState:
    """Serialisable training checkpoint for dual-stream resume support.

    Extends the base RSSM checkpoint with a second optimizer state and the
    warmup step counter needed to resume the CfC loss weight ramp correctly.
    """

    epoch: int
    best_loss: float
    warmup_step: int
    combined_dim: int
    model_state_dict: dict[str, torch.Tensor]
    gru_optimizer_state_dict: dict[str, torch.Tensor]
    cfc_optimizer_state_dict: dict[str, torch.Tensor]
    scaler_state_dict: dict[str, torch.Tensor] | None
    rng_state: torch.Tensor


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: DualStreamRSSM,
    gru_optimizer: torch.optim.Optimizer,
    cfc_optimizer: torch.optim.Optimizer,
    best_loss: float,
    warmup_step: int,
    combined_dim: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> None:
    """Save a dual-stream training checkpoint with full state.

    Args:
        path: Destination file path.
        epoch: Current epoch number (0-indexed internally, 1-indexed in filename).
        model: The DualStreamRSSM module being trained.
        gru_optimizer: Optimizer managing GRU + shared-head parameters.
        cfc_optimizer: Optimizer managing CfC cell parameters.
        best_loss: Best validation loss seen so far.
        warmup_step: Global step counter used for CfC loss warmup.
        combined_dim: Total hidden dim (gru_hidden + cfc_hidden) for shape validation.
        scaler: Optional AMP GradScaler.
    """
    state = DualStreamCheckpointState(
        epoch=epoch,
        best_loss=best_loss,
        warmup_step=warmup_step,
        combined_dim=combined_dim,
        model_state_dict=model.state_dict(),
        gru_optimizer_state_dict=gru_optimizer.state_dict(),
        cfc_optimizer_state_dict=cfc_optimizer.state_dict(),
        scaler_state_dict=scaler.state_dict() if scaler else None,
        rng_state=torch.get_rng_state(),
    )
    checkpoint: dict[str, object] = {
        "epoch": state.epoch,
        "best_loss": state.best_loss,
        "warmup_step": state.warmup_step,
        "combined_dim": state.combined_dim,
        "model_state_dict": state.model_state_dict,
        "gru_optimizer_state_dict": state.gru_optimizer_state_dict,
        "cfc_optimizer_state_dict": state.cfc_optimizer_state_dict,
        "scaler_state_dict": state.scaler_state_dict,
        "rng_state": state.rng_state,
    }
    torch.save(checkpoint, path)
    _log.info(
        "dual_stream_checkpoint_saved",
        path=str(path),
        epoch=epoch,
        warmup_step=warmup_step,
    )


def _load_checkpoint(
    path: Path,
    model: DualStreamRSSM,
    gru_optimizer: torch.optim.Optimizer,
    cfc_optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> tuple[int, float, int]:
    """Load a dual-stream training checkpoint.

    Supports both full dual-stream checkpoints and legacy single-optimizer
    checkpoints (e.g. checkpoints from ``train_rssm.py`` used for warm-start).

    Args:
        path: Checkpoint file path.
        model: The DualStreamRSSM module to restore weights into.
        gru_optimizer: GRU optimizer to restore state into.
        cfc_optimizer: CfC optimizer to restore state into.
        device: Target device for tensor mapping.
        scaler: Optional AMP GradScaler to restore.

    Returns:
        Tuple of ``(start_epoch, best_loss, warmup_step)``.

    Raises:
        TypeError: If the checkpoint file is not a dict.
        ValueError: If the checkpoint format cannot be recognised.
    """
    data = torch.load(path, map_location=device, weights_only=False)

    if not isinstance(data, dict):
        raise TypeError(
            f"Unsupported checkpoint format at {path!s}: expected a dict, got {type(data).__name__}"
        )

    if "model_state_dict" not in data:
        # Raw state_dict — treat as model-weights-only warm-start
        try:
            model.load_state_dict(data)
        except Exception as exc:
            raise ValueError(
                f"Unrecognised checkpoint format at {path!s}: expected a full "
                "training checkpoint dict with 'model_state_dict' or a raw "
                "DualStreamRSSM state_dict."
            ) from exc
        _log.info(
            "dual_stream_checkpoint_loaded",
            path=str(path),
            resume_epoch=0,
            best_loss=float("inf"),
            warmup_step=0,
            format="state_dict_only",
        )
        return 0, float("inf"), 0

    # Full training checkpoint
    model.load_state_dict(data["model_state_dict"])

    # Legacy single-optimizer checkpoints (from train_rssm.py) store
    # "optimizer_state_dict" covering GRU + shared-head params.  Load it
    # into gru_optimizer since that's the matching parameter group.
    legacy_opt_state = data.get("optimizer_state_dict")

    gru_opt_state = data.get("gru_optimizer_state_dict") or legacy_opt_state
    if gru_opt_state is not None:
        try:
            gru_optimizer.load_state_dict(gru_opt_state)
        except Exception:
            _log.warning(
                "gru_optimizer_state_load_failed_using_fresh",
                path=str(path),
                legacy=legacy_opt_state is not None and gru_opt_state is legacy_opt_state,
            )

    cfc_opt_state = data.get("cfc_optimizer_state_dict")
    if cfc_opt_state is not None:
        try:
            cfc_optimizer.load_state_dict(cfc_opt_state)
        except Exception:
            _log.warning(
                "cfc_optimizer_state_load_failed_using_fresh",
                path=str(path),
            )

    scaler_state = data.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    rng_state = data.get("rng_state")
    if rng_state is not None:
        torch.set_rng_state(rng_state)

    start_epoch = int(data.get("epoch", -1)) + 1
    best_loss = float(data.get("best_loss", float("inf")))
    warmup_step = int(data.get("warmup_step", 0))

    _log.info(
        "dual_stream_checkpoint_loaded",
        path=str(path),
        resume_epoch=start_epoch,
        best_loss=best_loss,
        warmup_step=warmup_step,
        format="full",
    )
    return start_epoch, best_loss, warmup_step


# ---------------------------------------------------------------------------
# CfC loss weight schedule
# ---------------------------------------------------------------------------


def _cfc_loss_weight(
    step: int,
    *,
    initial: float,
    final: float,
    warmup_steps: int,
) -> float:
    """Compute CfC loss weight via linear warmup schedule.

    Linearly ramps the CfC loss contribution from ``initial`` to ``final``
    over the first ``warmup_steps`` global steps, then holds at ``final``.

    Args:
        step: Current global training step.
        initial: Weight at step 0.
        final: Weight after warmup.
        warmup_steps: Number of steps to ramp over.

    Returns:
        Scalar loss weight in ``[initial, final]``.
    """
    if warmup_steps <= 0 or step >= warmup_steps:
        return final
    fraction = step / warmup_steps
    return initial + fraction * (final - initial)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_dual_stream_rssm(
    cfg: Settings,
    data_path: Path,
    device: torch.device | None = None,
    *,
    resume_from: Path | None = None,
    validate_only: bool = False,
) -> Path:
    """Train Dual-Stream CfC/GRU RSSM on synthetic sequences.

    Trains a :class:`~mousedroid.world_model.dual_stream_rssm.DualStreamRSSM`
    with dual optimizers, CfC loss warmup, and periodic fallback monitoring.
    Supports AMP and checkpoint resume.

    Args:
        cfg: Root settings with ``training``, ``dual_stream_training``, and
            ``model`` configs.
        data_path: Path to ``sequences.pt`` file produced by
            ``SyntheticSequenceGenerator``.
        device: Torch device (None = auto-detect via ``resolve_device``).
        resume_from: Optional checkpoint path to resume from.
        validate_only: When True, runs a single epoch then exits without
            saving a final checkpoint.

    Returns:
        Path to the final checkpoint (``final.pt``) — or the last periodic
        checkpoint when ``validate_only=True``.
    """
    device = device or resolve_device(cfg.training.gpu.device)
    log_gpu_info(device)

    tcfg = cfg.training
    mcfg = cfg.model
    dscfg = cfg.dual_stream_training

    _log.info(
        "dual_stream_training_start",
        gru_hidden_dim=mcfg.hidden_dim,
        cfc_hidden_dim=mcfg.cfc_hidden_dim,
        combined_dim=mcfg.hidden_dim + mcfg.cfc_hidden_dim,
        gru_lr=dscfg.gru_lr,
        cfc_lr=dscfg.cfc_lr,
        cfc_loss_weight_initial=dscfg.cfc_loss_weight_initial,
        cfc_loss_weight_final=dscfg.cfc_loss_weight_final,
        cfc_loss_warmup_steps=dscfg.cfc_loss_warmup_steps,
        validate_only=validate_only,
    )

    # ------------------------------------------------------------------
    # Build model and dual optimizers
    # ------------------------------------------------------------------
    model = DualStreamRSSM(mcfg).to(device)
    combined_dim = mcfg.hidden_dim + mcfg.cfc_hidden_dim

    gru_optimizer = torch.optim.Adam(
        list(model.gru_parameters()),
        lr=dscfg.gru_lr,
    )
    cfc_optimizer = torch.optim.Adam(
        list(model.cfc_parameters()),
        lr=dscfg.cfc_lr,
    )

    # ------------------------------------------------------------------
    # AMP setup (CUDA only)
    # ------------------------------------------------------------------
    use_amp = tcfg.gpu.enable_amp and device.type == "cuda"
    scaler: torch.cuda.amp.GradScaler | None = torch.cuda.amp.GradScaler() if use_amp else None
    _log.info("amp_status", enabled=use_amp, device=str(device))

    # ------------------------------------------------------------------
    # Dataset and DataLoader
    # ------------------------------------------------------------------
    dataset = RSSMSequenceDataset(data_path, seq_len=tcfg.sequence_length)
    loader = DataLoader(
        dataset,
        batch_size=tcfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Checkpoint directory
    # ------------------------------------------------------------------
    weights_dir = Path(tcfg.weights_dir) / "dual_stream_rssm"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 1
    best_loss = float("inf")
    warmup_step = 0
    resume_path = resume_from or (Path(tcfg.resume_from) if tcfg.resume_from else None)
    if resume_path and resume_path.exists():
        start_epoch, best_loss, warmup_step = _load_checkpoint(
            resume_path,
            model,
            gru_optimizer,
            cfc_optimizer,
            device,
            scaler,
        )

    mse_loss_fn = nn.MSELoss()
    max_epoch = start_epoch + 1 if validate_only else tcfg.epochs + 1

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, max_epoch):
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_gru_recon = 0.0
        epoch_cfc_recon = 0.0
        n_batches = 0

        for vision, ultrasonic, motor_state, valid_mask, actions in loader:
            vision = vision.to(device)
            ultrasonic = ultrasonic.to(device)
            motor_state = motor_state.to(device)
            valid_mask = valid_mask.to(device)
            actions = actions.to(device)

            batch_size: int = vision.shape[0]
            seq_len: int = vision.shape[1]

            # Init dual hidden states
            h_gru = torch.zeros(batch_size, mcfg.hidden_dim, device=device)
            h_cfc = torch.zeros(batch_size, mcfg.cfc_hidden_dim, device=device)
            z = torch.zeros(batch_size, mcfg.latent_dim, device=device)

            # Pre-allocate reusable zero buffers for stream-specific decoding
            zero_cfc = torch.zeros(batch_size, mcfg.cfc_hidden_dim, device=device)
            zero_gru = torch.zeros(batch_size, mcfg.hidden_dim, device=device)
            zero_action = torch.zeros(batch_size, mcfg.action_dim, device=device)

            # Current CfC loss weight for this step
            cfc_w = _cfc_loss_weight(
                warmup_step,
                initial=dscfg.cfc_loss_weight_initial,
                final=dscfg.cfc_loss_weight_final,
                warmup_steps=dscfg.cfc_loss_warmup_steps,
            )

            gru_optimizer.zero_grad()
            cfc_optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                total_recon = torch.tensor(0.0, device=device)
                total_kl = torch.tensor(0.0, device=device)
                # Track per-stream reconstruction for fallback monitoring
                total_gru_recon = torch.tensor(0.0, device=device)
                total_cfc_recon = torch.tensor(0.0, device=device)

                for t in range(seq_len):
                    obs_embed = model.encoder(
                        vision[:, t],
                        ultrasonic[:, t],
                        motor_state[:, t],
                        valid_mask[:, t],
                    )

                    prev_action = actions[:, t - 1] if t > 0 else zero_action
                    recurrent_input = torch.cat([z, prev_action], dim=-1)

                    # Dual-stream recurrent step
                    h_gru = model.gru(recurrent_input, h_gru)
                    h_cfc = model.cfc(recurrent_input, h_cfc)

                    # Fuse streams into combined hidden state
                    h = model.fusion.fuse(h_gru, h_cfc)

                    # Posterior: use combined state
                    post_params = model.posterior(torch.cat([h, obs_embed], dim=-1))
                    z, post_mean, post_logvar = model._sample_gaussian(post_params)

                    # Prior: use combined state
                    prior_params = model.prior(h)
                    _, prior_mean, prior_logvar = model._sample_gaussian(prior_params)

                    # Decode: combined state
                    obs_recon = model.decode(h, z)
                    step_recon = mse_loss_fn(obs_recon, obs_embed)
                    total_recon = total_recon + step_recon

                    # Stream-specific reconstructions for fallback monitor
                    # GRU-only: decode using just the GRU portion of hidden state
                    gru_h_padded = torch.cat([h_gru, zero_cfc], dim=-1)
                    gru_only_recon = model.decode(gru_h_padded, z)
                    total_gru_recon = total_gru_recon + mse_loss_fn(gru_only_recon, obs_embed)

                    # CfC-only: decode using just the CfC portion
                    cfc_h_padded = torch.cat([zero_gru, h_cfc], dim=-1)
                    cfc_only_recon = model.decode(cfc_h_padded, z)
                    total_cfc_recon = total_cfc_recon + mse_loss_fn(cfc_only_recon, obs_embed)

                    kl = model._kl_divergence(
                        post_mean,
                        post_logvar,
                        prior_mean,
                        prior_logvar,
                    )
                    total_kl = total_kl + kl

                total_recon = total_recon / seq_len
                total_kl = total_kl / seq_len
                total_gru_recon = total_gru_recon / seq_len
                total_cfc_recon = total_cfc_recon / seq_len

                # Combined loss — CfC stream weighted by warmup schedule
                loss = total_recon + cfc_w * total_cfc_recon + tcfg.kl_beta * total_kl

            # ----------------------------------------------------------
            # Backward + dual gradient clipping + optimizer steps
            # ----------------------------------------------------------
            if scaler:
                scaler.scale(loss).backward()

                # Unscale before clipping so clip norms are in true grad units
                scaler.unscale_(gru_optimizer)
                scaler.unscale_(cfc_optimizer)

                torch.nn.utils.clip_grad_norm_(
                    list(model.gru_parameters()),
                    max_norm=dscfg.gru_grad_clip,
                )
                torch.nn.utils.clip_grad_norm_(
                    list(model.cfc_parameters()),
                    max_norm=dscfg.cfc_grad_clip,
                )

                scaler.step(gru_optimizer)
                scaler.step(cfc_optimizer)
                scaler.update()
            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(model.gru_parameters()),
                    max_norm=dscfg.gru_grad_clip,
                )
                torch.nn.utils.clip_grad_norm_(
                    list(model.cfc_parameters()),
                    max_norm=dscfg.cfc_grad_clip,
                )

                gru_optimizer.step()
                cfc_optimizer.step()

            epoch_recon += total_recon.item()
            epoch_kl += total_kl.item()
            epoch_gru_recon += total_gru_recon.item()
            epoch_cfc_recon += total_cfc_recon.item()
            n_batches += 1
            warmup_step += 1

            # ----------------------------------------------------------
            # Fallback monitoring
            # ----------------------------------------------------------
            if warmup_step % dscfg.fallback_check_interval == 0:
                _log_cfc_contribution(
                    warmup_step=warmup_step,
                    gru_recon=total_gru_recon.item(),
                    cfc_recon=total_cfc_recon.item(),
                    combined_recon=total_recon.item(),
                    degradation_threshold=dscfg.fallback_degradation_threshold,
                )

        # ------------------------------------------------------------------
        # Epoch-level logging
        # ------------------------------------------------------------------
        avg_recon = epoch_recon / max(n_batches, 1)
        avg_kl = epoch_kl / max(n_batches, 1)
        avg_gru_recon = epoch_gru_recon / max(n_batches, 1)
        avg_cfc_recon = epoch_cfc_recon / max(n_batches, 1)
        current_cfc_w = _cfc_loss_weight(
            warmup_step,
            initial=dscfg.cfc_loss_weight_initial,
            final=dscfg.cfc_loss_weight_final,
            warmup_steps=dscfg.cfc_loss_warmup_steps,
        )
        avg_loss = avg_recon + current_cfc_w * avg_cfc_recon + tcfg.kl_beta * avg_kl

        _log.info(
            "dual_stream_epoch",
            epoch=epoch,
            recon_loss=round(avg_recon, 6),
            kl_loss=round(avg_kl, 6),
            gru_only_recon=round(avg_gru_recon, 6),
            cfc_only_recon=round(avg_cfc_recon, 6),
            cfc_loss_weight=round(current_cfc_w, 6),
            warmup_step=warmup_step,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss

        # ------------------------------------------------------------------
        # Periodic checkpoint
        # ------------------------------------------------------------------
        if epoch % tcfg.checkpoint_every_n == 0:
            ckpt_path = weights_dir / f"epoch_{epoch}.pt"
            _save_checkpoint(
                ckpt_path,
                epoch=epoch,
                model=model,
                gru_optimizer=gru_optimizer,
                cfc_optimizer=cfc_optimizer,
                best_loss=best_loss,
                warmup_step=warmup_step,
                combined_dim=combined_dim,
                scaler=scaler,
            )

        # ------------------------------------------------------------------
        # Memory budget check
        # ------------------------------------------------------------------
        if device.type == "cuda":
            check_memory_budget(tcfg.gpu.memory_limit_gb, device)

    # ------------------------------------------------------------------
    # Final checkpoint (skip in validate_only mode)
    # ------------------------------------------------------------------
    if validate_only:
        # Always save a checkpoint so validate-only has a guaranteed artefact
        validate_path = weights_dir / "validate_final.pt"
        _save_checkpoint(
            validate_path,
            epoch=max_epoch - 1,
            model=model,
            gru_optimizer=gru_optimizer,
            cfc_optimizer=cfc_optimizer,
            best_loss=best_loss,
            warmup_step=warmup_step,
            combined_dim=combined_dim,
            scaler=scaler,
        )
        _log.info(
            "validate_only_complete",
            checkpoint=str(validate_path),
            epochs_run=max_epoch - start_epoch,
        )
        return validate_path

    final_path = weights_dir / "final.pt"
    torch.save(model.state_dict(), final_path)
    _log.info("dual_stream_training_complete", final_checkpoint=str(final_path))
    return final_path


# ---------------------------------------------------------------------------
# Fallback monitoring helper
# ---------------------------------------------------------------------------


def _log_cfc_contribution(
    *,
    warmup_step: int,
    gru_recon: float,
    cfc_recon: float,
    combined_recon: float,
    degradation_threshold: float,
) -> None:
    """Log CfC stream contribution quality and emit a warning when degraded.

    CfC contribution is measured as the improvement of the combined
    reconstruction over the GRU-only reconstruction, normalised by the
    GRU-only error.  A negative contribution means the CfC stream is
    hurting reconstruction quality.

    Args:
        warmup_step: Current global step for log context.
        gru_recon: GRU-only reconstruction MSE for the last batch.
        cfc_recon: CfC-only reconstruction MSE for the last batch.
        combined_recon: Full dual-stream reconstruction MSE for the last batch.
        degradation_threshold: Maximum allowable quality drop (normalised).
    """
    # Positive = improvement; negative = regression
    cfc_improvement = (gru_recon - combined_recon) / gru_recon if gru_recon > 0 else 0.0

    _log.info(
        "cfc_fallback_check",
        warmup_step=warmup_step,
        gru_only_recon=round(gru_recon, 6),
        cfc_only_recon=round(cfc_recon, 6),
        combined_recon=round(combined_recon, 6),
        cfc_improvement_normalised=round(cfc_improvement, 6),
    )

    if cfc_improvement < -degradation_threshold:
        _log.warning(
            "cfc_stream_degradation_detected",
            warmup_step=warmup_step,
            cfc_improvement_normalised=round(cfc_improvement, 6),
            degradation_threshold=degradation_threshold,
            action="consider_reducing_cfc_lr_or_increasing_warmup_steps",
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for dual-stream RSSM training."""
    parser = argparse.ArgumentParser(
        description="Dual-Stream CfC/GRU RSSM pretraining (GPU-accelerated)"
    )
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
    parser.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help="Run a single epoch to validate the training pipeline, then exit",
    )
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        overrides = yaml.safe_load(f) or {}
    cfg = Settings(**overrides)

    data_path = Path(args.data) if args.data else Path(cfg.training.data_dir) / "sequences.pt"
    device = resolve_device(args.device) if args.device else None
    resume = Path(args.resume) if args.resume else None

    train_dual_stream_rssm(
        cfg,
        data_path,
        device=device,
        resume_from=resume,
        validate_only=args.validate_only,
    )


if __name__ == "__main__":
    main()
