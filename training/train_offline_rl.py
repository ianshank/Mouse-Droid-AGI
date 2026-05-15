"""Offline RL training — CQL or IQL from LMDB experience data.

Reads stored experience transitions and trains an offline RL policy
without any online environment interaction. Supports algorithm selection
via configuration.

Usage:
    python -m training.train_offline_rl --config config/default.yaml
    python -m training.train_offline_rl --algorithm iql
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import ExperienceConfig, OfflineRLConfig, Settings
from mousedroid.experience.dataset import OfflineRLDataset
from mousedroid.learning.offline_rl import CQLTrainer, IQLTrainer, OfflineRLTrainer
from mousedroid.training.replay.mixer import MixerConfig, RealSimMixer
from training.gpu_utils import resolve_device

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import Tensor

_log = structlog.get_logger(__name__)


def _resolve_device(cfg: Settings) -> str:
    """Resolve torch device from config.

    Args:
        cfg: Root settings.

    Returns:
        Device string (e.g. 'cuda:0', 'cpu').
    """
    return str(
        resolve_device(
            cfg.training.gpu.device,
            require_cuda=cfg.training.gpu.require_cuda,
        )
    )


def _build_trainer(
    algorithm: str,
    state_dim: int,
    action_dim: int,
    offline_cfg: OfflineRLConfig,
    device_str: str,
) -> OfflineRLTrainer:
    """Build the appropriate offline RL trainer.

    Args:
        algorithm: Algorithm name ('cql' or 'iql').
        state_dim: State vector dimension.
        action_dim: Action vector dimension.
        offline_cfg: Offline RL configuration.
        device_str: Torch device string.

    Returns:
        Configured trainer instance.

    Raises:
        ValueError: If algorithm is unknown.
    """
    import torch

    device = torch.device(device_str)

    if algorithm == "cql":
        return CQLTrainer(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=offline_cfg.hidden_dim,
            gamma=offline_cfg.gamma,
            tau=offline_cfg.tau,
            lr=offline_cfg.learning_rate,
            cql_alpha=offline_cfg.cql_alpha,
            n_random_actions=offline_cfg.cql_n_random_actions,
            device=device,
            bc_lr=offline_cfg.bc_lr,
            bc_batch_size=offline_cfg.bc_batch_size,
        )
    if algorithm == "iql":
        return IQLTrainer(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=offline_cfg.hidden_dim,
            gamma=offline_cfg.gamma,
            tau=offline_cfg.tau,
            lr=offline_cfg.learning_rate,
            iql_tau=offline_cfg.iql_expectile,
            beta=offline_cfg.iql_beta,
            device=device,
            bc_lr=offline_cfg.bc_lr,
            bc_batch_size=offline_cfg.bc_batch_size,
        )
    msg = f"Unknown offline RL algorithm: {algorithm!r}. Use 'cql' or 'iql'."
    raise ValueError(msg)


def _resolve_real_replay_dataset(
    cfg: Settings,
    device_str: str,
) -> OfflineRLDataset | None:
    """Resolve and open a separate real-replay :class:`OfflineRLDataset`.

    Phase 2.1: returns a second dataset only when both
    ``cfg.training.replay.enabled`` is True and
    ``cfg.training.replay.source_path`` points to a distinct LMDB store from
    ``cfg.experience.path``. Otherwise returns ``None`` and the caller falls
    back to the single-source path (byte-identical to pre-Phase-2.1 behavior).

    Args:
        cfg: Root settings.
        device_str: Torch device string.

    Returns:
        Opened :class:`OfflineRLDataset` over the real-replay LMDB, or ``None``.
    """
    import torch

    replay_cfg = cfg.training.replay
    if not replay_cfg.enabled:
        return None
    if not replay_cfg.source_path:
        return None
    if replay_cfg.source_path == cfg.experience.path:
        # Same store on both sides degenerates to a single-source path; skip.
        return None

    real_experience_cfg = copy.deepcopy(cfg.experience)
    # ``ExperienceConfig`` is the schema consumed by ``OfflineRLDataset``; we
    # only need to redirect the path. The other fields (map_size_gb, flush
    # cadence) are reused as-is so real and sim datasets honor the same
    # operator-tuned envelope.
    real_experience_cfg = ExperienceConfig(
        **{**real_experience_cfg.model_dump(), "path": replay_cfg.source_path}
    )
    real_dataset = OfflineRLDataset(
        experience_cfg=real_experience_cfg,
        model_cfg=cfg.model,
        device=torch.device(device_str),
    )
    real_dataset.open()
    return real_dataset


def _build_mixed_batch_iterator(
    sim_dataset: OfflineRLDataset,
    real_dataset: OfflineRLDataset,
    offline_cfg: OfflineRLConfig,
    mixer_cfg: MixerConfig,
    epoch: int,
) -> Iterator[dict[str, Tensor]]:
    """Wrap sim + real batch iterators in a deterministic :class:`RealSimMixer`.

    A fresh mixer is built per epoch so the alpha ramp restarts deterministically
    and the seeded interleaving is reproducible.

    Args:
        sim_dataset: Primary ("sim") :class:`OfflineRLDataset`.
        real_dataset: Secondary ("real") :class:`OfflineRLDataset`.
        offline_cfg: Offline-RL configuration (drives batch shape).
        mixer_cfg: Mixer configuration (drives alpha + seed).
        epoch: Current epoch index (forwarded as dataset shuffle seed).

    Returns:
        Iterator over interleaved batch dicts.
    """
    sim_iter = sim_dataset.iterate_batches(
        batch_size=offline_cfg.batch_size,
        seed=epoch,
        terminal_gap_s=offline_cfg.terminal_gap_s,
    )
    real_iter = real_dataset.iterate_batches(
        batch_size=offline_cfg.batch_size,
        seed=epoch,
        terminal_gap_s=offline_cfg.terminal_gap_s,
    )
    return iter(RealSimMixer(sim_iter, real_iter, mixer_cfg))


def train_offline_rl(
    cfg: Settings,
    algorithm: str | None = None,
    output_dir: Path | None = None,
    resume_from: str | None = None,
) -> tuple[Path, dict[str, object]]:
    """Run offline RL training from LMDB experience data.

    Args:
        cfg: Root settings.
        algorithm: Override algorithm selection ('cql' or 'iql').
        output_dir: Override output directory.
        resume_from: Path to checkpoint to resume from.

    Returns:
        Tuple of ``(output_dir, training_stats)``.
    """
    import torch

    offline_cfg = cfg.offline_rl
    algo = algorithm or offline_cfg.algorithm
    output_dir = output_dir or Path(cfg.training.weights_dir) / "offline_rl" / algo
    output_dir.mkdir(parents=True, exist_ok=True)

    device_str = _resolve_device(cfg)
    _log.info(
        "offline_rl_starting",
        algorithm=algo,
        device=device_str,
        output_dir=str(output_dir),
    )

    # Load dataset
    dataset = OfflineRLDataset(
        experience_cfg=cfg.experience,
        model_cfg=cfg.model,
        device=torch.device(device_str),
    )
    dataset.open()

    # Phase 2.1: optionally open a second real-replay dataset for sim/real
    # interleaving via :class:`RealSimMixer`. Returns ``None`` when the user
    # has not opted in, preserving the single-source path.
    real_dataset: OfflineRLDataset | None = None
    mixer_cfg: MixerConfig | None = None
    if offline_cfg.use_replay_mixer:
        real_dataset = _resolve_real_replay_dataset(cfg, device_str)
        if real_dataset is None:
            _log.warning(
                "offline_rl_mixer_requested_but_unavailable",
                reason=(
                    "use_replay_mixer=True but cfg.training.replay is disabled "
                    "or source_path is unset/identical — falling back to single LMDB"
                ),
                replay_enabled=cfg.training.replay.enabled,
                source_path=cfg.training.replay.source_path,
            )
        else:
            mixer_cfg = MixerConfig.from_settings(cfg.training.replay_mixer)
            _log.info(
                "offline_rl_mixer_active",
                alpha_target=mixer_cfg.alpha_target,
                alpha_ramp_steps=mixer_cfg.alpha_ramp_steps,
                sim_path=cfg.experience.path,
                real_path=cfg.training.replay.source_path,
            )

    try:
        n_transitions = len(dataset)
        if n_transitions == 0:
            _log.warning("offline_rl_no_data", path=cfg.experience.path)
            return output_dir, {"error": "no_data", "n_transitions": 0}

        _log.info("offline_rl_dataset_loaded", n_transitions=n_transitions)

        # Build trainer
        trainer = _build_trainer(
            algorithm=algo,
            state_dim=dataset.state_dim,
            action_dim=cfg.model.action_dim,
            offline_cfg=offline_cfg,
            device_str=device_str,
        )

        if resume_from is not None:
            trainer.load(resume_from)
            _log.info("offline_rl_resumed", checkpoint=resume_from)

        # Training loop
        total_steps = 0
        epoch_losses: dict[str, list[float]] = {}

        # Phase 2.1 — TD3+BC-style auxiliary supervised loss against the
        # real-replay batch. ``bc_update`` is a no-op at weight 0.0, which is
        # the schema default, so legacy training paths remain byte-identical.
        bc_weight = offline_cfg.real_supervised_weight
        if bc_weight > 0.0:
            _log.info(
                "offline_rl_bc_active",
                weight=bc_weight,
                rationale="td3_plus_bc_aux_loss",
            )

        for epoch in range(1, offline_cfg.epochs + 1):
            batch_count = 0
            batch_iter: Iterator[dict[str, Tensor]] = (
                _build_mixed_batch_iterator(
                    sim_dataset=dataset,
                    real_dataset=real_dataset,
                    offline_cfg=offline_cfg,
                    mixer_cfg=mixer_cfg,
                    epoch=epoch,
                )
                if real_dataset is not None and mixer_cfg is not None
                else dataset.iterate_batches(
                    batch_size=offline_cfg.batch_size,
                    seed=epoch,
                    terminal_gap_s=offline_cfg.terminal_gap_s,
                )
            )
            for batch in batch_iter:
                losses = trainer.update_step(
                    states=batch["states"],
                    actions=batch["actions"],
                    rewards=batch["rewards"],
                    next_states=batch["next_states"],
                    dones=batch["dones"],
                )

                # Phase 2.1: add the auxiliary BC term on the same (s, a)
                # batch. ``bc_update`` returns ``{"bc_loss": 0.0}`` and
                # performs no optimizer step when ``bc_weight <= 0``.
                bc_losses = trainer.bc_update(
                    states=batch["states"],
                    actions=batch["actions"],
                    weight=bc_weight,
                )
                losses.update(bc_losses)

                for key, val in losses.items():
                    epoch_losses.setdefault(key, []).append(val)

                total_steps += 1
                batch_count += 1

            # Log epoch summary
            if epoch % offline_cfg.log_every_n_epochs == 0:
                epoch_summary = {
                    k: round(sum(v[-batch_count:]) / max(batch_count, 1), 6)
                    for k, v in epoch_losses.items()
                }
                _log.info(
                    "offline_rl_epoch",
                    epoch=epoch,
                    total_steps=total_steps,
                    **epoch_summary,
                )

            # Checkpoint
            if epoch % offline_cfg.checkpoint_every_n_epochs == 0:
                ckpt_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
                trainer.save(str(ckpt_path))

        # Save final model
        final_path = output_dir / "final.pt"
        trainer.save(str(final_path))

        # Compute final stats
        final_stats: dict[str, object] = {
            "algorithm": algo,
            "n_transitions": n_transitions,
            "total_steps": total_steps,
            "epochs": offline_cfg.epochs,
        }
        for key, vals in epoch_losses.items():
            final_stats[f"final_{key}"] = round(
                sum(vals[-batch_count:]) / max(batch_count, 1),
                6,
            )

        _log.info("offline_rl_complete", **final_stats)
        return output_dir, final_stats

    finally:
        dataset.close()
        if real_dataset is not None:
            real_dataset.close()


def main() -> None:
    """CLI entry point for offline RL training."""
    parser = argparse.ArgumentParser(description="Offline RL training (CQL/IQL)")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["cql", "iql"],
        default=None,
        help="Override algorithm selection",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path",
    )

    args = parser.parse_args()
    cfg = load_settings(args.config)

    output_dir = Path(args.output_dir) if args.output_dir else None
    train_offline_rl(
        cfg=cfg,
        algorithm=args.algorithm,
        output_dir=output_dir,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
