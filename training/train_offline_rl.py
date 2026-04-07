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
from pathlib import Path

import structlog

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import OfflineRLConfig, Settings
from mousedroid.experience.dataset import OfflineRLDataset
from mousedroid.learning.offline_rl import CQLTrainer, IQLTrainer, OfflineRLTrainer

_log = structlog.get_logger(__name__)


def _resolve_device(cfg: Settings) -> str:
    """Resolve torch device from config.

    Args:
        cfg: Root settings.

    Returns:
        Device string (e.g. 'cuda:0', 'cpu').
    """
    import torch

    explicit = cfg.training.gpu.device
    if explicit is not None:
        return explicit
    return "cuda:0" if torch.cuda.is_available() else "cpu"


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
        )
    msg = f"Unknown offline RL algorithm: {algorithm!r}. Use 'cql' or 'iql'."
    raise ValueError(msg)


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

        for epoch in range(1, offline_cfg.epochs + 1):
            batch_count = 0
            for batch in dataset.iterate_batches(
                batch_size=offline_cfg.batch_size,
                seed=epoch,
                terminal_gap_s=offline_cfg.terminal_gap_s,
            ):
                losses = trainer.update_step(
                    states=batch["states"],
                    actions=batch["actions"],
                    rewards=batch["rewards"],
                    next_states=batch["next_states"],
                    dones=batch["dones"],
                )

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
