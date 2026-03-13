"""Unified pre-training pipeline — runs all phases end-to-end.

Usage:
    python -m training.run_pipeline --config config/mock_hardware.yaml
    python -m training.run_pipeline --config config/mock_hardware.yaml --upload
    python -m training.run_pipeline --phases 0,1,2  # Run specific phases only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import structlog

from mousedroid.config.schema import Settings
from training.gpu_utils import log_gpu_info, resolve_device

_log = structlog.get_logger(__name__)


def run_phase_0_data_gen(cfg: Settings) -> Path:
    """Phase 0: Generate synthetic training data.

    Returns:
        Path to the data directory containing sequences.pt.
    """
    _log.info("phase_0_start", phase="data_generation")
    from training.data_generator import SyntheticSequenceGenerator

    gen = SyntheticSequenceGenerator(cfg)
    output_dir = gen.generate_sequences(
        n_episodes=cfg.training.n_episodes,
        max_steps=cfg.training.sequence_length,
        output_dir=cfg.training.data_dir,
    )
    _log.info("phase_0_complete", data_dir=str(output_dir))
    return output_dir


def run_phase_0b_annotations(cfg: Settings) -> Path:
    """Phase 0b: Collect BDI intention annotations.

    Returns:
        Path to annotations file.
    """
    _log.info("phase_0b_start", phase="collect_annotations")
    from training.collect_annotations import collect_annotations

    annotations_path = collect_annotations(cfg, n_episodes=500, max_steps=50)
    _log.info("phase_0b_complete", path=str(annotations_path))
    return annotations_path


def run_phase_1_rssm(cfg: Settings, data_dir: Path) -> Path:
    """Phase 1 (2.1): RSSM pretraining with GPU + AMP.

    Returns:
        Path to final RSSM checkpoint.
    """
    _log.info("phase_1_start", phase="rssm_pretraining")
    from training.train_rssm import train_rssm

    data_path = data_dir / "sequences.pt"
    checkpoint = train_rssm(cfg, data_path)
    _log.info("phase_1_complete", checkpoint=str(checkpoint))
    return checkpoint


def run_phase_2_warmstart(cfg: Settings, rssm_checkpoint: Path, data_dir: Path) -> None:
    """Phase 2 (2.2): MCTS warm-start + UCB tuning.

    Returns:
        None (saves to weights dir).
    """
    _log.info("phase_2_start", phase="warmstart_policy")
    from training.warmstart_policy import run_warmstart

    run_warmstart(cfg, rssm_checkpoint, data_dir / "sequences.pt")
    _log.info("phase_2_complete")


def run_phase_3_bdi(cfg: Settings, annotations_path: Path) -> Path:
    """Phase 3 (2.3): BDI sub-network training (numpy SGD).

    Returns:
        Path to BDI weights directory.
    """
    _log.info("phase_3_start", phase="bdi_training")
    from training.train_bdi import train_bdi

    output_dir = train_bdi(annotations_path)
    _log.info("phase_3_complete", output_dir=str(output_dir))
    return output_dir


def run_phase_4_constitutional_rl(
    cfg: Settings,
    rssm_checkpoint: Path,
    policy_init_path: Path | None = None,
) -> Path:
    """Phase 4 (2.4): Constitutional RL PPO training.

    Returns:
        Path to output directory with policy and value weights.
    """
    _log.info("phase_4_start", phase="constitutional_rl")
    from training.train_constitutional_rl import train_constitutional_rl

    output_dir, _results = train_constitutional_rl(
        cfg, rssm_checkpoint, policy_init_path=policy_init_path
    )
    _log.info("phase_4_complete", output_dir=str(output_dir))
    return output_dir


def run_upload(weights_dir: str, repo_id: str = "ianshank/mousedroid-weights") -> None:
    """Upload trained weights to HuggingFace Hub."""
    _log.info("upload_start", repo_id=repo_id)
    from training.upload_weights import upload_weights

    upload_weights(weights_dir, repo_id=repo_id)
    _log.info("upload_complete", repo_id=repo_id)


def run_pipeline(
    cfg: Settings,
    *,
    phases: set[int] | None = None,
    upload: bool = False,
) -> None:
    """Run the full pre-training pipeline.

    Args:
        cfg: Root settings.
        phases: Optional set of phase numbers to run (0,1,2,3,4).
            None runs all phases.
        upload: Whether to upload weights to HuggingFace after training.
    """
    all_phases = phases or {0, 1, 2, 3, 4}
    t0 = time.monotonic()

    device = resolve_device(cfg.training.gpu.device)
    log_gpu_info(device)

    data_dir = Path(cfg.training.data_dir)
    annotations_path = data_dir / "bdi_annotations.npz"
    rssm_checkpoint = Path(cfg.training.weights_dir) / "rssm" / "final.pt"
    policy_init_path = Path(cfg.training.weights_dir) / "mcts" / "policy_init.npz"

    _log.info(
        "pipeline_start",
        phases=sorted(all_phases),
        device=str(device),
        data_dir=str(data_dir),
    )

    # Phase 0: Data generation
    if 0 in all_phases:
        data_dir = run_phase_0_data_gen(cfg)
        annotations_path = run_phase_0b_annotations(cfg)

    # Phase 1: RSSM
    if 1 in all_phases:
        rssm_checkpoint = run_phase_1_rssm(cfg, data_dir)

    # Phase 2: Warm-start
    if 2 in all_phases:
        run_phase_2_warmstart(cfg, rssm_checkpoint, data_dir)

    # Phase 3: BDI
    if 3 in all_phases:
        run_phase_3_bdi(cfg, annotations_path)

    # Phase 4: Constitutional RL
    if 4 in all_phases:
        pi_path = policy_init_path if policy_init_path.exists() else None
        run_phase_4_constitutional_rl(cfg, rssm_checkpoint, pi_path)

    elapsed = time.monotonic() - t0
    _log.info("pipeline_complete", total_time_s=round(elapsed, 1))

    if upload:
        run_upload(cfg.training.weights_dir)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="MouseDroid full pre-training pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/mock_hardware.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--phases",
        type=str,
        default=None,
        help="Comma-separated phase numbers to run (e.g. '0,1,2'). Default: all",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload weights to HuggingFace after training",
    )
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        overrides = yaml.safe_load(f) or {}
    cfg = Settings(**{**overrides, "mock_hardware": True})

    phases = {int(p) for p in args.phases.split(",")} if args.phases else None

    run_pipeline(cfg, phases=phases, upload=args.upload)


if __name__ == "__main__":
    main()
