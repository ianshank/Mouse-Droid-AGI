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

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import Settings
from training.gpu_utils import log_gpu_info, resolve_device

_log = structlog.get_logger(__name__)


def _load_training_settings(config_path: str) -> Settings:
    """Load training settings while preserving legacy mock-hardware behavior.

    The historical training CLI always forced ``mock_hardware=True``. Keep that
    behavior for backwards compatibility, but load configuration via the shared
    overlay loader so default.yaml, overlays, and top-level env overrides all
    compose correctly.
    """
    settings = load_settings(Path(config_path))
    if not settings.mock_hardware:
        _log.warning(
            "training_forcing_mock_hardware",
            config_path=config_path,
        )
        settings = settings.model_copy(update={"mock_hardware": True})

    _log.info(
        "training_settings_loaded",
        config_path=config_path,
        batch_size=settings.training.batch_size,
        checkpoint_every_n=settings.training.checkpoint_every_n,
        mock_hardware=settings.mock_hardware,
    )
    return settings


def _require_existing_path(path: Path, *, description: str, phase: int) -> Path:
    """Validate prerequisite artifacts for partial pipeline runs."""
    if path.exists():
        return path
    msg = (
        f"Phase {phase} requires {description} at '{path}'. "
        "Run the prerequisite phase first or provide the artifact in the configured location."
    )
    raise FileNotFoundError(msg)


def run_phase_0_data_gen(cfg: Settings, *, seed: int | None = None) -> Path:
    """Phase 0: Generate synthetic training data.

    Args:
        cfg: Root settings.
        seed: Optional integer seed forwarded to the data generator. Only
            consumed when ``cfg.domain_randomization.enabled`` is true.

    Returns:
        Path to the data directory containing sequences.pt.
    """
    dr_cfg = cfg.domain_randomization
    _log.info(
        "phase_0_start",
        phase="data_generation",
        n_episodes=cfg.training.n_episodes,
        max_steps=cfg.training.sequence_length,
        output_dir=cfg.training.data_dir,
        domain_randomization_enabled=dr_cfg.enabled,
        seed=seed,
    )
    if dr_cfg.enabled:
        _log.info(
            "rssm_epoch_randomization",
            brightness=[dr_cfg.brightness.low, dr_cfg.brightness.high],
            contrast=[dr_cfg.contrast.low, dr_cfg.contrast.high],
            ultrasonic_noise_m=[
                dr_cfg.ultrasonic_noise_m.low,
                dr_cfg.ultrasonic_noise_m.high,
            ],
            ultrasonic_dropout_prob=[
                dr_cfg.ultrasonic_dropout_prob.low,
                dr_cfg.ultrasonic_dropout_prob.high,
            ],
            wheel_friction=[dr_cfg.wheel_friction.low, dr_cfg.wheel_friction.high],
            motor_gain=[dr_cfg.motor_gain.low, dr_cfg.motor_gain.high],
            feature_noise_std=[
                dr_cfg.feature_noise_std.low,
                dr_cfg.feature_noise_std.high,
            ],
        )
    from training.data_generator import SyntheticSequenceGenerator

    gen = SyntheticSequenceGenerator(cfg, seed=seed)
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
    _log.info(
        "phase_0b_start",
        phase="collect_annotations",
        n_episodes=cfg.training.annotation.n_episodes,
        max_steps=cfg.training.annotation.max_steps,
    )
    from training.collect_annotations import collect_annotations

    annotations_path = collect_annotations(
        cfg,
        n_episodes=cfg.training.annotation.n_episodes,
        max_steps=cfg.training.annotation.max_steps,
    )
    _log.info("phase_0b_complete", path=str(annotations_path))
    return annotations_path


def run_phase_1_rssm(cfg: Settings, data_dir: Path, *, resume_from: Path | None = None) -> Path:
    """Phase 1 (2.1): RSSM pretraining with GPU + AMP.

    Args:
        cfg: Root settings.
        data_dir: Data directory containing sequences.pt.
        resume_from: Optional checkpoint path to resume from.

    Returns:
        Path to final RSSM checkpoint.
    """
    _log.info("phase_1_start", phase="rssm_pretraining")
    from training.train_rssm import train_rssm

    data_path = _require_existing_path(
        data_dir / "sequences.pt",
        description="RSSM training data file 'sequences.pt'",
        phase=1,
    )
    checkpoint = train_rssm(cfg, data_path, resume_from=resume_from)
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

    output_dir = train_bdi(
        annotations_path,
        output_dir=Path(cfg.training.weights_dir) / "bdi",
        lr=cfg.training.learning_rate,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        gradient_scale=cfg.training.gradient_scale,
    )
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

    success = upload_weights(weights_dir, repo_id=repo_id)
    if success:
        _log.info("upload_complete", repo_id=repo_id)
    else:
        _log.error("upload_failed", repo_id=repo_id, weights_dir=weights_dir)
        raise RuntimeError(f"Failed to upload weights from '{weights_dir}' to '{repo_id}'")


def run_pipeline(
    cfg: Settings,
    *,
    phases: set[int] | None = None,
    upload: bool = False,
    resume_from: str | None = None,
) -> None:
    """Run the full pre-training pipeline.

    Args:
        cfg: Root settings.
        phases: Optional set of phase numbers to run (0,1,2,3,4).
            None runs all phases.
        upload: Whether to upload weights to HuggingFace after training.
        resume_from: Optional checkpoint path to resume RSSM training from.
    """
    all_phases = phases or {0, 1, 2, 3, 4}
    t0 = time.monotonic()

    device = resolve_device(
        cfg.training.gpu.device,
        require_cuda=cfg.training.gpu.require_cuda,
    )
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
        batch_size=cfg.training.batch_size,
        checkpoint_every_n=cfg.training.checkpoint_every_n,
        weights_dir=str(cfg.training.weights_dir),
    )

    # Phase 0: Data generation
    if 0 in all_phases:
        data_dir = run_phase_0_data_gen(cfg)
        annotations_path = run_phase_0b_annotations(cfg)

    # Phase 1: RSSM
    if 1 in all_phases:
        if 0 not in all_phases:
            data_dir = _require_existing_path(
                data_dir,
                description="training data directory",
                phase=1,
            )
        effective_resume_from = resume_from or cfg.training.resume_from
        resume_path = Path(effective_resume_from) if effective_resume_from else None
        rssm_checkpoint = run_phase_1_rssm(cfg, data_dir, resume_from=resume_path)

    # Phase 2: Warm-start
    if 2 in all_phases:
        if 0 not in all_phases:
            data_dir = _require_existing_path(
                data_dir,
                description="training data directory",
                phase=2,
            )
        if 1 not in all_phases:
            rssm_checkpoint = _require_existing_path(
                rssm_checkpoint,
                description="RSSM checkpoint",
                phase=2,
            )
        run_phase_2_warmstart(cfg, rssm_checkpoint, data_dir)

    # Phase 3: BDI
    if 3 in all_phases:
        if 0 not in all_phases:
            annotations_path = _require_existing_path(
                annotations_path,
                description="annotation dataset",
                phase=3,
            )
        run_phase_3_bdi(cfg, annotations_path)

    # Phase 4: Constitutional RL
    if 4 in all_phases:
        if 1 not in all_phases:
            rssm_checkpoint = _require_existing_path(
                rssm_checkpoint,
                description="RSSM checkpoint",
                phase=4,
            )
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to RSSM checkpoint to resume Phase 1 training from",
    )
    args = parser.parse_args()

    cfg = _load_training_settings(args.config)

    valid_phases = {0, 1, 2, 3, 4}
    phases: set[int] | None = None
    if args.phases:
        _parsed: set[int] = set()
        for raw_token in args.phases.split(","):
            token = raw_token.strip()
            if not token:
                continue
            try:
                phase_num = int(token)
            except ValueError:
                parser.error(
                    f"Invalid phase '{token}'. Phases must be integers in {sorted(valid_phases)}."
                )
            if phase_num not in valid_phases:
                parser.error(
                    f"Unknown phase '{phase_num}'. Supported phases are {sorted(valid_phases)}."
                )
            _parsed.add(phase_num)
        phases = _parsed

    run_pipeline(cfg, phases=phases, upload=args.upload, resume_from=args.resume)


if __name__ == "__main__":
    main()
