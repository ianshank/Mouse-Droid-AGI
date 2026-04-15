"""GPU pre-training pipeline orchestrator (ADR-005).

Sequences the 4 training phases (RSSM -> Warm-start -> BDI ->
Constitutional RL) with thermal monitoring, batch tuning, and
checkpoint-based resume support.

CLI usage::

    python -m mousedroid.training.pipeline_orchestrator \
        --config config/local_training.yaml

    python -m mousedroid.training.pipeline_orchestrator \
        --config config/local_training.yaml --resume
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import structlog

from mousedroid.config.schema import Settings, TrainingPipelineConfig
from mousedroid.training.batch_tuner import VRAMBatchTuner
from mousedroid.training.gpu_monitor import JetsonGPUMonitor

logger = structlog.get_logger(__name__)


class PipelineOrchestrator:
    """Orchestrates the multi-phase GPU pre-training pipeline.

    Reads phase order and batch sizes from ``TrainingPipelineConfig``,
    checks GPU thermals before each phase, tunes batch sizes based on
    available VRAM, and validates checkpoints between phases.

    Args:
        settings: Root application settings.
        pipeline_config: Pipeline-specific configuration.
        gpu_monitor: GPU thermal/VRAM monitor instance.
        batch_tuner: Dynamic batch size tuner instance.
    """

    def __init__(
        self,
        settings: Settings,
        pipeline_config: TrainingPipelineConfig,
        gpu_monitor: JetsonGPUMonitor | Any,
        batch_tuner: VRAMBatchTuner | Any,
    ) -> None:
        self._settings = settings
        self._config = pipeline_config
        self._gpu_monitor = gpu_monitor
        self._batch_tuner = batch_tuner
        self._checkpoint_dir = Path(pipeline_config.checkpoint_dir)

    async def run(self) -> None:
        """Execute all configured training phases in order.

        Supports resume: if ``resume_from_phase`` is set, phases before
        that phase are skipped.

        Raises:
            RuntimeError: If a phase fails or a checkpoint is missing
                between phases.
        """
        phases = self._config.phases
        resume_phase = self._config.resume_from_phase

        # Determine start index for resume.
        start_idx = 0
        if resume_phase is not None:
            if resume_phase not in phases:
                msg = f"resume_from_phase '{resume_phase}' not in configured phases {phases}"
                raise RuntimeError(msg)
            start_idx = phases.index(resume_phase)
            logger.info(
                "pipeline_resuming",
                resume_phase=resume_phase,
                skipped=phases[:start_idx],
            )

        logger.info(
            "pipeline_started",
            total_phases=len(phases),
            start_index=start_idx,
            phases=phases[start_idx:],
        )

        for idx in range(start_idx, len(phases)):
            phase = phases[idx]
            phase_log = logger.bind(phase=phase, phase_index=idx)

            # Thermal check before each phase.
            await self._wait_for_thermal_clearance(phase_log)

            # Tune batch size.
            base_batch = self._config.batch_sizes.get(phase, self._settings.training.batch_size)
            tuned_batch = self._batch_tuner.tune_batch_size(phase, base_batch)

            # Validate prior checkpoint (skip for first executed phase).
            if idx > start_idx:
                prev_phase = phases[idx - 1]
                if not self._checkpoint_exists(prev_phase):
                    msg = (
                        f"Missing checkpoint for phase '{prev_phase}' — cannot proceed to '{phase}'"
                    )
                    raise RuntimeError(msg)

            phase_log.info(
                "phase_starting",
                batch_size=tuned_batch,
                amp_enabled=self._config.amp_enabled,
            )

            try:
                await self._run_phase(phase, tuned_batch)
            except Exception:
                phase_log.exception("phase_failed")
                raise

            phase_log.info("phase_completed")

        logger.info("pipeline_completed", phases_run=phases[start_idx:])

    async def _run_phase(self, phase: str, batch_size: int) -> None:
        """Execute a single training phase.

        Each phase creates its own checkpoint file on completion.

        Args:
            phase: Phase name (e.g. "rssm").
            batch_size: Tuned batch size for this phase.
        """
        # Create checkpoint directory if needed.
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Dispatch to phase-specific training logic.
        phase_fn = self._get_phase_runner(phase)
        await phase_fn(self._settings, batch_size, self._checkpoint_dir)

        # Write checkpoint marker.
        marker_suffix = self._config.checkpoint_marker_suffix
        checkpoint_path = self._checkpoint_dir / f"{phase}{marker_suffix}"
        checkpoint_path.write_text(f"phase={phase}\n")
        logger.info("checkpoint_written", path=str(checkpoint_path))

    @staticmethod
    def _get_settings_with_batch(cfg: Settings, batch_size: int) -> Settings:
        """Return an immutable copy of settings with training.batch_size overridden.

        Args:
            cfg: Root application settings.
            batch_size: New batch size to inject.

        Returns:
            Deep copy of settings with updated batch_size.
        """
        return cfg.model_copy(
            update={"training": cfg.training.model_copy(update={"batch_size": batch_size})}
        )

    @staticmethod
    def _get_rssm_checkpoint_path(cfg: Settings) -> Path:
        """Resolve the RSSM final checkpoint path from config.

        Args:
            cfg: Root application settings.

        Returns:
            Path to the expected RSSM checkpoint file.
        """
        tcfg = cfg.training
        return Path(tcfg.weights_dir) / tcfg.rssm_subdir / tcfg.rssm_checkpoint_filename

    def _get_phase_runner(self, phase: str) -> Any:
        """Resolve phase name to its async runner function.

        Args:
            phase: Phase name.

        Returns:
            Async callable accepting (cfg, batch_size, checkpoint_dir).
        """
        runners: dict[str, Any] = {
            "rssm": self._train_rssm,
            "warmstart": self._train_warmstart,
            "bdi": self._train_bdi,
            "constitutional_rl": self._train_constitutional_rl,
        }
        if phase not in runners:
            msg = f"Unknown training phase: '{phase}'"
            raise ValueError(msg)
        return runners[phase]

    async def _train_rssm(self, cfg: Settings, batch_size: int, checkpoint_dir: Path) -> None:
        """Run RSSM pre-training phase.

        Delegates to ``training.run_pipeline.run_phase_1_rssm`` via
        ``asyncio.to_thread`` (training is GPU-bound / synchronous).

        Args:
            cfg: Root application settings.
            batch_size: Tuned batch size for this phase.
            checkpoint_dir: Directory for writing checkpoints / artifacts.
        """
        logger.info("rssm_training_start", batch_size=batch_size)

        from training.run_pipeline import run_phase_1_rssm

        tcfg = cfg.training
        data_dir = Path(tcfg.data_dir)
        resume_raw = tcfg.resume_from
        resume_path = Path(resume_raw) if resume_raw else None

        # Override batch_size in a copy of settings (avoids mutating shared state).
        updated_cfg = self._get_settings_with_batch(cfg, batch_size)

        checkpoint = await asyncio.to_thread(
            run_phase_1_rssm,
            updated_cfg,
            data_dir,
            resume_from=resume_path,
        )
        logger.info("rssm_training_complete", checkpoint=str(checkpoint))

    async def _train_warmstart(
        self, cfg: Settings, batch_size: int, checkpoint_dir: Path
    ) -> None:
        """Run warm-start policy tuning phase.

        Delegates to ``training.run_pipeline.run_phase_2_warmstart`` via
        ``asyncio.to_thread``.

        Args:
            cfg: Root application settings.
            batch_size: Tuned batch size for this phase.
            checkpoint_dir: Directory for writing checkpoints / artifacts.
        """
        logger.info("warmstart_training_start", batch_size=batch_size)

        from training.run_pipeline import run_phase_2_warmstart

        data_dir = Path(cfg.training.data_dir)
        rssm_checkpoint = self._get_rssm_checkpoint_path(cfg)

        updated_cfg = self._get_settings_with_batch(cfg, batch_size)

        await asyncio.to_thread(
            run_phase_2_warmstart,
            updated_cfg,
            rssm_checkpoint,
            data_dir,
        )
        logger.info("warmstart_training_complete")

    async def _train_bdi(self, cfg: Settings, batch_size: int, checkpoint_dir: Path) -> None:
        """Run BDI training phase.

        Delegates to ``training.run_pipeline.run_phase_3_bdi`` via
        ``asyncio.to_thread``.

        Args:
            cfg: Root application settings.
            batch_size: Tuned batch size for this phase.
            checkpoint_dir: Directory for writing checkpoints / artifacts.
        """
        logger.info("bdi_training_start", batch_size=batch_size)

        from training.run_pipeline import run_phase_3_bdi

        tcfg = cfg.training
        annotations_path = Path(tcfg.data_dir) / tcfg.bdi_annotations_filename

        updated_cfg = self._get_settings_with_batch(cfg, batch_size)

        await asyncio.to_thread(
            run_phase_3_bdi,
            updated_cfg,
            annotations_path,
        )
        logger.info("bdi_training_complete")

    async def _train_constitutional_rl(
        self, cfg: Settings, batch_size: int, checkpoint_dir: Path
    ) -> None:
        """Run constitutional RL training phase.

        Delegates to ``training.run_pipeline.run_phase_4_constitutional_rl``
        via ``asyncio.to_thread``.

        Args:
            cfg: Root application settings.
            batch_size: Tuned batch size for this phase.
            checkpoint_dir: Directory for writing checkpoints / artifacts.
        """
        logger.info("constitutional_rl_training_start", batch_size=batch_size)

        from training.run_pipeline import run_phase_4_constitutional_rl

        tcfg = cfg.training
        rssm_checkpoint = self._get_rssm_checkpoint_path(cfg)
        policy_init_path = (
            Path(tcfg.weights_dir) / tcfg.mcts_subdir / tcfg.policy_init_filename
        )
        pi_path = policy_init_path if policy_init_path.exists() else None

        updated_cfg = self._get_settings_with_batch(cfg, batch_size)

        await asyncio.to_thread(
            run_phase_4_constitutional_rl,
            updated_cfg,
            rssm_checkpoint,
            pi_path,
        )
        logger.info("constitutional_rl_training_complete")

    async def _wait_for_thermal_clearance(self, phase_log: Any) -> None:
        """Block until GPU temperature is below thermal limit.

        Args:
            phase_log: Bound logger with phase context.
        """
        while await self._gpu_monitor.should_pause():
            phase_log.warning(
                "thermal_pause",
                pause_seconds=self._config.thermal_pause_seconds,
            )
            await asyncio.sleep(self._config.thermal_pause_seconds)
        phase_log.info("thermal_clearance_ok")

    def _checkpoint_exists(self, phase: str) -> bool:
        """Check whether a phase checkpoint marker file exists.

        Args:
            phase: Phase name.

        Returns:
            True if the checkpoint marker exists.
        """
        marker_suffix = self._config.checkpoint_marker_suffix
        return (self._checkpoint_dir / f"{phase}{marker_suffix}").exists()


def _load_settings(config_path: str) -> Settings:
    """Load Settings from a YAML config file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Populated Settings instance.
    """
    import yaml

    path = Path(config_path)
    if not path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    raw = yaml.safe_load(path.read_text()) or {}
    return Settings(**raw)


async def async_main(config_path: str, resume: bool) -> None:
    """Async entry point for the pipeline orchestrator.

    Args:
        config_path: Path to YAML configuration file.
        resume: If True, resume from last incomplete phase.
    """
    settings = _load_settings(config_path)

    pipeline_config = settings.training_pipeline or TrainingPipelineConfig()  # type: ignore[call-arg]

    if resume and pipeline_config.resume_from_phase is None:
        # Auto-detect resume point from existing checkpoints.
        checkpoint_dir = Path(pipeline_config.checkpoint_dir)
        for phase in reversed(pipeline_config.phases):
            marker = f"{phase}{pipeline_config.checkpoint_marker_suffix}"
            if (checkpoint_dir / marker).exists():
                idx = pipeline_config.phases.index(phase)
                if idx + 1 < len(pipeline_config.phases):
                    pipeline_config = pipeline_config.model_copy(
                        update={"resume_from_phase": pipeline_config.phases[idx + 1]}
                    )
                    break

    gpu_monitor = JetsonGPUMonitor(pipeline_config)
    batch_tuner = VRAMBatchTuner(pipeline_config)

    orchestrator = PipelineOrchestrator(
        settings=settings,
        pipeline_config=pipeline_config,
        gpu_monitor=gpu_monitor,
        batch_tuner=batch_tuner,
    )
    await orchestrator.run()


def main() -> None:
    """CLI entry point for the pipeline orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="MouseDroid GPU pre-training pipeline orchestrator"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed phase",
    )
    args = parser.parse_args()

    asyncio.run(async_main(args.config, args.resume))


if __name__ == "__main__":
    main()
    sys.exit(0)
