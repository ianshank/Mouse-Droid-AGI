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
        # In a full implementation these would import and call the actual
        # training modules (train_rssm, warmstart_policy, etc.).
        phase_fn = self._get_phase_runner(phase)
        await phase_fn(batch_size)

        # Write checkpoint marker.
        checkpoint_path = self._checkpoint_dir / f"{phase}.done"
        checkpoint_path.write_text(f"phase={phase}\n")
        logger.info("checkpoint_written", path=str(checkpoint_path))

    def _get_phase_runner(self, phase: str) -> Any:
        """Resolve phase name to its async runner function.

        Args:
            phase: Phase name.

        Returns:
            Async callable accepting batch_size.
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

    async def _train_rssm(self, batch_size: int) -> None:
        """Run RSSM dynamics pretraining on MuJoCo-generated episodes.

        Inert (byte-identical to the prior stub) unless
        ``training.rssm_pretrain_enabled`` is True AND a rover with the
        ``mujoco`` backend is configured. The synchronous torch loop runs in a
        worker thread so the orchestrator event loop (and the cooperative
        thermal-pause check) is not blocked.

        Args:
            batch_size: Tuned batch size for this phase (currently advisory; the
                generator's ``n_episodes`` sets the batch dimension).
        """
        tcfg = self._settings.training
        if tcfg.rssm_vision_finetune_enabled:
            await self._run_vision_finetune()
            return
        if not tcfg.rssm_pretrain_enabled:
            logger.info("rssm_training_skipped", reason="pretrain_disabled", batch_size=batch_size)
            return
        rover = self._settings.rover
        if rover is None or rover.sim.backend != "mujoco":
            logger.info("rssm_training_skipped", reason="non_mujoco_backend")
            return

        import torch  # local import keeps cold-start light

        from mousedroid.factory import build_rover_env, build_rssm_trainable
        from mousedroid.training.domain_randomization import DomainRandomizer
        from mousedroid.training.rover_obs_adapter import RoverObsAdapter
        from mousedroid.training.rssm_pretrainer import RSSMPretrainer
        from mousedroid.training.sim_episode_generator import SimEpisodeGenerator

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_rssm_trainable(self._settings)
        env = build_rover_env(self._settings)
        adapter = RoverObsAdapter(battery_v=rover.sim.mujoco.battery_voltage_const_v)
        generator = SimEpisodeGenerator(
            env,
            adapter,
            n_episodes=tcfg.n_episodes,
            seq_len=tcfg.sequence_length,
            seed=tcfg.rssm_data_seed,
            explore_action_rad_s=tcfg.rssm_explore_action_rad_s,
            explore_smoothing=tcfg.rssm_explore_smoothing,
            domain_randomizer=DomainRandomizer(self._settings.domain_randomization),
        )
        checkpoint = Path(tcfg.weights_dir) / tcfg.rssm_checkpoint_name

        def _run() -> list[float]:
            batch = generator.generate()
            trainer = RSSMPretrainer(
                model,
                lr=tcfg.learning_rate,
                grad_clip=tcfg.rssm_grad_clip,
                amp=self._config.amp_enabled,
                device=device,
            )
            return trainer.train([batch], epochs=tcfg.epochs, checkpoint_path=checkpoint)

        logger.info("rssm_training_started", n_episodes=tcfg.n_episodes, device=str(device))
        history = await asyncio.to_thread(_run)
        env.close()
        logger.info("rssm_training_done", first_loss=history[0], last_loss=history[-1])

    async def _run_vision_finetune(self) -> None:
        """Vision-on fine-tune: migrate a vision-OFF checkpoint and train with RGB.

        Inert unless a ``mujoco`` rover is configured AND
        ``training.rssm_finetune_checkpoint`` points at an existing vision-OFF
        checkpoint. Renders RGB → ``MeanPoolExtractor`` → ``vision_features`` and
        fine-tunes the migrated vision-ON RSSM. Runs the blocking torch loop in a
        worker thread (thermal-pause safe).
        """
        tcfg = self._settings.training
        rover = self._settings.rover
        if rover is None or rover.sim.backend != "mujoco":
            logger.info("rssm_vision_finetune_skipped", reason="non_mujoco_backend")
            return
        checkpoint = Path(tcfg.rssm_finetune_checkpoint)
        if not tcfg.rssm_finetune_checkpoint or not checkpoint.exists():
            logger.info("rssm_vision_finetune_skipped", reason="missing_checkpoint")
            return

        import torch  # local import keeps cold-start light

        from mousedroid.factory import (
            build_rover_env,
            build_rssm_vision_finetune,
            build_vision_feature_extractor,
        )
        from mousedroid.training.domain_randomization import DomainRandomizer
        from mousedroid.training.rover_obs_adapter import RoverObsAdapter
        from mousedroid.training.rssm_pretrainer import RSSMPretrainer
        from mousedroid.training.sim_episode_generator import SimEpisodeGenerator

        # Force render_vision on for the fine-tune env (it must produce frames).
        render_cfg = self._settings.model_copy(
            update={
                "rover": rover.model_copy(
                    update={
                        "sim": rover.sim.model_copy(
                            update={
                                "mujoco": rover.sim.mujoco.model_copy(
                                    update={"render_vision": True}
                                )
                            }
                        )
                    }
                )
            }
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_rssm_vision_finetune(self._settings, checkpoint)
        env = build_rover_env(render_cfg)
        adapter = RoverObsAdapter(battery_v=rover.sim.mujoco.battery_voltage_const_v)
        generator = SimEpisodeGenerator(
            env,
            adapter,
            n_episodes=tcfg.n_episodes,
            seq_len=tcfg.sequence_length,
            seed=tcfg.rssm_data_seed,
            explore_action_rad_s=tcfg.rssm_explore_action_rad_s,
            explore_smoothing=tcfg.rssm_explore_smoothing,
            domain_randomizer=DomainRandomizer(self._settings.domain_randomization),
            feature_extractor=build_vision_feature_extractor(self._settings),
        )
        out_ckpt = Path(tcfg.weights_dir) / tcfg.rssm_vision_checkpoint_name

        def _run() -> list[float]:
            batch = generator.generate()
            trainer = RSSMPretrainer(
                model,
                lr=tcfg.learning_rate,
                grad_clip=tcfg.rssm_grad_clip,
                amp=self._config.amp_enabled,
                device=device,
            )
            return trainer.train(
                [batch], epochs=tcfg.rssm_finetune_epochs, checkpoint_path=out_ckpt
            )

        logger.info("rssm_vision_finetune_started", checkpoint=str(checkpoint), device=str(device))
        history = await asyncio.to_thread(_run)
        env.close()
        logger.info("rssm_vision_finetune_done", first_loss=history[0], last_loss=history[-1])

    async def _train_warmstart(self, batch_size: int) -> None:
        """Run warm-start policy tuning phase."""
        logger.info("warmstart_training", batch_size=batch_size)

    async def _train_bdi(self, batch_size: int) -> None:
        """Run BDI training phase."""
        logger.info("bdi_training", batch_size=batch_size)

    async def _train_constitutional_rl(self, batch_size: int) -> None:
        """Run constitutional RL training phase."""
        logger.info("constitutional_rl_training", batch_size=batch_size)

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
        return (self._checkpoint_dir / f"{phase}.done").exists()


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
            if (checkpoint_dir / f"{phase}.done").exists():
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
