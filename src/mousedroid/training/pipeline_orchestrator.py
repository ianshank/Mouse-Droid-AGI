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
import json
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mousedroid.config.schema import RoverConfig, Settings, TrainingPipelineConfig
from mousedroid.logging.setup import get_logger
from mousedroid.training.batch_tuner import VRAMBatchTuner
from mousedroid.training.gpu_monitor import JetsonGPUMonitor

if TYPE_CHECKING:
    from mousedroid.hardware.camera.feature_extractor import FeatureExtractorProtocol
    from mousedroid.sim.protocols import RoverEnvProtocol
    from mousedroid.training.observability import ExperimentLoggerProtocol
    from mousedroid.world_model.rssm import RSSM

logger = get_logger(__name__)


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
        experiment_logger: Optional experiment logger. Defaults to
            ``NoOpExperimentLogger`` so callers never need a ``None``
            guard.
    """

    def __init__(
        self,
        settings: Settings,
        pipeline_config: TrainingPipelineConfig,
        gpu_monitor: JetsonGPUMonitor | Any,
        batch_tuner: VRAMBatchTuner | Any,
        *,
        experiment_logger: ExperimentLoggerProtocol | None = None,
    ) -> None:
        self._settings = settings
        self._config = pipeline_config
        self._gpu_monitor = gpu_monitor
        self._batch_tuner = batch_tuner
        self._checkpoint_dir = Path(pipeline_config.checkpoint_dir)
        # Default to NoOp so call sites never need a None guard. The factory
        # provides the real logger when the user opts in.
        if experiment_logger is None:
            from mousedroid.training.observability import NoOpExperimentLogger

            experiment_logger = NoOpExperimentLogger()
        self._experiment_logger: ExperimentLoggerProtocol = experiment_logger

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

        observability_cfg = self._settings.observability
        configured_run_name = (
            observability_cfg.experiment_logger.run_name if observability_cfg is not None else None
        )
        self._experiment_logger.start_run(
            run_name=configured_run_name,  # may be None → impl-defined default
            params={
                "total_phases": len(phases),
                "start_index": start_idx,
                "amp_enabled": self._config.amp_enabled,
            },
            tags={"track": "training"},
        )

        if observability_cfg is not None and observability_cfg.experiment_logger.log_artifacts:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    settings_path = Path(tmpdir) / "resolved_settings.json"
                    settings_path.write_text(
                        json.dumps(
                            self._settings.model_dump(mode="json"),
                            indent=2,
                            default=str,
                        ),
                        encoding="utf-8",
                    )
                    self._experiment_logger.log_artifact(str(settings_path))
                    logger.info(
                        "pipeline_settings_artifact_logged",
                        artifact_path=str(settings_path),
                    )
            except asyncio.CancelledError:
                raise  # cooperative cancellation is not a backend failure
            except Exception as exc:  # broad — settings dump must never break the run
                logger.warning(
                    "pipeline_settings_artifact_failed",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

        run_status = "FINISHED"

        try:
            for idx in range(start_idx, len(phases)):
                phase = phases[idx]
                phase_log = logger.bind(phase=phase, phase_index=idx)

                await self._wait_for_thermal_clearance(phase_log)

                base_batch = self._config.batch_sizes.get(phase, self._settings.training.batch_size)
                tuned_batch = self._batch_tuner.tune_batch_size(phase, base_batch)

                if idx > start_idx:
                    prev_phase = phases[idx - 1]
                    if not self._checkpoint_exists(prev_phase):
                        msg = (
                            f"Missing checkpoint for phase '{prev_phase}' — "
                            f"cannot proceed to '{phase}'"
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
        except Exception:
            run_status = "FAILED"
            raise
        finally:
            self._experiment_logger.end_run(status=run_status)

        logger.info("pipeline_completed", phases_run=phases[start_idx:])

    async def _run_phase(self, phase: str, batch_size: int) -> None:
        """Execute a single training phase under a nested phase run.

        Each phase creates its own checkpoint file on completion and uploads
        it as a phase-run artifact.

        Args:
            phase: Phase name (e.g. "rssm").
            batch_size: Tuned batch size for this phase.
        """
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ctx = self._experiment_logger.start_phase(
            phase=phase,
            params={"batch_size": batch_size, "amp_enabled": self._config.amp_enabled},
            tags={"phase": phase},
        )
        phase_status = "FINISHED"
        try:
            phase_fn = self._get_phase_runner(phase)
            await phase_fn(batch_size)
            checkpoint_path = self._checkpoint_dir / f"{phase}.done"
            checkpoint_path.write_text(f"phase={phase}\n")
            logger.info("checkpoint_written", path=str(checkpoint_path))
            observability_cfg = self._settings.observability
            if observability_cfg is not None and observability_cfg.experiment_logger.log_artifacts:
                self._experiment_logger.log_phase_artifact(ctx, str(checkpoint_path))
        except Exception:
            phase_status = "FAILED"
            raise
        finally:
            self._experiment_logger.end_phase(ctx, status=phase_status)

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
        rover = self._settings.rover
        handled = False

        # Pretrain (vision OFF) runs FIRST so a fine-tune in the same pass can
        # consume the freshly written checkpoint. The two flags are independent —
        # vision fine-tune must NOT short-circuit pretraining.
        if tcfg.rssm_pretrain_enabled:
            if rover is None or rover.sim.backend != "mujoco":
                logger.info("rssm_training_skipped", reason="non_mujoco_backend")
            else:
                from mousedroid.factory import build_rover_env, build_rssm_trainable

                await self._run_rssm_training(
                    model=build_rssm_trainable(self._settings),
                    env=build_rover_env(self._settings),
                    battery_v=rover.sim.mujoco.battery_voltage_const_v,
                    checkpoint=Path(tcfg.weights_dir) / tcfg.rssm_checkpoint_name,
                    epochs=tcfg.epochs,
                    event_prefix="rssm_training",
                )
            handled = True

        if tcfg.rssm_vision_finetune_enabled:
            await self._run_vision_finetune()
            handled = True

        if not handled:
            logger.info("rssm_training_skipped", reason="pretrain_disabled", batch_size=batch_size)

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

        from mousedroid.factory import (
            build_rover_env,
            build_rssm_vision_finetune,
            build_vision_feature_extractor,
        )

        # Force render_vision on for the fine-tune env (it must produce frames);
        # build the model + extractor from the SAME cfg so dims never diverge.
        render_cfg = self._render_enabled_settings(rover)
        await self._run_rssm_training(
            model=build_rssm_vision_finetune(render_cfg, checkpoint),
            env=build_rover_env(render_cfg),
            battery_v=rover.sim.mujoco.battery_voltage_const_v,
            checkpoint=Path(tcfg.weights_dir) / tcfg.rssm_vision_checkpoint_name,
            epochs=tcfg.rssm_finetune_epochs,
            event_prefix="rssm_vision_finetune",
            feature_extractor=build_vision_feature_extractor(render_cfg),
        )

    def _render_enabled_settings(self, rover: RoverConfig) -> Settings:
        """Return settings with ``rover.sim.mujoco.render_vision`` forced on."""
        return self._settings.model_copy(
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

    async def _run_rssm_training(
        self,
        *,
        model: RSSM,
        env: RoverEnvProtocol,
        battery_v: float,
        checkpoint: Path,
        epochs: int,
        event_prefix: str,
        feature_extractor: FeatureExtractorProtocol | None = None,
    ) -> None:
        """Shared RSSM training loop for the pretrain + vision-fine-tune paths.

        Builds the obs adapter + episode generator, runs the synchronous torch
        loop in a worker thread (so the orchestrator event loop + thermal-pause
        check are not blocked), closes the env, and logs ``{event_prefix}_started``
        / ``{event_prefix}_done``. Centralising this prevents the two phases from
        silently diverging on shared knobs.
        """
        import torch  # local import keeps cold-start light

        from mousedroid.training.domain_randomization import DomainRandomizer
        from mousedroid.training.rover_obs_adapter import RoverObsAdapter
        from mousedroid.training.rssm_pretrainer import RSSMPretrainer
        from mousedroid.training.sim_episode_generator import SimEpisodeGenerator

        tcfg = self._settings.training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adapter = RoverObsAdapter(battery_v=battery_v)
        generator = SimEpisodeGenerator(
            env,
            adapter,
            n_episodes=tcfg.n_episodes,
            seq_len=tcfg.sequence_length,
            seed=tcfg.rssm_data_seed,
            explore_action_rad_s=tcfg.rssm_explore_action_rad_s,
            explore_smoothing=tcfg.rssm_explore_smoothing,
            domain_randomizer=DomainRandomizer(self._settings.domain_randomization),
            feature_extractor=feature_extractor,
        )

        def _run() -> list[float]:
            batch = generator.generate()
            trainer = RSSMPretrainer(
                model,
                lr=tcfg.learning_rate,
                grad_clip=tcfg.rssm_grad_clip,
                amp=self._config.amp_enabled,
                device=device,
            )
            return trainer.train([batch], epochs=epochs, checkpoint_path=checkpoint)

        # Static event name + structured `phase` field (no runtime-built event
        # strings) — keeps the structured-logging contract stable for consumers.
        logger.info(
            "rssm_phase_started", phase=event_prefix, n_episodes=tcfg.n_episodes, device=str(device)
        )
        try:
            history = await asyncio.to_thread(_run)
        finally:
            # Always release the env (+ its MuJoCo renderer/GL context) even if
            # episode generation or training raises — otherwise the context leaks.
            env.close()
        logger.info(
            "rssm_phase_done", phase=event_prefix, first_loss=history[0], last_loss=history[-1]
        )

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

    pipeline_config = settings.training_pipeline or TrainingPipelineConfig()

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

    # Resolve the experiment logger from config so a YAML/env opt-in
    # (observability.experiment_logger.backend = mlflow) actually takes effect
    # on the CLI path — NoOp otherwise.
    from mousedroid.factory import build_experiment_logger

    orchestrator = PipelineOrchestrator(
        settings=settings,
        pipeline_config=pipeline_config,
        gpu_monitor=gpu_monitor,
        batch_tuner=batch_tuner,
        experiment_logger=build_experiment_logger(settings),
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
