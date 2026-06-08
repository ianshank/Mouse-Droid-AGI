"""Integration: pipeline orchestrator emits parent + nested phase runs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("mlflow")
from mlflow import MlflowClient

from mousedroid.config.schema import (
    ExperimentLoggerConfig,
    ObservabilityConfig,
    Settings,
    TrainingPipelineConfig,
)
from mousedroid.training.observability.mlflow_logger import (
    MlflowExperimentLogger,
)
from mousedroid.training.pipeline_orchestrator import PipelineOrchestrator


@pytest.fixture
def tracking_uri(tmp_path: Path) -> str:
    return f"file:{tmp_path / 'mlruns'}"


@pytest.fixture
def settings(tracking_uri: str) -> Settings:
    base = Settings(mock_hardware=True)
    return base.model_copy(
        update={
            "observability": ObservabilityConfig(
                experiment_logger=ExperimentLoggerConfig(
                    backend="mlflow",
                    tracking_uri=tracking_uri,
                    experiment_name="pipeline-test",
                ),
            ),
            "training_pipeline": TrainingPipelineConfig(
                phases=["rssm", "warmstart"],
                checkpoint_dir=str(Path(tracking_uri.removeprefix("file:")).parent / "ckpt"),
                batch_sizes={"rssm": 16, "warmstart": 16},
                amp_enabled=False,
                resume_from_phase=None,
            ),
        }
    )


@pytest.mark.asyncio
async def test_run_creates_parent_and_nested_phase_runs(
    settings: Settings, tracking_uri: str
) -> None:
    logger = MlflowExperimentLogger(
        tracking_uri=tracking_uri,
        experiment_name=settings.observability.experiment_logger.experiment_name,  # type: ignore[union-attr]
    )
    gpu = MagicMock()
    gpu.should_pause = AsyncMock(return_value=False)
    tuner = MagicMock()
    tuner.tune_batch_size = MagicMock(side_effect=lambda phase, base: base)

    orch = PipelineOrchestrator(
        settings=settings,
        pipeline_config=settings.training_pipeline,  # type: ignore[arg-type]
        gpu_monitor=gpu,
        batch_tuner=tuner,
        experiment_logger=logger,
    )
    await orch.run()

    client = MlflowClient(tracking_uri=tracking_uri)
    runs = client.search_runs(
        experiment_ids=[logger._experiment_id],
        order_by=["attributes.start_time ASC"],
    )
    # 1 parent + 2 phase children
    assert len(runs) == 3
    parent = runs[0]
    children = runs[1:]
    assert parent.data.tags.get("mlflow.parentRunId") is None
    assert {c.data.tags.get("phase") for c in children} == {"rssm", "warmstart"}
    for c in children:
        assert c.data.tags.get("mlflow.parentRunId") == parent.info.run_id
    assert parent.info.status == "FINISHED"
    assert all(c.info.status == "FINISHED" for c in children)


@pytest.mark.asyncio
async def test_run_marks_parent_failed_when_phase_raises(
    settings: Settings, tracking_uri: str
) -> None:
    logger = MlflowExperimentLogger(
        tracking_uri=tracking_uri,
        experiment_name=settings.observability.experiment_logger.experiment_name,  # type: ignore[union-attr]
    )
    gpu = MagicMock()
    gpu.should_pause = AsyncMock(return_value=False)
    tuner = MagicMock()
    tuner.tune_batch_size = MagicMock(side_effect=lambda phase, base: base)

    orch = PipelineOrchestrator(
        settings=settings,
        pipeline_config=settings.training_pipeline,  # type: ignore[arg-type]
        gpu_monitor=gpu,
        batch_tuner=tuner,
        experiment_logger=logger,
    )
    # Force the first phase to raise.
    orch._train_rssm = AsyncMock(side_effect=RuntimeError("simulated"))
    with pytest.raises(RuntimeError, match="simulated"):
        await orch.run()

    client = MlflowClient(tracking_uri=tracking_uri)
    runs = client.search_runs(
        experiment_ids=[logger._experiment_id],
        order_by=["attributes.start_time ASC"],
    )
    parent = runs[0]
    assert parent.info.status == "FAILED"


@pytest.mark.asyncio
async def test_configured_run_name_reaches_mlflow_run(tracking_uri: str) -> None:
    """cfg.observability.experiment_logger.run_name is passed to the parent run."""
    base = Settings(mock_hardware=True)
    settings_with_name = base.model_copy(
        update={
            "observability": ObservabilityConfig(
                experiment_logger=ExperimentLoggerConfig(
                    backend="mlflow",
                    tracking_uri=tracking_uri,
                    experiment_name="pipeline-name-test",
                    run_name="my-pipeline-name",
                ),
            ),
            "training_pipeline": TrainingPipelineConfig(
                phases=["rssm"],
                checkpoint_dir=str(Path(tracking_uri.removeprefix("file:")).parent / "ckpt2"),
                batch_sizes={"rssm": 4},
                amp_enabled=False,
                resume_from_phase=None,
            ),
        }
    )
    logger = MlflowExperimentLogger(
        tracking_uri=tracking_uri,
        experiment_name="pipeline-name-test",
        run_name="my-pipeline-name",
    )
    gpu = MagicMock()
    gpu.should_pause = AsyncMock(return_value=False)
    tuner = MagicMock()
    tuner.tune_batch_size = MagicMock(side_effect=lambda phase, base: base)

    orch = PipelineOrchestrator(
        settings=settings_with_name,
        pipeline_config=settings_with_name.training_pipeline,  # type: ignore[arg-type]
        gpu_monitor=gpu,
        batch_tuner=tuner,
        experiment_logger=logger,
    )
    await orch.run()

    client = MlflowClient(tracking_uri=tracking_uri)
    runs = client.search_runs(
        experiment_ids=[logger._experiment_id],
        order_by=["attributes.start_time ASC"],
    )
    parent = runs[0]
    assert parent.info.run_name == "my-pipeline-name"


@pytest.mark.asyncio
async def test_log_artifacts_false_skips_checkpoint_upload(
    settings: Settings, tracking_uri: str
) -> None:
    """When log_artifacts=False, no checkpoint artifact is uploaded for a phase run."""
    base = Settings(mock_hardware=True)
    settings_no_artifacts = base.model_copy(
        update={
            "observability": ObservabilityConfig(
                experiment_logger=ExperimentLoggerConfig(
                    backend="mlflow",
                    tracking_uri=tracking_uri,
                    experiment_name="pipeline-no-artifact-test",
                    log_artifacts=False,
                ),
            ),
            "training_pipeline": TrainingPipelineConfig(
                phases=["rssm"],
                checkpoint_dir=str(Path(tracking_uri.removeprefix("file:")).parent / "ckpt3"),
                batch_sizes={"rssm": 4},
                amp_enabled=False,
                resume_from_phase=None,
            ),
        }
    )
    logger = MlflowExperimentLogger(
        tracking_uri=tracking_uri,
        experiment_name="pipeline-no-artifact-test",
    )
    gpu = MagicMock()
    gpu.should_pause = AsyncMock(return_value=False)
    tuner = MagicMock()
    tuner.tune_batch_size = MagicMock(side_effect=lambda phase, base: base)

    orch = PipelineOrchestrator(
        settings=settings_no_artifacts,
        pipeline_config=settings_no_artifacts.training_pipeline,  # type: ignore[arg-type]
        gpu_monitor=gpu,
        batch_tuner=tuner,
        experiment_logger=logger,
    )
    await orch.run()

    client = MlflowClient(tracking_uri=tracking_uri)
    runs = client.search_runs(
        experiment_ids=[logger._experiment_id],
        order_by=["attributes.start_time ASC"],
    )
    # Find the rssm phase child run
    phase_run = next(r for r in runs if r.data.tags.get("phase") == "rssm")
    artifacts = client.list_artifacts(phase_run.info.run_id)
    assert artifacts == [], "Expected no artifacts when log_artifacts=False"
