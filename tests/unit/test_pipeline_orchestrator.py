"""Unit tests for pipeline orchestrator module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from mousedroid.config.schema import Settings, TrainingPipelineConfig
from mousedroid.training.pipeline_orchestrator import (
    PipelineOrchestrator,
    _load_settings,
    async_main,
)


@pytest.fixture
def pipeline_config(tmp_path: Path) -> TrainingPipelineConfig:
    """Create pipeline config with temp checkpoint dir."""
    return TrainingPipelineConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )


@pytest.fixture
def settings() -> Settings:
    """Create minimal Settings for testing."""
    return Settings(mock_hardware=True, ultrasonic=None)


@pytest.fixture
def gpu_monitor() -> AsyncMock:
    """Create a mock GPU monitor that never pauses."""
    mock = AsyncMock()
    mock.should_pause.return_value = False
    mock.get_temperature.return_value = 50.0
    return mock


@pytest.fixture
def batch_tuner() -> MagicMock:
    """Create a mock batch tuner that returns base size."""
    mock = MagicMock()
    mock.tune_batch_size.side_effect = lambda phase, base: base
    return mock


@pytest.fixture
def orchestrator(
    settings: Settings,
    pipeline_config: TrainingPipelineConfig,
    gpu_monitor: AsyncMock,
    batch_tuner: MagicMock,
) -> PipelineOrchestrator:
    """Create an orchestrator with mocks."""
    return PipelineOrchestrator(
        settings=settings,
        pipeline_config=pipeline_config,
        gpu_monitor=gpu_monitor,
        batch_tuner=batch_tuner,
    )


@pytest.mark.asyncio
async def test_all_phases_run_in_order(
    orchestrator: PipelineOrchestrator,
    pipeline_config: TrainingPipelineConfig,
) -> None:
    """All 4 phases should execute in sequence."""
    phases_run: list[str] = []

    async def track_phase(phase: str, batch_size: int) -> None:
        phases_run.append(phase)
        # Write checkpoint to satisfy validation
        cp_dir = Path(pipeline_config.checkpoint_dir)
        cp_dir.mkdir(parents=True, exist_ok=True)
        (cp_dir / f"{phase}.done").write_text(f"phase={phase}\n")

    with patch.object(orchestrator, "_run_phase", side_effect=track_phase):
        await orchestrator.run()

    assert phases_run == ["rssm", "warmstart", "bdi", "constitutional_rl"]


@pytest.mark.asyncio
async def test_resume_skips_prior_phases(
    settings: Settings,
    gpu_monitor: AsyncMock,
    batch_tuner: MagicMock,
    tmp_path: Path,
) -> None:
    """Resume from phase 3 should skip phases 1 and 2."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    # Create checkpoint for warmstart (phase before bdi)
    (checkpoint_dir / "warmstart.done").write_text("phase=warmstart\n")

    config = TrainingPipelineConfig(
        checkpoint_dir=str(checkpoint_dir),
        resume_from_phase="bdi",
    )
    orchestrator = PipelineOrchestrator(
        settings=settings,
        pipeline_config=config,
        gpu_monitor=gpu_monitor,
        batch_tuner=batch_tuner,
    )

    phases_run: list[str] = []

    async def track_phase(phase: str, batch_size: int) -> None:
        phases_run.append(phase)
        (checkpoint_dir / f"{phase}.done").write_text(f"phase={phase}\n")

    with patch.object(orchestrator, "_run_phase", side_effect=track_phase):
        await orchestrator.run()

    assert "rssm" not in phases_run
    assert "warmstart" not in phases_run
    assert phases_run == ["bdi", "constitutional_rl"]


@pytest.mark.asyncio
async def test_phase_failure_halts_pipeline(
    orchestrator: PipelineOrchestrator,
    pipeline_config: TrainingPipelineConfig,
) -> None:
    """Pipeline should halt when a phase raises an exception."""

    async def fail_on_warmstart(phase: str, batch_size: int) -> None:
        cp_dir = Path(pipeline_config.checkpoint_dir)
        cp_dir.mkdir(parents=True, exist_ok=True)
        if phase == "warmstart":
            raise RuntimeError("training diverged")
        (cp_dir / f"{phase}.done").write_text(f"phase={phase}\n")

    with (
        patch.object(orchestrator, "_run_phase", side_effect=fail_on_warmstart),
        pytest.raises(RuntimeError, match="training diverged"),
    ):
        await orchestrator.run()


@pytest.mark.asyncio
async def test_missing_checkpoint_halts_pipeline(
    settings: Settings,
    gpu_monitor: AsyncMock,
    batch_tuner: MagicMock,
    tmp_path: Path,
) -> None:
    """Pipeline halts if prior phase checkpoint is missing."""
    config = TrainingPipelineConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    orchestrator = PipelineOrchestrator(
        settings=settings,
        pipeline_config=config,
        gpu_monitor=gpu_monitor,
        batch_tuner=batch_tuner,
    )

    call_count = 0

    async def run_but_no_checkpoint(phase: str, batch_size: int) -> None:
        nonlocal call_count
        call_count += 1
        # Intentionally do NOT write checkpoint

    with (
        patch.object(orchestrator, "_run_phase", side_effect=run_but_no_checkpoint),
        pytest.raises(RuntimeError, match="Missing checkpoint"),
    ):
        await orchestrator.run()

    # Only the first phase should have run before failure
    assert call_count == 1


@pytest.mark.asyncio
async def test_config_driven_phase_list(
    settings: Settings,
    gpu_monitor: AsyncMock,
    batch_tuner: MagicMock,
    tmp_path: Path,
) -> None:
    """Custom phase list from config should be respected."""
    config = TrainingPipelineConfig(
        phases=["rssm", "bdi"],
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    orchestrator = PipelineOrchestrator(
        settings=settings,
        pipeline_config=config,
        gpu_monitor=gpu_monitor,
        batch_tuner=batch_tuner,
    )

    phases_run: list[str] = []

    async def track_phase(phase: str, batch_size: int) -> None:
        phases_run.append(phase)
        cp_dir = Path(config.checkpoint_dir)
        cp_dir.mkdir(parents=True, exist_ok=True)
        (cp_dir / f"{phase}.done").write_text(f"phase={phase}\n")

    with patch.object(orchestrator, "_run_phase", side_effect=track_phase):
        await orchestrator.run()

    assert phases_run == ["rssm", "bdi"]


@pytest.mark.asyncio
async def test_checkpoint_detection(tmp_path: Path) -> None:
    """Checkpoint existence detection should work correctly."""
    config = TrainingPipelineConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    settings = Settings(mock_hardware=True, ultrasonic=None)
    orchestrator = PipelineOrchestrator(
        settings=settings,
        pipeline_config=config,
        gpu_monitor=AsyncMock(),
        batch_tuner=MagicMock(),
    )

    # No checkpoint yet
    assert orchestrator._checkpoint_exists("rssm") is False

    # Create checkpoint
    cp_dir = tmp_path / "checkpoints"
    cp_dir.mkdir()
    (cp_dir / "rssm.done").write_text("phase=rssm\n")

    assert orchestrator._checkpoint_exists("rssm") is True
    assert orchestrator._checkpoint_exists("warmstart") is False


@pytest.mark.asyncio
async def test_run_phase_creates_checkpoint(
    settings: Settings,
    gpu_monitor: AsyncMock,
    batch_tuner: MagicMock,
    tmp_path: Path,
) -> None:
    """Running a phase should create a checkpoint marker file."""
    config = TrainingPipelineConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    orchestrator = PipelineOrchestrator(
        settings=settings,
        pipeline_config=config,
        gpu_monitor=gpu_monitor,
        batch_tuner=batch_tuner,
    )
    with patch(
        "mousedroid.training.pipeline_orchestrator.asyncio.to_thread",
        new_callable=AsyncMock,
        return_value=Path("fake.pt"),
    ):
        await orchestrator._run_phase("rssm", 16)
    assert (tmp_path / "checkpoints" / "rssm.done").exists()


@pytest.mark.asyncio
async def test_get_phase_runner_unknown_phase(
    settings: Settings,
    gpu_monitor: AsyncMock,
    batch_tuner: MagicMock,
    tmp_path: Path,
) -> None:
    """Unknown phase name should raise ValueError."""
    config = TrainingPipelineConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    orchestrator = PipelineOrchestrator(
        settings=settings,
        pipeline_config=config,
        gpu_monitor=gpu_monitor,
        batch_tuner=batch_tuner,
    )
    with pytest.raises(ValueError, match="Unknown training phase"):
        orchestrator._get_phase_runner("nonexistent")


@pytest.mark.asyncio
async def test_phase_runners_exist_for_all_default_phases(
    settings: Settings,
    gpu_monitor: AsyncMock,
    batch_tuner: MagicMock,
    tmp_path: Path,
) -> None:
    """All default phases should have registered runners."""
    config = TrainingPipelineConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    orchestrator = PipelineOrchestrator(
        settings=settings,
        pipeline_config=config,
        gpu_monitor=gpu_monitor,
        batch_tuner=batch_tuner,
    )
    for phase in ["rssm", "warmstart", "bdi", "constitutional_rl"]:
        runner = orchestrator._get_phase_runner(phase)
        assert callable(runner)


@pytest.mark.asyncio
async def test_thermal_wait_pauses_then_continues(
    settings: Settings,
    batch_tuner: MagicMock,
    tmp_path: Path,
) -> None:
    """Pipeline should wait for thermal clearance before starting a phase."""
    config = TrainingPipelineConfig(
        phases=["rssm"],
        checkpoint_dir=str(tmp_path / "checkpoints"),
        thermal_pause_seconds=0.01,  # Very short for test
    )
    # First call: True (pause), second call: False (continue)
    gpu_monitor = AsyncMock()
    gpu_monitor.should_pause.side_effect = [True, False]

    orchestrator = PipelineOrchestrator(
        settings=settings,
        pipeline_config=config,
        gpu_monitor=gpu_monitor,
        batch_tuner=batch_tuner,
    )
    with patch(
        "mousedroid.training.pipeline_orchestrator.asyncio.to_thread",
        new_callable=AsyncMock,
        return_value=Path("fake.pt"),
    ):
        await orchestrator.run()
    assert gpu_monitor.should_pause.call_count == 2


@pytest.mark.asyncio
async def test_resume_from_invalid_phase_raises(
    settings: Settings,
    gpu_monitor: AsyncMock,
    batch_tuner: MagicMock,
    tmp_path: Path,
) -> None:
    """Resuming from a nonexistent phase should raise RuntimeError."""
    config = TrainingPipelineConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        resume_from_phase="nonexistent_phase",
    )
    orchestrator = PipelineOrchestrator(
        settings=settings,
        pipeline_config=config,
        gpu_monitor=gpu_monitor,
        batch_tuner=batch_tuner,
    )
    with pytest.raises(RuntimeError, match="not in configured phases"):
        await orchestrator.run()


def test_load_settings_from_yaml(tmp_path: Path) -> None:
    """_load_settings should load a valid YAML config."""
    config_data = {
        "mock_hardware": True,
        "training_pipeline": {
            "phases": ["rssm", "bdi"],
            "checkpoint_dir": str(tmp_path / "cp"),
        },
    }
    yaml_path = tmp_path / "test_config.yaml"
    yaml_path.write_text(yaml.dump(config_data))

    settings = _load_settings(str(yaml_path))
    assert settings.mock_hardware is True
    assert settings.training_pipeline is not None
    assert settings.training_pipeline.phases == ["rssm", "bdi"]


def test_load_settings_file_not_found() -> None:
    """_load_settings should raise FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        _load_settings("/nonexistent/path/config.yaml")


@pytest.mark.asyncio
async def test_async_main_runs_pipeline(tmp_path: Path) -> None:
    """async_main should load config and run the pipeline."""
    config_data = {
        "mock_hardware": True,
        "training_pipeline": {
            "phases": ["rssm"],
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "thermal_sysfs_path": str(tmp_path / "nonexistent_thermal"),
        },
    }
    yaml_path = tmp_path / "test_config.yaml"
    yaml_path.write_text(yaml.dump(config_data))

    mock_monitor = AsyncMock()
    mock_monitor.should_pause = AsyncMock(return_value=False)

    mock_batch_tuner = MagicMock()
    mock_batch_tuner.tune_batch_size = MagicMock(side_effect=lambda p, b: b)

    with (
        patch(
            "mousedroid.training.pipeline_orchestrator.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=Path("fake.pt"),
        ),
        patch(
            "mousedroid.training.pipeline_orchestrator.JetsonGPUMonitor",
            return_value=mock_monitor,
        ),
        patch(
            "mousedroid.training.pipeline_orchestrator.VRAMBatchTuner",
            return_value=mock_batch_tuner,
        ),
    ):
        await async_main(str(yaml_path), resume=False)
    # Verify checkpoint was written
    assert (tmp_path / "checkpoints" / "rssm.done").exists()


@pytest.mark.asyncio
async def test_async_main_resume_auto_detects(tmp_path: Path) -> None:
    """async_main with resume=True should auto-detect completed phases."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "rssm.done").write_text("phase=rssm\n")

    config_data = {
        "mock_hardware": True,
        "training_pipeline": {
            "phases": ["rssm", "warmstart"],
            "checkpoint_dir": str(checkpoint_dir),
            "thermal_sysfs_path": str(tmp_path / "nonexistent_thermal"),
        },
    }
    yaml_path = tmp_path / "test_config.yaml"
    yaml_path.write_text(yaml.dump(config_data))

    mock_monitor = AsyncMock()
    mock_monitor.should_pause = AsyncMock(return_value=False)

    mock_batch_tuner = MagicMock()
    mock_batch_tuner.tune_batch_size = MagicMock(side_effect=lambda p, b: b)

    with (
        patch(
            "mousedroid.training.pipeline_orchestrator.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "mousedroid.training.pipeline_orchestrator.JetsonGPUMonitor",
            return_value=mock_monitor,
        ),
        patch(
            "mousedroid.training.pipeline_orchestrator.VRAMBatchTuner",
            return_value=mock_batch_tuner,
        ),
    ):
        await async_main(str(yaml_path), resume=True)
    # Only warmstart should have been run (rssm skipped)
    assert (checkpoint_dir / "warmstart.done").exists()
