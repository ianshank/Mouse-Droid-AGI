"""Smoke tests for the GPU pre-training pipeline.

Exercises pipeline orchestrator instantiation, phase execution,
resume logic, thermal pause triggers, batch auto-tuning, checkpoint
directory creation, and config loading — all without requiring a real GPU.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from mousedroid.config.schema import Settings, TrainingPipelineConfig
from mousedroid.training.batch_tuner import VRAMBatchTuner
from mousedroid.training.gpu_monitor import JetsonGPUMonitor
from mousedroid.training.pipeline_orchestrator import (
    PipelineOrchestrator,
    _load_settings,
    async_main,
)

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_config(tmp_path: Path) -> TrainingPipelineConfig:
    """Create pipeline config with temp checkpoint dir."""
    return TrainingPipelineConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        thermal_pause_seconds=0.01,
    )


@pytest.fixture
def settings() -> Settings:
    """Create minimal Settings for testing."""
    return Settings(mock_hardware=True, ultrasonic=None)


@pytest.fixture
def gpu_monitor_mock() -> AsyncMock:
    """Create a mock GPU monitor that never pauses."""
    mock = AsyncMock()
    mock.should_pause.return_value = False
    mock.get_temperature.return_value = 50.0
    mock.get_vram_free_mb.return_value = 4096
    return mock


@pytest.fixture
def batch_tuner_mock() -> MagicMock:
    """Create a mock batch tuner that returns base size."""
    mock = MagicMock()
    mock.tune_batch_size.side_effect = lambda phase, base: base
    return mock


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_orchestrator_instantiation(
    settings: Settings,
    pipeline_config: TrainingPipelineConfig,
    gpu_monitor_mock: AsyncMock,
    batch_tuner_mock: MagicMock,
) -> None:
    """Pipeline orchestrator can be instantiated with mock config."""
    orch = PipelineOrchestrator(
        settings=settings,
        pipeline_config=pipeline_config,
        gpu_monitor=gpu_monitor_mock,
        batch_tuner=batch_tuner_mock,
    )
    assert orch is not None
    assert orch._config is pipeline_config


# ---------------------------------------------------------------------------
# Phase invocation without GPU
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_each_phase_can_be_invoked(
    settings: Settings,
    pipeline_config: TrainingPipelineConfig,
    gpu_monitor_mock: AsyncMock,
    batch_tuner_mock: MagicMock,
) -> None:
    """Each training phase (rssm, warmstart, bdi, constitutional_rl) runs without GPU."""
    orch = PipelineOrchestrator(
        settings=settings,
        pipeline_config=pipeline_config,
        gpu_monitor=gpu_monitor_mock,
        batch_tuner=batch_tuner_mock,
    )
    for phase in ["rssm", "warmstart", "bdi", "constitutional_rl"]:
        await orch._run_phase(phase, batch_size=8)
        checkpoint = Path(pipeline_config.checkpoint_dir) / f"{phase}.done"
        assert checkpoint.exists(), f"Checkpoint missing for phase {phase}"


# ---------------------------------------------------------------------------
# Resume flag skips completed phases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_skips_completed_phases(
    settings: Settings,
    gpu_monitor_mock: AsyncMock,
    batch_tuner_mock: MagicMock,
    tmp_path: Path,
) -> None:
    """--resume flag causes completed phases to be skipped."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    # Mark rssm and warmstart as done.
    (checkpoint_dir / "rssm.done").write_text("phase=rssm\n")
    (checkpoint_dir / "warmstart.done").write_text("phase=warmstart\n")

    config = TrainingPipelineConfig(
        checkpoint_dir=str(checkpoint_dir),
        resume_from_phase="bdi",
        thermal_pause_seconds=0.01,
    )
    orch = PipelineOrchestrator(
        settings=settings,
        pipeline_config=config,
        gpu_monitor=gpu_monitor_mock,
        batch_tuner=batch_tuner_mock,
    )

    phases_run: list[str] = []

    async def track(phase: str, batch_size: int) -> None:
        phases_run.append(phase)
        (checkpoint_dir / f"{phase}.done").write_text(f"phase={phase}\n")

    with patch.object(orch, "_run_phase", side_effect=track):
        await orch.run()

    assert "rssm" not in phases_run
    assert "warmstart" not in phases_run
    assert phases_run == ["bdi", "constitutional_rl"]


# ---------------------------------------------------------------------------
# Thermal pause
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thermal_pause_triggers_correctly(
    settings: Settings,
    batch_tuner_mock: MagicMock,
    tmp_path: Path,
) -> None:
    """Thermal pause should block until GPU cools below limit."""
    config = TrainingPipelineConfig(
        phases=["rssm"],
        checkpoint_dir=str(tmp_path / "checkpoints"),
        thermal_pause_seconds=0.01,
    )
    gpu_mock = AsyncMock()
    # First two checks hot, then cool.
    gpu_mock.should_pause.side_effect = [True, True, False]

    orch = PipelineOrchestrator(
        settings=settings,
        pipeline_config=config,
        gpu_monitor=gpu_mock,
        batch_tuner=batch_tuner_mock,
    )
    await orch.run()

    assert gpu_mock.should_pause.call_count == 3


# ---------------------------------------------------------------------------
# Batch size auto-tuning with mocked VRAM
# ---------------------------------------------------------------------------


def test_batch_tuner_scales_down_with_low_vram() -> None:
    """Batch tuner reduces batch size when VRAM is limited."""
    config = TrainingPipelineConfig(vram_headroom_mb=512)
    tuner = VRAMBatchTuner(config)

    # Mock torch.cuda to report constrained VRAM: 1024 free, 2048 total.
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.mem_get_info.return_value = (
        1024 * 1024 * 1024,  # 1024 MB free
        2048 * 1024 * 1024,  # 2048 MB total
    )

    with patch.dict("sys.modules", {"torch": mock_torch}):
        tuned = tuner.tune_batch_size("rssm", base_size=32)

    # With 1024 free - 512 headroom = 512 usable out of 2048 total => ratio ~0.25
    # tuned = max(1, min(32, int(32 * 0.25))) = 8
    assert 1 <= tuned <= 32
    assert tuned < 32  # Should be scaled down.


def test_batch_tuner_no_cuda_returns_base() -> None:
    """Without CUDA, batch tuner returns base size unchanged."""
    config = TrainingPipelineConfig(vram_headroom_mb=512)
    tuner = VRAMBatchTuner(config)

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": mock_torch}):
        tuned = tuner.tune_batch_size("rssm", base_size=64)

    assert tuned == 64


# ---------------------------------------------------------------------------
# Checkpoint directory creation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_checkpoint_directory_created(
    settings: Settings,
    gpu_monitor_mock: AsyncMock,
    batch_tuner_mock: MagicMock,
    tmp_path: Path,
) -> None:
    """Running a phase creates the checkpoint directory if it does not exist."""
    deep_path = tmp_path / "nested" / "deep" / "checkpoints"
    config = TrainingPipelineConfig(
        checkpoint_dir=str(deep_path),
        thermal_pause_seconds=0.01,
    )
    orch = PipelineOrchestrator(
        settings=settings,
        pipeline_config=config,
        gpu_monitor=gpu_monitor_mock,
        batch_tuner=batch_tuner_mock,
    )
    await orch._run_phase("rssm", 4)
    assert deep_path.exists()
    assert (deep_path / "rssm.done").exists()


# ---------------------------------------------------------------------------
# Config loading from local_training.yaml
# ---------------------------------------------------------------------------


def test_config_loading_from_yaml(tmp_path: Path) -> None:
    """Settings load correctly from a training YAML config."""
    config_data = {
        "mock_hardware": True,
        "training_pipeline": {
            "phases": ["rssm", "warmstart", "bdi", "constitutional_rl"],
            "batch_sizes": {"rssm": 32, "warmstart": 64, "bdi": 64, "constitutional_rl": 128},
            "thermal_limit_celsius": 85.0,
            "checkpoint_dir": str(tmp_path / "cp"),
            "amp_enabled": True,
        },
    }
    yaml_path = tmp_path / "test_training.yaml"
    yaml_path.write_text(yaml.dump(config_data))

    settings = _load_settings(str(yaml_path))
    assert settings.training_pipeline is not None
    assert settings.training_pipeline.phases == [
        "rssm",
        "warmstart",
        "bdi",
        "constitutional_rl",
    ]
    assert settings.training_pipeline.batch_sizes["rssm"] == 32
    assert settings.training_pipeline.amp_enabled is True


def test_local_training_yaml_is_loadable() -> None:
    """The real local_training.yaml file loads without error."""
    config_path = Path(__file__).resolve().parents[2] / "config" / "local_training.yaml"
    if not config_path.exists():
        pytest.skip("local_training.yaml not available in worktree")
    settings = _load_settings(str(config_path))
    assert settings.training_pipeline is not None
    assert len(settings.training_pipeline.phases) >= 1
    assert settings.training.batch_size == 32


# ---------------------------------------------------------------------------
# GPU monitor with mocked sysfs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gpu_monitor_reads_sysfs_temperature(tmp_path: Path) -> None:
    """GPU monitor reads temperature from mocked sysfs file."""
    sysfs = tmp_path / "thermal_zone0" / "temp"
    sysfs.parent.mkdir(parents=True)
    sysfs.write_text("72000")  # 72.0 degrees C in millidegrees.

    config = TrainingPipelineConfig(thermal_sysfs_path=str(sysfs))
    monitor = JetsonGPUMonitor(config)

    temp = await monitor.get_temperature()
    assert temp == pytest.approx(72.0)


@pytest.mark.asyncio
async def test_gpu_monitor_should_pause_when_hot(tmp_path: Path) -> None:
    """GPU monitor returns should_pause=True when over thermal limit."""
    sysfs = tmp_path / "thermal_zone0" / "temp"
    sysfs.parent.mkdir(parents=True)
    sysfs.write_text("90000")  # 90 degrees — over default 85 limit.

    config = TrainingPipelineConfig(
        thermal_sysfs_path=str(sysfs),
        thermal_limit_celsius=85.0,
    )
    monitor = JetsonGPUMonitor(config)
    assert await monitor.should_pause() is True


@pytest.mark.asyncio
async def test_gpu_monitor_should_not_pause_when_cool(tmp_path: Path) -> None:
    """GPU monitor returns should_pause=False when below thermal limit."""
    sysfs = tmp_path / "thermal_zone0" / "temp"
    sysfs.parent.mkdir(parents=True)
    sysfs.write_text("50000")  # 50 degrees — well under limit.

    config = TrainingPipelineConfig(
        thermal_sysfs_path=str(sysfs),
        thermal_limit_celsius=85.0,
    )
    monitor = JetsonGPUMonitor(config)
    assert await monitor.should_pause() is False


# ---------------------------------------------------------------------------
# End-to-end async_main
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_main_single_phase(tmp_path: Path) -> None:
    """async_main runs a single-phase pipeline to completion."""
    config_data = {
        "mock_hardware": True,
        "training_pipeline": {
            "phases": ["rssm"],
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "thermal_sysfs_path": str(tmp_path / "nonexistent_thermal"),
        },
    }
    yaml_path = tmp_path / "smoke_config.yaml"
    yaml_path.write_text(yaml.dump(config_data))

    await async_main(str(yaml_path), resume=False)
    assert (tmp_path / "checkpoints" / "rssm.done").exists()


@pytest.mark.asyncio
async def test_async_main_resume_auto_detect(tmp_path: Path) -> None:
    """async_main with resume=True auto-detects last completed phase."""
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
    yaml_path = tmp_path / "smoke_resume.yaml"
    yaml_path.write_text(yaml.dump(config_data))

    await async_main(str(yaml_path), resume=True)
    assert (checkpoint_dir / "warmstart.done").exists()
