"""Unit tests for training validation gates."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from mousedroid.config.schema import (
    Settings,
    TrainingPipelineConfig,
    TrainingValidationConfig,
)
from mousedroid.training.pipeline_orchestrator import (
    PipelineOrchestrator,
)
from mousedroid.training.validation import (
    validate_bdi_accuracy,
    validate_constitutional_rl,
    validate_rssm_convergence,
    validate_warmstart_policy,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings() -> Settings:
    """Create minimal Settings for testing."""
    return Settings(mock_hardware=True, ultrasonic=None)


@pytest.fixture
def gpu_monitor() -> AsyncMock:
    """Mock GPU monitor that never pauses."""
    mock = AsyncMock()
    mock.should_pause.return_value = False
    return mock


@pytest.fixture
def batch_tuner() -> MagicMock:
    """Mock batch tuner that returns base size."""
    mock = MagicMock()
    mock.tune_batch_size.side_effect = lambda phase, base: base
    return mock


# ---------------------------------------------------------------------------
# validate_rssm_convergence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rssm_validation_passes_below_threshold(tmp_path: Path) -> None:
    """RSSM validation passes when loss is below max_loss."""
    checkpoint = tmp_path / "rssm.pt"
    checkpoint.write_text("checkpoint")
    data = tmp_path / "rssm_data"
    data.mkdir()

    result = await validate_rssm_convergence(checkpoint, data, max_loss=0.5)
    assert result is True


@pytest.mark.asyncio
async def test_rssm_validation_fails_above_threshold(tmp_path: Path) -> None:
    """RSSM validation fails when loss exceeds max_loss."""
    checkpoint = tmp_path / "rssm.pt"
    checkpoint.write_text("checkpoint")
    data = tmp_path / "rssm_data"
    data.mkdir()

    # Stub returns 0.1; set threshold below that to trigger failure.
    result = await validate_rssm_convergence(checkpoint, data, max_loss=0.05)
    assert result is False


@pytest.mark.asyncio
async def test_rssm_validation_missing_checkpoint(tmp_path: Path) -> None:
    """RSSM validation returns False for missing checkpoint (no crash)."""
    data = tmp_path / "rssm_data"
    data.mkdir()

    result = await validate_rssm_convergence(tmp_path / "nonexistent.pt", data, max_loss=0.5)
    assert result is False


@pytest.mark.asyncio
async def test_rssm_validation_missing_data(tmp_path: Path) -> None:
    """RSSM validation returns False for missing data directory."""
    checkpoint = tmp_path / "rssm.pt"
    checkpoint.write_text("checkpoint")

    result = await validate_rssm_convergence(
        checkpoint, tmp_path / "nonexistent_data", max_loss=0.5
    )
    assert result is False


# ---------------------------------------------------------------------------
# validate_warmstart_policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warmstart_validation_passes(tmp_path: Path) -> None:
    """Warm-start validation passes when reward exceeds minimum."""
    checkpoint = tmp_path / "warmstart.pt"
    checkpoint.write_text("checkpoint")

    # Stub returns 5.0; min_reward=-10 => passes.
    result = await validate_warmstart_policy(checkpoint, min_reward=-10.0)
    assert result is True


@pytest.mark.asyncio
async def test_warmstart_validation_fails(tmp_path: Path) -> None:
    """Warm-start validation fails when reward is below minimum."""
    checkpoint = tmp_path / "warmstart.pt"
    checkpoint.write_text("checkpoint")

    # Stub returns 5.0; min_reward=100 => fails.
    result = await validate_warmstart_policy(checkpoint, min_reward=100.0)
    assert result is False


@pytest.mark.asyncio
async def test_warmstart_validation_missing_checkpoint(tmp_path: Path) -> None:
    """Warm-start validation returns False for missing checkpoint."""
    result = await validate_warmstart_policy(tmp_path / "nonexistent.pt", min_reward=-10.0)
    assert result is False


# ---------------------------------------------------------------------------
# validate_bdi_accuracy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bdi_validation_passes(tmp_path: Path) -> None:
    """BDI validation passes when accuracy exceeds minimum."""
    weights = tmp_path / "bdi"
    weights.mkdir()
    data = tmp_path / "bdi_data"
    data.mkdir()

    # Stub returns 0.6; min=0.3 => passes.
    result = await validate_bdi_accuracy(weights, data, min_accuracy=0.3)
    assert result is True


@pytest.mark.asyncio
async def test_bdi_validation_fails(tmp_path: Path) -> None:
    """BDI validation fails when accuracy is below minimum."""
    weights = tmp_path / "bdi"
    weights.mkdir()
    data = tmp_path / "bdi_data"
    data.mkdir()

    # Stub returns 0.6; min=0.9 => fails.
    result = await validate_bdi_accuracy(weights, data, min_accuracy=0.9)
    assert result is False


@pytest.mark.asyncio
async def test_bdi_validation_missing_weights(tmp_path: Path) -> None:
    """BDI validation returns False for missing weights directory."""
    data = tmp_path / "bdi_data"
    data.mkdir()

    result = await validate_bdi_accuracy(tmp_path / "nonexistent", data, min_accuracy=0.3)
    assert result is False


# ---------------------------------------------------------------------------
# validate_constitutional_rl
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_constitutional_validation_passes(tmp_path: Path) -> None:
    """Constitutional RL validation passes when both thresholds met."""
    output = tmp_path / "constitutional_rl"
    output.mkdir()

    # Stub returns (0.02, 2.0); max_violation=0.1, min_reward=-5 => passes.
    result = await validate_constitutional_rl(output, max_violation_rate=0.1, min_reward=-5.0)
    assert result is True


@pytest.mark.asyncio
async def test_constitutional_validation_fails_violation_rate(tmp_path: Path) -> None:
    """Constitutional RL validation fails when violation rate too high."""
    output = tmp_path / "constitutional_rl"
    output.mkdir()

    # Stub returns violation_rate=0.02; set threshold below that.
    result = await validate_constitutional_rl(output, max_violation_rate=0.01, min_reward=-5.0)
    assert result is False


@pytest.mark.asyncio
async def test_constitutional_validation_missing_output(tmp_path: Path) -> None:
    """Constitutional RL validation returns False for missing directory."""
    result = await validate_constitutional_rl(
        tmp_path / "nonexistent", max_violation_rate=0.1, min_reward=-5.0
    )
    assert result is False


# ---------------------------------------------------------------------------
# Pipeline orchestrator — validation gate integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_halts_on_validation_failure(
    settings: Settings,
    gpu_monitor: AsyncMock,
    batch_tuner: MagicMock,
    tmp_path: Path,
) -> None:
    """Pipeline halts when a validation gate returns False."""
    config = TrainingPipelineConfig(
        phases=["rssm"],
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    validation = TrainingValidationConfig(enabled=True)

    orchestrator = PipelineOrchestrator(
        settings=settings,
        pipeline_config=config,
        gpu_monitor=gpu_monitor,
        batch_tuner=batch_tuner,
        validation_config=validation,
    )

    with (
        patch(
            "mousedroid.training.pipeline_orchestrator.validate_rssm_convergence",
            new_callable=AsyncMock,
            return_value=False,
        ),
        pytest.raises(RuntimeError, match="Validation gate failed"),
    ):
        await orchestrator.run()


@pytest.mark.asyncio
async def test_pipeline_continues_on_validation_success(
    settings: Settings,
    gpu_monitor: AsyncMock,
    batch_tuner: MagicMock,
    tmp_path: Path,
) -> None:
    """Pipeline continues to next phase when validation passes."""
    config = TrainingPipelineConfig(
        phases=["rssm", "warmstart"],
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    validation = TrainingValidationConfig(enabled=True)

    orchestrator = PipelineOrchestrator(
        settings=settings,
        pipeline_config=config,
        gpu_monitor=gpu_monitor,
        batch_tuner=batch_tuner,
        validation_config=validation,
    )

    with (
        patch(
            "mousedroid.training.pipeline_orchestrator.validate_rssm_convergence",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "mousedroid.training.pipeline_orchestrator.validate_warmstart_policy",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        await orchestrator.run()

    # Both checkpoints should exist.
    assert (tmp_path / "checkpoints" / "rssm.done").exists()
    assert (tmp_path / "checkpoints" / "warmstart.done").exists()


@pytest.mark.asyncio
async def test_validate_only_skips_training(
    settings: Settings,
    gpu_monitor: AsyncMock,
    batch_tuner: MagicMock,
    tmp_path: Path,
) -> None:
    """validate_only=True skips training, only validates checkpoints."""
    config = TrainingPipelineConfig(
        phases=["rssm"],
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    validation = TrainingValidationConfig(enabled=True, validate_only=True)

    orchestrator = PipelineOrchestrator(
        settings=settings,
        pipeline_config=config,
        gpu_monitor=gpu_monitor,
        batch_tuner=batch_tuner,
        validation_config=validation,
    )

    with patch(
        "mousedroid.training.pipeline_orchestrator.validate_rssm_convergence",
        new_callable=AsyncMock,
        return_value=True,
    ):
        await orchestrator.run()

    # No checkpoint should be written — training was skipped.
    assert not (tmp_path / "checkpoints" / "rssm.done").exists()


@pytest.mark.asyncio
async def test_config_driven_thresholds(tmp_path: Path) -> None:
    """Validation thresholds from config are passed to validators."""
    config = TrainingPipelineConfig(
        phases=["rssm"],
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    validation = TrainingValidationConfig(
        enabled=True,
        rssm_max_loss=0.001,  # Very tight threshold — will fail with stub.
    )
    settings = Settings(mock_hardware=True, ultrasonic=None)

    orchestrator = PipelineOrchestrator(
        settings=settings,
        pipeline_config=config,
        gpu_monitor=AsyncMock(should_pause=AsyncMock(return_value=False)),
        batch_tuner=MagicMock(tune_batch_size=MagicMock(side_effect=lambda p, b: b)),
        validation_config=validation,
    )

    # Stub _compute_rssm_loss returns 0.1 which exceeds 0.001 threshold.
    with pytest.raises(RuntimeError, match="Validation gate failed"):
        await orchestrator.run()


@pytest.mark.asyncio
async def test_pipeline_no_validation_when_disabled(
    settings: Settings,
    gpu_monitor: AsyncMock,
    batch_tuner: MagicMock,
    tmp_path: Path,
) -> None:
    """Pipeline skips validation when validation_config is None."""
    config = TrainingPipelineConfig(
        phases=["rssm"],
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    orchestrator = PipelineOrchestrator(
        settings=settings,
        pipeline_config=config,
        gpu_monitor=gpu_monitor,
        batch_tuner=batch_tuner,
        validation_config=None,
    )
    # Should complete without any validation.
    await orchestrator.run()
    assert (tmp_path / "checkpoints" / "rssm.done").exists()


# ---------------------------------------------------------------------------
# Config backwards compatibility
# ---------------------------------------------------------------------------


def test_training_validation_config_defaults() -> None:
    """TrainingValidationConfig has sensible defaults."""
    cfg = TrainingValidationConfig()
    assert cfg.enabled is False
    assert cfg.rssm_max_loss == 0.5
    assert cfg.warmstart_min_reward == -10.0
    assert cfg.bdi_min_accuracy == 0.3
    assert cfg.constitutional_max_violation_rate == 0.1
    assert cfg.constitutional_min_reward == -5.0
    assert cfg.validation_data_dir == "training/validation_data"
    assert cfg.validate_only is False


def test_settings_loads_without_training_validation() -> None:
    """Settings loads correctly when training_validation is absent."""
    settings = Settings(mock_hardware=True, ultrasonic=None)
    assert settings.training_validation is None


def test_settings_loads_with_training_validation() -> None:
    """Settings loads correctly with training_validation present."""
    settings = Settings(
        mock_hardware=True,
        ultrasonic=None,
        training_validation=TrainingValidationConfig(enabled=True),
    )
    assert settings.training_validation is not None
    assert settings.training_validation.enabled is True


def test_existing_yaml_loads_unchanged(tmp_path: Path) -> None:
    """Existing YAML without training_validation still loads correctly."""
    config_data = {
        "mock_hardware": True,
        "training_pipeline": {
            "phases": ["rssm"],
            "checkpoint_dir": str(tmp_path / "cp"),
        },
    }
    yaml_path = tmp_path / "test_config.yaml"
    yaml_path.write_text(yaml.dump(config_data))

    settings = Settings(**yaml.safe_load(yaml_path.read_text()))
    assert settings.training_validation is None
    assert settings.training_pipeline is not None
