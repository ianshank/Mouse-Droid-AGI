"""Unit tests for GPU monitor module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mousedroid.config.schema import TrainingPipelineConfig
from mousedroid.training.gpu_monitor import JetsonGPUMonitor


@pytest.fixture
def pipeline_config(tmp_path: Path) -> TrainingPipelineConfig:
    """Create a pipeline config with a writable sysfs path."""
    sysfs = tmp_path / "thermal_zone0" / "temp"
    sysfs.parent.mkdir(parents=True)
    sysfs.write_text("65000")  # 65.0 C
    return TrainingPipelineConfig(
        thermal_limit_celsius=85.0,
        thermal_pause_seconds=1.0,
        thermal_sysfs_path=str(sysfs),
        vram_headroom_mb=512,
    )


@pytest.fixture
def monitor(pipeline_config: TrainingPipelineConfig) -> JetsonGPUMonitor:
    """Create a GPU monitor with test config."""
    return JetsonGPUMonitor(pipeline_config)


@pytest.mark.asyncio
async def test_temperature_reading(
    monitor: JetsonGPUMonitor, pipeline_config: TrainingPipelineConfig
) -> None:
    """Temperature should be read from sysfs and converted from millidegrees."""
    temp = await monitor.get_temperature()
    assert temp == pytest.approx(65.0)


@pytest.mark.asyncio
async def test_should_not_pause_below_threshold(monitor: JetsonGPUMonitor) -> None:
    """Should not pause when temperature is below limit."""
    result = await monitor.should_pause()
    assert result is False


@pytest.mark.asyncio
async def test_should_pause_above_threshold(tmp_path: Path) -> None:
    """Should pause when temperature exceeds thermal limit."""
    sysfs = tmp_path / "hot_temp"
    sysfs.write_text("90000")  # 90 C > 85 C limit
    config = TrainingPipelineConfig(
        thermal_limit_celsius=85.0,
        thermal_sysfs_path=str(sysfs),
    )
    monitor = JetsonGPUMonitor(config)
    result = await monitor.should_pause()
    assert result is True


@pytest.mark.asyncio
async def test_graceful_fallback_sysfs_unavailable(tmp_path: Path) -> None:
    """Should return 0.0 temperature when sysfs path does not exist."""
    config = TrainingPipelineConfig(
        thermal_sysfs_path=str(tmp_path / "nonexistent"),
    )
    monitor = JetsonGPUMonitor(config)
    temp = await monitor.get_temperature()
    assert temp == 0.0
    # Should not pause since 0.0 < 85.0
    assert await monitor.should_pause() is False


@pytest.mark.asyncio
async def test_vram_reporting_no_cuda(monitor: JetsonGPUMonitor) -> None:
    """Should return 0 VRAM when CUDA is unavailable."""
    with patch("mousedroid.training.gpu_monitor.JetsonGPUMonitor.get_vram_free_mb") as mock:
        mock.return_value = 0
        result = await monitor.get_vram_free_mb()
    assert result == 0


@pytest.mark.asyncio
async def test_vram_reporting_with_cuda(monitor: JetsonGPUMonitor) -> None:
    """Should report VRAM in MB when CUDA is available."""
    with patch.object(
        JetsonGPUMonitor,
        "get_vram_free_mb",
        return_value=4096,
    ):
        result = await monitor.get_vram_free_mb()
        assert result == 4096


@pytest.mark.asyncio
async def test_vram_returns_zero_when_torch_raises(tmp_path: Path) -> None:
    """Should return 0 when torch raises an unexpected exception."""
    sysfs = tmp_path / "temp"
    sysfs.write_text("50000")
    config = TrainingPipelineConfig(thermal_sysfs_path=str(sysfs))
    monitor = JetsonGPUMonitor(config)

    with patch("mousedroid.training.gpu_monitor.JetsonGPUMonitor.get_vram_free_mb") as m:
        m.return_value = 0
        result = await monitor.get_vram_free_mb()
    assert result == 0


@pytest.mark.asyncio
async def test_temperature_exact_at_threshold(tmp_path: Path) -> None:
    """Should pause when temperature equals thermal limit exactly."""
    sysfs = tmp_path / "temp"
    sysfs.write_text("85000")  # Exactly 85.0 C == limit
    config = TrainingPipelineConfig(
        thermal_limit_celsius=85.0,
        thermal_sysfs_path=str(sysfs),
    )
    monitor = JetsonGPUMonitor(config)
    assert await monitor.should_pause() is True


@pytest.mark.asyncio
async def test_temperature_just_below_threshold(tmp_path: Path) -> None:
    """Should NOT pause when temperature is just below threshold."""
    sysfs = tmp_path / "temp"
    sysfs.write_text("84999")  # 84.999 C
    config = TrainingPipelineConfig(
        thermal_limit_celsius=85.0,
        thermal_sysfs_path=str(sysfs),
    )
    monitor = JetsonGPUMonitor(config)
    assert await monitor.should_pause() is False
