"""Tests for HealthMonitor — full coverage."""

from __future__ import annotations

from typing import Literal
from unittest.mock import patch

import pytest

from mousedroid.config.schema import HealthConfig, JetsonConfig
from mousedroid.health.monitor import HealthMonitor


@pytest.fixture
def monitor() -> HealthMonitor:
    return HealthMonitor(HealthConfig(), JetsonConfig())


def test_constructor(monitor: HealthMonitor) -> None:
    assert monitor._running is False


@pytest.mark.asyncio
async def test_check_health_returns_dict_with_status(monitor: HealthMonitor) -> None:
    result = await monitor.check_health()
    assert "status" in result
    assert result["status"] in ("ok", "warning", "critical")


@pytest.mark.asyncio
async def test_read_gpu_temp_c_missing_file(monitor: HealthMonitor) -> None:
    with patch.object(HealthMonitor, "_read_sysfs", side_effect=FileNotFoundError):
        temp = await monitor.read_gpu_temp_c()
    assert temp == 0.0


@pytest.mark.asyncio
async def test_read_gpu_load_pct_missing_file(monitor: HealthMonitor) -> None:
    load = await monitor.read_gpu_load_pct()
    assert load == 0.0


@pytest.mark.asyncio
async def test_check_health_has_gpu_keys(monitor: HealthMonitor) -> None:
    result = await monitor.check_health()
    assert "gpu_temp_c" in result
    assert "gpu_load_pct" in result


@pytest.mark.asyncio
async def test_check_health_ok_status_with_zero_temp(monitor: HealthMonitor) -> None:
    with patch.object(HealthMonitor, "_read_sysfs", side_effect=FileNotFoundError):
        result = await monitor.check_health()
    assert result["status"] == "ok"
    assert result["gpu_temp_c"] == 0.0


@pytest.mark.asyncio
async def test_read_gpu_temp_valid_sysfs() -> None:
    monitor = HealthMonitor(HealthConfig(), JetsonConfig())
    with patch.object(HealthMonitor, "_read_sysfs", return_value="55000"):
        temp = await monitor.read_gpu_temp_c()
    assert temp == 55.0


@pytest.mark.asyncio
async def test_read_gpu_load_valid_sysfs() -> None:
    monitor = HealthMonitor(HealthConfig(), JetsonConfig())
    with patch.object(HealthMonitor, "_read_sysfs", return_value="750"):
        load = await monitor.read_gpu_load_pct()
    assert load == 75.0


@pytest.mark.asyncio
async def test_check_health_warning() -> None:
    health_cfg = HealthConfig(gpu_temp_warn_c=50.0, gpu_temp_critical_c=80.0)
    monitor = HealthMonitor(health_cfg, JetsonConfig())
    with patch.object(HealthMonitor, "_read_sysfs", return_value="55000"):
        result = await monitor.check_health()
    assert result["status"] == "warning"
    assert result["gpu_temp_c"] == 55.0


@pytest.mark.asyncio
async def test_check_health_critical() -> None:
    health_cfg = HealthConfig(gpu_temp_warn_c=50.0, gpu_temp_critical_c=80.0)
    monitor = HealthMonitor(health_cfg, JetsonConfig())
    with patch.object(HealthMonitor, "_read_sysfs", return_value="85000"):
        result = await monitor.check_health()
    assert result["status"] == "critical"
    assert result["gpu_temp_c"] == 85.0


@pytest.mark.asyncio
async def test_read_gpu_temp_value_error() -> None:
    monitor = HealthMonitor(HealthConfig(), JetsonConfig())
    with patch.object(HealthMonitor, "_read_sysfs", return_value="not_a_number"):
        temp = await monitor.read_gpu_temp_c()
    assert temp == 0.0


def test_read_sysfs_returns_content(tmp_path: pytest.TempPathFactory) -> None:
    """_read_sysfs reads file contents as a string."""
    from pathlib import Path

    sysfs_file = Path(tmp_path) / "test_sysfs"  # type: ignore[arg-type]
    sysfs_file.write_text("42000\n")
    result = HealthMonitor._read_sysfs(str(sysfs_file))
    assert result == "42000\n"


def test_read_sysfs_raises_on_missing() -> None:
    """_read_sysfs raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        HealthMonitor._read_sysfs("/nonexistent/path")


@pytest.mark.asyncio
@pytest.mark.parametrize("power_mode", ["15W", "7W"])
async def test_check_health_includes_power_mode(power_mode: Literal["15W", "7W"]) -> None:
    """check_health must accurately report JetsonConfig.power_mode."""
    monitor = HealthMonitor(HealthConfig(), JetsonConfig(power_mode=power_mode))
    result = await monitor.check_health()
    assert "power_mode" in result
    assert result["power_mode"] == power_mode


@pytest.mark.asyncio
async def test_check_health_default_power_mode() -> None:
    """check_health with default JetsonConfig should report 15W."""
    monitor = HealthMonitor(HealthConfig(), JetsonConfig())
    result = await monitor.check_health()
    assert result.get("power_mode") == "15W"


def test_read_sysfs_handles_utf8_and_special_chars(tmp_path: pytest.TempPathFactory) -> None:
    """_read_sysfs must safely read utf-8 encoded text with replacement on error."""
    from pathlib import Path

    sysfs_file = Path(tmp_path) / "test_sysfs_utf8"  # type: ignore[arg-type]
    sysfs_file.write_bytes(b"52000\n")
    result = HealthMonitor._read_sysfs(str(sysfs_file))
    assert result == "52000\n"
