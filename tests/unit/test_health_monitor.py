from __future__ import annotations

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
    result = await monitor.check_health()
    assert result["status"] == "ok"
    assert result["gpu_temp_c"] == 0.0
