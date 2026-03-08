"""Tests for PowerProfiler."""

from __future__ import annotations

from mousedroid.config.schema import JetsonConfig
from mousedroid.efficiency.profiler import PowerProfiler


def test_constructor():
    cfg = JetsonConfig()
    profiler = PowerProfiler(cfg)
    assert profiler._thermal_path == cfg.thermal_zone_path
    assert profiler._gpu_load_path == cfg.gpu_load_path


async def test_read_gpu_temp_missing_sysfs():
    cfg = JetsonConfig(
        thermal_zone_path="/nonexistent/path/temp",
    )
    profiler = PowerProfiler(cfg)
    temp = await profiler.read_gpu_temp_c()
    assert temp == 0.0


async def test_read_gpu_load_missing_sysfs():
    cfg = JetsonConfig(
        gpu_load_path="/nonexistent/path/load",
    )
    profiler = PowerProfiler(cfg)
    load = await profiler.read_gpu_load_pct()
    assert load == 0.0


async def test_read_gpu_temp_with_valid_file(tmp_path):
    # Write a fake sysfs temp file (45000 millidegrees = 45.0 C)
    temp_file = tmp_path / "temp"
    temp_file.write_text("45000\n")
    cfg = JetsonConfig(thermal_zone_path=temp_file)
    profiler = PowerProfiler(cfg)
    temp = await profiler.read_gpu_temp_c()
    assert temp == 45.0


async def test_read_gpu_load_with_valid_file(tmp_path):
    # Write a fake sysfs load file (750 / 10 = 75.0%)
    load_file = tmp_path / "load"
    load_file.write_text("750\n")
    cfg = JetsonConfig(gpu_load_path=load_file)
    profiler = PowerProfiler(cfg)
    load = await profiler.read_gpu_load_pct()
    assert load == 75.0
