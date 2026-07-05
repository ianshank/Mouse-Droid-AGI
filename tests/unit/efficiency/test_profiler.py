"""Unit tests for the Jetson power/thermal profiler.

Points the sysfs paths at temp files so the read paths exercise real I/O, and
verifies the graceful ``0.0`` fallback when a sysfs node is absent or malformed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.schema import JetsonConfig
from mousedroid.constants import GPU_LOAD_PERCENTAGE_DIVISOR, MILLIDEGREE_DIVISOR
from mousedroid.efficiency.profiler import PowerProfiler


def _profiler(thermal: Path, load: Path) -> PowerProfiler:
    cfg = JetsonConfig(thermal_zone_path=str(thermal), gpu_load_path=str(load))
    return PowerProfiler(cfg)


async def test_read_gpu_temp_scales_millidegrees(tmp_path: Path) -> None:
    """A millidegree sysfs reading is scaled to Celsius."""
    thermal = tmp_path / "temp"
    thermal.write_text("52000\n")
    profiler = _profiler(thermal, tmp_path / "load")
    assert await profiler.read_gpu_temp_c() == 52000 / MILLIDEGREE_DIVISOR


async def test_read_gpu_load_scales(tmp_path: Path) -> None:
    """A raw load reading is scaled to a 0-100 percentage."""
    load = tmp_path / "load"
    load.write_text("850\n")
    profiler = _profiler(tmp_path / "temp", load)
    assert await profiler.read_gpu_load_pct() == 850 / GPU_LOAD_PERCENTAGE_DIVISOR


async def test_missing_thermal_node_returns_zero(tmp_path: Path) -> None:
    """A missing sysfs node degrades to 0.0 rather than raising."""
    profiler = _profiler(tmp_path / "does_not_exist", tmp_path / "nope")
    assert await profiler.read_gpu_temp_c() == 0.0
    assert await profiler.read_gpu_load_pct() == 0.0


async def test_malformed_reading_returns_zero(tmp_path: Path) -> None:
    """A non-numeric sysfs value degrades to 0.0 (ValueError swallowed)."""
    thermal = tmp_path / "temp"
    thermal.write_text("not-a-number")
    profiler = _profiler(thermal, tmp_path / "load")
    assert await profiler.read_gpu_temp_c() == 0.0


@pytest.mark.parametrize("raw", ["0", "1000", "37500"])
async def test_temp_reading_is_finite(tmp_path: Path, raw: str) -> None:
    """Well-formed readings always produce a finite float."""
    thermal = tmp_path / "temp"
    thermal.write_text(raw)
    profiler = _profiler(thermal, tmp_path / "load")
    result = await profiler.read_gpu_temp_c()
    assert result == float(raw) / MILLIDEGREE_DIVISOR
