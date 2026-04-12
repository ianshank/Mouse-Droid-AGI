"""GPU thermal and VRAM monitoring for training pipeline.

Reads Jetson sysfs thermal zone and torch.cuda VRAM stats to decide
whether training should pause (thermal) or reduce batch size (VRAM).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol, runtime_checkable

import structlog

from mousedroid.config.schema import TrainingPipelineConfig

logger = structlog.get_logger(__name__)


@runtime_checkable
class GPUMonitorProtocol(Protocol):
    """Protocol for GPU health monitoring during training."""

    async def get_temperature(self) -> float:
        """Return current GPU temperature in Celsius."""
        ...

    async def get_vram_free_mb(self) -> int:
        """Return free VRAM in megabytes."""
        ...

    async def should_pause(self) -> bool:
        """Return True if training should pause due to thermal limits."""
        ...


class JetsonGPUMonitor:
    """Concrete GPU monitor reading Jetson sysfs thermal zone and torch.cuda VRAM.

    Args:
        config: Training pipeline configuration with thermal thresholds.
    """

    def __init__(self, config: TrainingPipelineConfig) -> None:
        self._thermal_limit = config.thermal_limit_celsius
        self._thermal_pause_s = config.thermal_pause_seconds
        self._sysfs_path = Path(config.thermal_sysfs_path)
        self._vram_headroom_mb = config.vram_headroom_mb

    async def get_temperature(self) -> float:
        """Read GPU temperature from sysfs (millidegrees -> Celsius).

        Returns:
            Temperature in Celsius. Returns 0.0 if sysfs is unavailable.
        """
        try:
            raw = await asyncio.to_thread(self._read_sysfs_temp)
            temp_c = raw / 1000.0
            return temp_c
        except (OSError, ValueError) as exc:
            logger.warning(
                "gpu_temperature_read_failed",
                path=str(self._sysfs_path),
                error=str(exc),
            )
            return 0.0

    async def get_vram_free_mb(self) -> int:
        """Query free VRAM via torch.cuda.mem_get_info.

        Returns:
            Free VRAM in MB. Returns 0 if CUDA is unavailable.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return 0
            free, _total = torch.cuda.mem_get_info()
            return int(free / (1024 * 1024))
        except Exception as exc:
            logger.warning("vram_query_failed", error=str(exc))
            return 0

    async def should_pause(self) -> bool:
        """Check if temperature exceeds thermal limit.

        Returns:
            True if training should pause to cool down.
        """
        temp = await self.get_temperature()
        if temp >= self._thermal_limit:
            logger.warning(
                "thermal_pause_triggered",
                temperature_c=temp,
                limit_c=self._thermal_limit,
                pause_seconds=self._thermal_pause_s,
            )
            return True
        return False

    def _read_sysfs_temp(self) -> float:
        """Synchronous sysfs read (run in thread pool)."""
        return float(self._sysfs_path.read_text().strip())
