"""Power and performance profiler for Jetson deployment (Pillar 10)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import JetsonConfig

_log = get_logger(__name__)

_MILLIDEGREE_DIVISOR: float = 1000.0
_LOAD_DIVISOR: float = 10.0


class PowerProfiler:
    """Monitor Jetson power consumption and thermal state.

    Reads from sysfs paths configured in ``JetsonConfig``.
    """

    def __init__(self, cfg: JetsonConfig) -> None:
        """Initialise profiler.

        Args:
            cfg: Jetson hardware configuration.
        """
        self._thermal_path = cfg.thermal_zone_path
        self._gpu_load_path = cfg.gpu_load_path

    async def read_gpu_temp_c(self) -> float:
        """Read GPU temperature from sysfs.

        Returns:
            GPU temperature in Celsius.
        """
        try:
            raw = await asyncio.to_thread(self._read_file, str(self._thermal_path))
            return float(raw.strip()) / _MILLIDEGREE_DIVISOR
        except (FileNotFoundError, ValueError):
            _log.debug("gpu_temp_unavailable")
            return 0.0

    async def read_gpu_load_pct(self) -> float:
        """Read GPU utilization percentage from sysfs.

        Returns:
            GPU load as percentage (0-100).
        """
        try:
            raw = await asyncio.to_thread(self._read_file, str(self._gpu_load_path))
            return float(raw.strip()) / _LOAD_DIVISOR
        except (FileNotFoundError, ValueError):
            _log.debug("gpu_load_unavailable")
            return 0.0

    @staticmethod
    def _read_file(path: str) -> str:
        """Read a file (blocking).

        Args:
            path: File path.

        Returns:
            File contents.
        """
        with open(path) as fh:
            return fh.read()
