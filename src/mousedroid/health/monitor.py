"""Health monitor — Jetson thermals, GPU, memory monitoring."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from mousedroid.constants import GPU_LOAD_PERCENTAGE_DIVISOR, MILLIDEGREE_DIVISOR
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import HealthConfig, JetsonConfig

_log = get_logger(__name__)

_MILLIDEGREE_DIVISOR: float = MILLIDEGREE_DIVISOR
_LOAD_PERCENTAGE_DIVISOR: float = GPU_LOAD_PERCENTAGE_DIVISOR


class HealthMonitor:
    """Monitor Jetson health metrics (temperature, GPU load, memory).

    Reads from sysfs paths configured in JetsonConfig.
    """

    def __init__(self, health_cfg: HealthConfig, jetson_cfg: JetsonConfig) -> None:
        """Initialise health monitor.

        Args:
            health_cfg: Health monitoring configuration.
            jetson_cfg: Jetson hardware configuration.
        """
        self._health_cfg = health_cfg
        self._jetson_cfg = jetson_cfg
        self._running = False

    async def read_gpu_temp_c(self) -> float:
        """Read GPU temperature from sysfs.

        Returns:
            GPU temperature in Celsius.
        """
        try:
            raw = await asyncio.to_thread(
                self._read_sysfs,
                str(self._jetson_cfg.thermal_zone_path),
            )
            return float(raw.strip()) / _MILLIDEGREE_DIVISOR
        except (FileNotFoundError, ValueError):
            _log.debug("gpu_temp_read_failed")
            return 0.0

    async def read_gpu_load_pct(self) -> float:
        """Read GPU utilisation percentage from sysfs.

        Returns:
            GPU load as percentage (0-100).
        """
        try:
            raw = await asyncio.to_thread(
                self._read_sysfs,
                str(self._jetson_cfg.gpu_load_path),
            )
            return float(raw.strip()) / _LOAD_PERCENTAGE_DIVISOR
        except (FileNotFoundError, ValueError):
            _log.debug("gpu_load_read_failed")
            return 0.0

    async def check_health(self) -> dict[str, object]:
        """Run comprehensive health check.

        Returns:
            Dict with health metrics.
        """
        gpu_temp = await self.read_gpu_temp_c()
        gpu_load = await self.read_gpu_load_pct()

        status = "ok"
        if gpu_temp >= self._health_cfg.gpu_temp_critical_c:
            status = "critical"
            _log.error("gpu_temp_critical", temp_c=gpu_temp)
        elif gpu_temp >= self._health_cfg.gpu_temp_warn_c:
            status = "warning"
            _log.warning("gpu_temp_warning", temp_c=gpu_temp)

        return {
            "status": status,
            "gpu_temp_c": gpu_temp,
            "gpu_load_pct": gpu_load,
            "power_mode": self._jetson_cfg.power_mode,
        }

    @staticmethod
    def _read_sysfs(path: str) -> str:
        """Read a sysfs file (blocking).

        Args:
            path: Sysfs file path.

        Returns:
            File contents as string.
        """
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
