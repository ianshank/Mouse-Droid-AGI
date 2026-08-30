"""Factory builders — health monitor and watchdog."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.health.watchdog import WatchdogProtocol
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import (
        Settings,
    )
    from mousedroid.health.monitor import HealthMonitor

_log = get_logger(__name__)


def build_health_monitor(cfg: Settings) -> HealthMonitor:
    """Build health monitor for GPU/thermal monitoring.

    Args:
        cfg: Root settings.

    Returns:
        Configured ``HealthMonitor``.
    """
    from mousedroid.health.monitor import HealthMonitor

    _log.info("health_monitor_built")
    return HealthMonitor(cfg.health, cfg.jetson)


def build_watchdog(cfg: Settings) -> WatchdogProtocol:
    """Build watchdog notifier based on config.

    Returns :class:`SystemdNotifier` when the ``NOTIFY_SOCKET`` env var is
    present (set automatically by systemd for ``Type=notify`` services),
    :class:`FileHeartbeatNotifier` for Docker/custom monitoring, or
    :class:`NullNotifier` when watchdog is disabled.

    Args:
        cfg: Root settings.

    Returns:
        Watchdog notifier satisfying :class:`WatchdogProtocol`.
    """
    import os
    from pathlib import Path

    from mousedroid.health.watchdog import (
        FileHeartbeatNotifier,
        NullNotifier,
        SystemdNotifier,
    )

    if not cfg.loop.watchdog_enabled:
        return NullNotifier()

    mode = cfg.loop.watchdog_mode
    if mode == "none":
        return NullNotifier()
    if mode == "systemd":
        return SystemdNotifier()
    if mode == "file":
        return FileHeartbeatNotifier(Path(cfg.loop.watchdog_heartbeat_path))
    if mode == "auto":
        if os.environ.get("NOTIFY_SOCKET"):
            return SystemdNotifier()
        return FileHeartbeatNotifier(Path(cfg.loop.watchdog_heartbeat_path))

    _log.warning("unknown_watchdog_mode_falling_back_to_null", mode=mode)
    return NullNotifier()
