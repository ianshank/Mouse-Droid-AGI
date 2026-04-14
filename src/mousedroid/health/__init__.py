"""Health monitoring and watchdog for Jetson hardware."""

from mousedroid.health.monitor import HealthMonitor
from mousedroid.health.watchdog import (
    FileHeartbeatNotifier,
    NullNotifier,
    SystemdNotifier,
    WatchdogProtocol,
)

__all__ = [
    "FileHeartbeatNotifier",
    "HealthMonitor",
    "NullNotifier",
    "SystemdNotifier",
    "WatchdogProtocol",
]
