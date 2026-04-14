"""Watchdog notifier — signals liveness to systemd or file-based health checks.

Provides protocol-based DI with three implementations:

- ``SystemdNotifier``: Sends ``WATCHDOG=1`` via the NOTIFY_SOCKET for systemd
  integration (requires ``Type=notify`` + ``WatchdogSec`` in the unit file).
- ``FileHeartbeatNotifier``: Touches a file on each call, suitable for Docker
  ``HEALTHCHECK`` or custom monitoring scripts.
- ``NullNotifier``: No-op for development / mock-hardware mode.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


@runtime_checkable
class WatchdogProtocol(Protocol):
    """Protocol for watchdog liveness notification."""

    def notify(self) -> None:
        """Signal liveness to the watchdog supervisor."""
        ...


class NullNotifier:
    """No-op watchdog — used when watchdog is disabled or in dev mode."""

    def notify(self) -> None:
        """No-op."""


class FileHeartbeatNotifier:
    """Touches a heartbeat file on each ``notify()`` call.

    Suitable for Docker ``HEALTHCHECK`` commands that stat a file's mtime,
    or custom monitoring scripts that compare heartbeat age against a
    configurable threshold.

    Args:
        heartbeat_path: Filesystem path to the heartbeat file.
    """

    def __init__(self, heartbeat_path: Path) -> None:
        self._path = heartbeat_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        _log.info("file_heartbeat_watchdog_enabled", path=str(self._path))

    def notify(self) -> None:
        """Write monotonic timestamp to the heartbeat file."""
        self._path.write_text(str(time.monotonic()))


class SystemdNotifier:
    """Sends ``WATCHDOG=1`` to systemd via the ``NOTIFY_SOCKET``.

    The socket path is read from the ``NOTIFY_SOCKET`` environment variable,
    which systemd sets automatically for ``Type=notify`` services.  If the
    variable is absent, ``notify()`` is a silent no-op.
    """

    def __init__(self) -> None:
        self._notify_socket: str | None = os.environ.get("NOTIFY_SOCKET")
        if self._notify_socket:
            _log.info("systemd_watchdog_enabled", socket=self._notify_socket)
        else:
            _log.info("systemd_watchdog_disabled_no_notify_socket")

    def notify(self) -> None:
        """Send ``WATCHDOG=1`` datagram to systemd."""
        if not self._notify_socket:
            return
        try:
            import socket as _socket

            if sys.platform == "win32":
                _log.debug("systemd_watchdog_skipped_windows")
                return

            af_unix: int = getattr(_socket, "AF_UNIX", 0)
            sock = _socket.socket(af_unix, _socket.SOCK_DGRAM)
            try:
                addr: str
                if self._notify_socket.startswith("@"):
                    addr = "\0" + self._notify_socket[1:]
                else:
                    addr = self._notify_socket
                sock.sendto(b"WATCHDOG=1", addr)
            finally:
                sock.close()
        except Exception:
            _log.warning("systemd_watchdog_notify_failed", exc_info=True)
