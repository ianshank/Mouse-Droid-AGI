"""Watchdog notifiers for liveness signalling.

Three concrete implementations:

* **SystemdNotifier** — sends ``WATCHDOG=1`` via the ``sdnotify`` library
  (or a subprocess fallback) so systemd knows the service is alive.
* **FileHeartbeatNotifier** — touches a file on disk, useful for Docker
  ``HEALTHCHECK`` directives that check file recency.
* **NullNotifier** — no-op, used in dev/mock mode.

The orchestrator calls ``notify()`` after each successful tick.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


@runtime_checkable
class WatchdogProtocol(Protocol):
    """Watchdog heartbeat protocol.

    Implementations signal liveness to an external supervisor (systemd,
    Docker health check, etc.).
    """

    def notify(self) -> None:
        """Send a heartbeat / keepalive signal."""
        ...


# ---------------------------------------------------------------------------
# Null (dev / mock mode)
# ---------------------------------------------------------------------------


class NullNotifier:
    """No-op watchdog — used when watchdog is disabled."""

    def notify(self) -> None:
        """Do nothing."""


# ---------------------------------------------------------------------------
# Systemd sd_notify
# ---------------------------------------------------------------------------


class SystemdNotifier:
    """Send ``WATCHDOG=1`` to systemd via sd_notify.

    Tries the ``sdnotify`` Python package first; falls back to
    ``systemd-notify --ready`` subprocess if unavailable.

    Args:
        ready_on_init: If ``True``, send ``READY=1`` immediately on
            construction (useful for ``Type=notify`` services).
    """

    def __init__(self, *, ready_on_init: bool = True) -> None:
        self._notifier = self._build_notifier()
        if ready_on_init:
            # Always attempt to send READY=1 so Type=notify services are not
            # killed by systemd when sdnotify is unavailable.  _send() falls
            # back to the systemd-notify subprocess if NOTIFY_SOCKET is set.
            self._send("READY=1")
            _log.info("systemd_notifier_ready")

    # -- internal ----------------------------------------------------------

    @staticmethod
    def _build_notifier() -> Any:
        """Try to import sdnotify; return notifier or None."""
        try:
            import sdnotify

            return sdnotify.SystemdNotifier()
        except ImportError:
            _log.debug("sdnotify_not_available")
            return None

    def _send(self, state: str) -> None:
        if self._notifier is not None:
            self._notifier.notify(state)
        elif os.environ.get("NOTIFY_SOCKET"):
            # Subprocess fallback — only if systemd socket exists. Subprocess
            # spawn here is intentional; sdnotify is the preferred path and is
            # cached in ``self._notifier``. This fallback runs only when the
            # package is unavailable, so the per-call cost is bounded.
            subprocess.run(  # noqa: S603 — fixed argv list, no shell, trusted constant program
                ["systemd-notify", f"--pid={os.getpid()}", state],  # noqa: S607 — systemd-notify on PATH
                check=False,
                capture_output=True,
            )

    def notify(self) -> None:
        """Send WATCHDOG=1 heartbeat to systemd."""
        self._send("WATCHDOG=1")


# ---------------------------------------------------------------------------
# File heartbeat (Docker health checks)
# ---------------------------------------------------------------------------


class FileHeartbeatNotifier:
    """Touch a file on each heartbeat for Docker health checks.

    Docker ``HEALTHCHECK`` can test file recency with::

        test $(( $(date +%s) - $(stat -c %Y "$WATCHDOG_HEARTBEAT_PATH") )) -lt 30

    Args:
        path: Filesystem path for the heartbeat file.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """Return the heartbeat file path."""
        return self._path

    def notify(self) -> None:
        """Touch the heartbeat file with the current timestamp."""
        self._path.write_text(str(time.monotonic()), encoding="utf-8")
