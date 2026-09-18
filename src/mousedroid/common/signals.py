"""Asyncio shutdown-signal registration (S-1).

SIGTERM's default disposition terminates the process *immediately*. Both
``docker stop`` and ``systemctl stop`` send SIGTERM first, so without a
handler the orchestrator's ``try/finally`` never unwinds and
``_halt_actuators`` — the one call site that issues
``ESP32CommProtocol.emergency_stop`` on shutdown — never runs. The rover
keeps whatever velocity was last latched in firmware until SIGKILL lands
(10 s later under Docker's default grace period), and then keeps it
afterwards too, because SIGKILL cannot be handled at all.

``scripts/mousedroid_entrypoint.sh`` already ``exec``s the Python process
specifically so "signals (SIGTERM from ``docker stop``) are delivered
directly to mousedroid". This module is the Python half that intent was
missing.

Deliberately framework-agnostic — it knows nothing about orchestrators,
hardware, or config — so the registration mechanics stay testable on their
own and any future entry point can reuse them.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from collections.abc import Callable, Sequence
from typing import Final

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

#: Signals a service manager uses to request a graceful stop.
#:
#: SIGTERM is what ``docker stop`` and ``systemctl stop`` send, and is the
#: one that actually matters on the rover. SIGINT is included for parity so
#: an interactive ``Ctrl-C`` run unwinds through exactly the same path an
#: operator's ``docker stop`` will — otherwise the interactive path is the
#: only one anybody ever exercises, and it is the one that already worked.
DEFAULT_SHUTDOWN_SIGNALS: Final[tuple[signal.Signals, ...]] = (
    signal.SIGTERM,
    signal.SIGINT,
)


def install_shutdown_handlers(
    on_shutdown: Callable[[str], None],
    *,
    signals: Sequence[signal.Signals] = DEFAULT_SHUTDOWN_SIGNALS,
) -> Callable[[], None]:
    """Route *signals* to *on_shutdown* on the running event loop.

    *on_shutdown* is called with the signal's name (e.g. ``"SIGTERM"``) from
    within the loop, so it must not block — flip a flag and return.

    Never raises. :meth:`asyncio.AbstractEventLoop.add_signal_handler` is
    POSIX-only: on Windows (``ProactorEventLoop``) it raises
    ``NotImplementedError``, and on any platform it raises ``RuntimeError``
    when the loop is not on the main thread. Both degrade to a logged
    warning rather than taking down a rover at boot over a missing
    convenience — a process that cannot register handlers is exactly as
    safe as it was before this module existed, and strictly safer than one
    that refuses to start.

    Args:
        on_shutdown: Called with the delivered signal's name.
        signals: Signals to route. Defaults to
            :data:`DEFAULT_SHUTDOWN_SIGNALS`.

    Returns:
        An idempotent callable that releases every handler this call
        installed, restoring the default disposition. Call it before the
        loop closes; a second call is a no-op.
    """
    loop = asyncio.get_running_loop()
    installed: list[signal.Signals] = []

    for sig in signals:
        try:
            loop.add_signal_handler(sig, on_shutdown, sig.name)
        except (NotImplementedError, RuntimeError):
            # Windows, or a non-main thread. Logged once per signal so the
            # absence of a graceful-shutdown path is visible in container
            # logs rather than discovered from a rover that kept driving.
            _log.warning(
                "shutdown_signal_handler_unavailable",
                signal=sig.name,
                platform=sys.platform,
            )
            continue
        installed.append(sig)

    if installed:
        _log.info(
            "shutdown_signal_handlers_installed",
            signals=[s.name for s in installed],
        )

    def _uninstall() -> None:
        """Release every handler installed above; idempotent by draining."""
        while installed:
            sig = installed.pop()
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError):
                _log.debug("shutdown_signal_handler_release_failed", signal=sig.name)

    return _uninstall
