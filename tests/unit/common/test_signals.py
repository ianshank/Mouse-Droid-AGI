"""Unit tests for :mod:`mousedroid.common.signals`.

S-1: SIGTERM's default disposition terminates the process immediately, so
``main.py``'s ``finally: await orchestrator.stop()`` — and with it
``_halt_actuators``' ``emergency_stop()`` — never ran under ``docker stop``
or ``systemctl stop``. These tests pin the signal-registration mechanics
that make that ``finally`` reachable.

Signal delivery is genuinely platform-divergent, so the POSIX cases raise
real signals at the real running loop rather than asserting on a mock, and
the Windows-degradation case forces ``NotImplementedError`` out of
``add_signal_handler`` — the one boundary that cannot be reproduced
faithfully on the other platform.
"""

from __future__ import annotations

import asyncio
import signal
import sys

import pytest

from mousedroid.common.signals import install_shutdown_handlers

_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32",
    reason="loop.add_signal_handler is POSIX-only; Windows takes the degraded path",
)


#: Bound on how long a raised signal may take to surface on the loop.
#:
#: asyncio's POSIX signal delivery goes through ``signal.set_wakeup_fd``'s
#: self-pipe, which the selector only drains on a real loop iteration — a
#: bare ``await asyncio.sleep(0)`` yields without polling it and sees
#: nothing. Waiting on an Event keeps these tests fast in the happy path
#: while still failing (rather than hanging) if delivery regresses.
_DELIVERY_TIMEOUT_S = 5.0


async def _await_signal_name(sig: signal.Signals) -> str:
    """Raise *sig* and return the name the installed handler was called with."""
    delivered: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    def _record(name: str) -> None:
        if not delivered.done():
            delivered.set_result(name)

    uninstall = install_shutdown_handlers(_record)
    try:
        signal.raise_signal(sig)
        return await asyncio.wait_for(delivered, timeout=_DELIVERY_TIMEOUT_S)
    finally:
        uninstall()


@_POSIX_ONLY
async def test_sigterm_invokes_callback_with_signal_name() -> None:
    """A real SIGTERM reaches the callback instead of killing the process.

    This is the S-1 regression itself: before the handler existed, the
    default disposition terminated the process here and no ``finally``
    ever ran.
    """
    assert await _await_signal_name(signal.SIGTERM) == "SIGTERM"


@_POSIX_ONLY
async def test_sigint_invokes_callback_for_parity() -> None:
    """SIGINT is handled too, so an interactive run unwinds identically."""
    assert await _await_signal_name(signal.SIGINT) == "SIGINT"


@_POSIX_ONLY
async def test_uninstall_restores_default_disposition() -> None:
    """Uninstalling must actually release the signal, not just stop calling back.

    Asserted via ``signal.getsignal`` rather than by raising a second
    SIGTERM — an unhandled SIGTERM would terminate the test runner.
    """
    uninstall = install_shutdown_handlers(lambda _name: None)
    assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
    uninstall()
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


@_POSIX_ONLY
async def test_uninstall_is_idempotent() -> None:
    """A second uninstall must not raise — ``stop()`` paths may double-call it."""
    uninstall = install_shutdown_handlers(lambda _name: None)
    uninstall()
    uninstall()


# ---------------------------------------------------------------------------
# Registration bookkeeping, with only the OS boundary faked
#
# The POSIX cases above are the proof that a real signal reaches the
# callback. These cover the surrounding bookkeeping — which signals were
# registered, what gets released, what happens on a partial install — on
# every platform, because that logic is not platform-specific and should
# not go unexercised on the one CI leg that cannot raise a signal.
# ---------------------------------------------------------------------------


class _FakeSignalLoop:
    """Records add/remove calls in place of the real POSIX loop methods."""

    def __init__(self, *, fail_from: int | None = None) -> None:
        self.added: list[tuple[int, str]] = []
        self.removed: list[int] = []
        self._fail_from = fail_from

    def add_signal_handler(self, sig: int, callback: object, *args: object) -> None:
        if self._fail_from is not None and len(self.added) >= self._fail_from:
            raise NotImplementedError
        name = str(args[0]) if args else ""
        self.added.append((int(sig), name))

    def remove_signal_handler(self, sig: int) -> bool:
        self.removed.append(int(sig))
        return True


def _patch_loop(monkeypatch: pytest.MonkeyPatch, fake: _FakeSignalLoop) -> None:
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", fake.add_signal_handler)
    monkeypatch.setattr(loop, "remove_signal_handler", fake.remove_signal_handler)


async def test_registers_every_default_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both defaults are registered, each carrying its own name as the argument."""
    fake = _FakeSignalLoop()
    _patch_loop(monkeypatch, fake)

    install_shutdown_handlers(lambda _name: None)

    assert fake.added == [
        (int(signal.SIGTERM), "SIGTERM"),
        (int(signal.SIGINT), "SIGINT"),
    ]


async def test_honours_an_explicit_signal_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers may narrow the set; nothing outside it is touched."""
    fake = _FakeSignalLoop()
    _patch_loop(monkeypatch, fake)

    install_shutdown_handlers(lambda _name: None, signals=(signal.SIGTERM,))

    assert fake.added == [(int(signal.SIGTERM), "SIGTERM")]


async def test_uninstall_releases_every_registered_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Everything installed is released — a leaked handler outlives the run."""
    fake = _FakeSignalLoop()
    _patch_loop(monkeypatch, fake)

    install_shutdown_handlers(lambda _name: None)()

    assert sorted(fake.removed) == sorted([int(signal.SIGTERM), int(signal.SIGINT)])


async def test_uninstall_does_not_release_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Idempotent by draining: a second call releases nothing further."""
    fake = _FakeSignalLoop()
    _patch_loop(monkeypatch, fake)

    uninstall = install_shutdown_handlers(lambda _name: None)
    uninstall()
    uninstall()

    assert len(fake.removed) == 2


async def test_partial_install_releases_only_what_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A half-failed install must not try to release what it never took."""
    fake = _FakeSignalLoop(fail_from=1)
    _patch_loop(monkeypatch, fake)

    install_shutdown_handlers(lambda _name: None)()

    assert fake.added == [(int(signal.SIGTERM), "SIGTERM")]
    assert fake.removed == [int(signal.SIGTERM)]


async def test_release_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A loop already tearing down may refuse the release — never raise from cleanup.

    ``uninstall`` runs in a ``finally`` on the shutdown path; raising there
    would mask the original reason the process was stopping.
    """
    fake = _FakeSignalLoop()
    _patch_loop(monkeypatch, fake)
    uninstall = install_shutdown_handlers(lambda _name: None)

    def _refuse(_sig: int) -> bool:
        raise RuntimeError("event loop is closed")

    monkeypatch.setattr(asyncio.get_running_loop(), "remove_signal_handler", _refuse)

    uninstall()


async def test_degrades_when_platform_lacks_signal_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows raises NotImplementedError — install must degrade, never crash.

    Forced explicitly so the contract is pinned on POSIX CI legs too; the
    advisory ``test-windows`` job exercises the same branch natively.
    """
    loop = asyncio.get_running_loop()

    def _unsupported(*_args: object, **_kwargs: object) -> None:
        raise NotImplementedError

    monkeypatch.setattr(loop, "add_signal_handler", _unsupported)

    uninstall = install_shutdown_handlers(lambda _name: None)

    # Returns a callable no-op uninstaller rather than raising.
    uninstall()


async def test_degrades_when_not_on_main_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``add_signal_handler`` raises RuntimeError off the main thread."""
    loop = asyncio.get_running_loop()

    def _off_main_thread(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("set_wakeup_fd only works in main thread")

    monkeypatch.setattr(loop, "add_signal_handler", _off_main_thread)

    uninstall = install_shutdown_handlers(lambda _name: None)
    uninstall()


@_POSIX_ONLY
async def test_partial_install_still_uninstalls_what_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the second signal fails to register, the first must still be released.

    Otherwise a half-installed handler outlives the run and the next
    ``install`` sees a dirty disposition.
    """
    loop = asyncio.get_running_loop()
    real = loop.add_signal_handler
    calls: list[int] = []

    def _fail_on_second(sig: int, callback: object, *args: object) -> None:
        calls.append(sig)
        if len(calls) > 1:
            raise NotImplementedError
        real(sig, callback, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(loop, "add_signal_handler", _fail_on_second)

    uninstall = install_shutdown_handlers(lambda _name: None)
    uninstall()

    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
