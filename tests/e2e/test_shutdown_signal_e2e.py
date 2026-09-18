"""End-to-end: `docker stop` against a real mousedroid process (S-1).

The closest reproduction of the production failure available without a
rover. A real ``python -m mousedroid.main`` process is started, allowed to
reach its 30 Hz loop, then sent a real SIGTERM — exactly what ``docker
stop`` and ``systemctl stop`` deliver to PID 1 after
``scripts/mousedroid_entrypoint.sh`` ``exec``s it.

What only this tier can prove: that the handler survives real process
startup (imports, factory wiring, ``asyncio.run``'s own loop), that the
process *exits on its own* rather than being killed, and that the halt is
visible in the logs an operator would actually read.

Before the fix this process died on SIGTERM with no shutdown logging at
all — the ``finally`` never ran.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX signal delivery; Windows has no loop.add_signal_handler",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Generous: a cold start imports torch and builds the whole factory graph.
_STARTUP_TIMEOUT_S = 180.0
#: ``shutdown_grace_s`` (2 s) plus teardown; far under Docker's 10 s default.
_SHUTDOWN_TIMEOUT_S = 30.0


def _child_env() -> dict[str, str]:
    """Build the child's env explicitly.

    ``tests/e2e/conftest.py`` has an autouse fixture forcing
    ``MOUSEDROID_MOCK_HARDWARE=false`` for the Jetson suites in this
    directory, and the child inherits ``os.environ`` — so the override here
    is load-bearing, not belt-and-braces.
    """
    env = dict(os.environ)
    env["MOUSEDROID_MOCK_HARDWARE"] = "true"
    env["PYTHONUNBUFFERED"] = "1"
    # Keep the child off the parent's coverage/pytest plumbing.
    env.pop("COV_CORE_SOURCE", None)
    env.pop("PYTEST_CURRENT_TEST", None)
    return env


def _spawn() -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "mousedroid.main", "--mock-hardware"],
        cwd=str(_REPO_ROOT),
        env=_child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def _wait_for_loop(proc: subprocess.Popen[str]) -> list[str]:
    """Read stdout until the control loop announces itself.

    Returns the lines consumed so far, so the caller can keep the full
    transcript for its assertions.
    """
    lines: list[str] = []
    deadline = time.monotonic() + _STARTUP_TIMEOUT_S
    assert proc.stdout is not None

    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line)
        if "main_loop_starting" in line:
            return lines

    proc.kill()
    proc.wait(timeout=30)
    raise AssertionError(
        "mousedroid never reached its control loop; transcript:\n" + "".join(lines[-40:])
    )


@pytest.mark.slow
def test_real_process_halts_motors_on_sigterm() -> None:
    """SIGTERM to a live process must halt the wheels and exit cleanly."""
    proc = _spawn()
    try:
        transcript = _wait_for_loop(proc)

        proc.send_signal(signal.SIGTERM)

        remaining = proc.communicate(timeout=_SHUTDOWN_TIMEOUT_S)[0]
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)
        raise AssertionError(
            f"process did not exit within {_SHUTDOWN_TIMEOUT_S}s of SIGTERM — "
            "docker would escalate to SIGKILL with the motors still latched"
        ) from None
    finally:
        if proc.poll() is None:  # pragma: no cover - only on an unexpected hang
            proc.kill()
            proc.wait(timeout=30)

    output = "".join(transcript) + (remaining or "")

    # The signal was received by Python rather than killing the process.
    assert "shutdown_requested" in output, output[-3000:]
    # The teardown actually reached the actuators. Which event name appears
    # depends on which driver the factory wired: MockESP32Driver logs
    # ``mock_emergency_stop``, the real BaseESP32Driver logs
    # ``esp32_emergency_stop``. Accept either — this test is about the halt
    # being reached at all, not about which transport served it.
    assert any(
        marker in output for marker in ("esp32_emergency_stop", "mock_emergency_stop")
    ), output[-3000:]
    # The serial transport was released, so a restart can reclaim the port.
    assert any(
        marker in output for marker in ("esp32_disconnected", "mock_esp32_disconnected")
    ), output[-3000:]
    # ...and teardown completed, rather than dying partway through.
    assert "orchestrator_stopped" in output, output[-3000:]

    # Exited under its own control. A process killed by a signal reports a
    # negative returncode (-SIGTERM) on POSIX; a handled one does not.
    assert proc.returncode is not None
    assert proc.returncode >= 0, (
        f"process was killed by signal {-proc.returncode}, not shut down gracefully"
    )


@pytest.mark.slow
def test_real_process_installs_handlers_at_startup() -> None:
    """The install must be visible in boot logs, so a degraded rover is diagnosable."""
    proc = _spawn()
    try:
        transcript = _wait_for_loop(proc)
        proc.send_signal(signal.SIGTERM)
        remaining = proc.communicate(timeout=_SHUTDOWN_TIMEOUT_S)[0]
    finally:
        if proc.poll() is None:  # pragma: no cover - only on an unexpected hang
            proc.kill()
            proc.wait(timeout=30)

    output = "".join(transcript) + (remaining or "")
    assert "shutdown_signal_handlers_installed" in output, output[-3000:]
