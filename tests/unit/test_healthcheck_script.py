"""Unit tests for ``scripts/mousedroid_healthcheck.sh``.

Drives the actual POSIX shell script with ``subprocess.run`` against
synthetic heartbeat / env-file fixtures. Each case exercises one branch
of the script's control flow: fresh heartbeat, stale heartbeat, missing
heartbeat (in/out of grace window), unset env vars (defaults path),
float-tolerant threshold.

Skipped on Windows — the script is POSIX sh.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "mousedroid_healthcheck.sh"

pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("sh") is None or not _SCRIPT.exists(),
    reason="POSIX sh + script required",
)


def _run(env_file: Path | None, extra_env: dict[str, str] | None = None) -> int:
    """Invoke the healthcheck script with optional env file + extra env vars."""
    env = dict(os.environ)
    if env_file is not None:
        env["MOUSEDROID_HEALTHCHECK_ENV_FILE"] = str(env_file)
    else:
        # Force the script's default lookup to a path that won't exist
        # so we exercise the "no env file" branch deterministically.
        env["MOUSEDROID_HEALTHCHECK_ENV_FILE"] = "/nonexistent/mousedroid.env"
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        ["sh", str(_SCRIPT)],
        env=env,
        check=False,
        capture_output=True,
        timeout=10,
    )
    return result.returncode


def _write_env(path: Path, **values: str) -> None:
    """Write a shell-sourceable env file."""
    body = "\n".join(f"{k}='{v}'" for k, v in values.items())
    path.write_text(body + "\n", encoding="utf-8")


def _age(path: Path, seconds: float) -> None:
    """Backdate ``path`` mtime by ``seconds``."""
    now = time.time()
    os.utime(path, (now - seconds, now - seconds))


def test_fresh_heartbeat_exits_zero(tmp_path: Path) -> None:
    hb = tmp_path / "hb"
    hb.touch()
    env_file = tmp_path / "env"
    _write_env(
        env_file,
        MOUSEDROID_HEARTBEAT_PATH=str(hb),
        MOUSEDROID_HEARTBEAT_STALE_S="30",
        MOUSEDROID_START_GRACE_S="60",
        MOUSEDROID_START_GRACE_FILE=str(tmp_path / "start"),
    )
    assert _run(env_file) == 0


def test_stale_heartbeat_exits_one(tmp_path: Path) -> None:
    hb = tmp_path / "hb"
    hb.touch()
    _age(hb, 60.0)
    env_file = tmp_path / "env"
    _write_env(
        env_file,
        MOUSEDROID_HEARTBEAT_PATH=str(hb),
        MOUSEDROID_HEARTBEAT_STALE_S="30",
        MOUSEDROID_START_GRACE_S="60",
        MOUSEDROID_START_GRACE_FILE=str(tmp_path / "start"),
    )
    assert _run(env_file) == 1


def test_missing_heartbeat_in_grace_window_exits_zero(tmp_path: Path) -> None:
    """No heartbeat yet, but we're still within the start-grace window."""
    start = tmp_path / "start"
    start.touch()
    # start file is fresh; grace=60s should accept
    env_file = tmp_path / "env"
    _write_env(
        env_file,
        MOUSEDROID_HEARTBEAT_PATH=str(tmp_path / "hb-absent"),
        MOUSEDROID_HEARTBEAT_STALE_S="30",
        MOUSEDROID_START_GRACE_S="60",
        MOUSEDROID_START_GRACE_FILE=str(start),
    )
    assert _run(env_file) == 0


def test_missing_heartbeat_past_grace_window_exits_one(tmp_path: Path) -> None:
    """Heartbeat absent and grace window has elapsed → unhealthy."""
    start = tmp_path / "start"
    start.touch()
    _age(start, 120.0)
    env_file = tmp_path / "env"
    _write_env(
        env_file,
        MOUSEDROID_HEARTBEAT_PATH=str(tmp_path / "hb-absent"),
        MOUSEDROID_HEARTBEAT_STALE_S="30",
        MOUSEDROID_START_GRACE_S="60",
        MOUSEDROID_START_GRACE_FILE=str(start),
    )
    assert _run(env_file) == 1


def test_missing_heartbeat_and_no_grace_marker_exits_one(tmp_path: Path) -> None:
    """Heartbeat absent and no start-grace anchor → unhealthy."""
    env_file = tmp_path / "env"
    _write_env(
        env_file,
        MOUSEDROID_HEARTBEAT_PATH=str(tmp_path / "hb-absent"),
        MOUSEDROID_HEARTBEAT_STALE_S="30",
        MOUSEDROID_START_GRACE_S="60",
        MOUSEDROID_START_GRACE_FILE=str(tmp_path / "start-absent"),
    )
    assert _run(env_file) == 1


def test_env_file_absent_uses_safe_defaults(tmp_path: Path) -> None:
    """Old image without entrypoint hook → script falls back to defaults.

    Drives the default-path branch by pointing env vars at a fresh file
    that satisfies the default 30s threshold. Verifies the script
    doesn't hard-require the env file.
    """
    # No env file; override the default heartbeat path via direct env var
    # so we don't touch the real /tmp/mousedroid_heartbeat.
    hb = tmp_path / "hb-default"
    hb.touch()
    rc = _run(
        None,
        extra_env={
            "MOUSEDROID_HEARTBEAT_PATH": str(hb),
            # MOUSEDROID_HEARTBEAT_STALE_S left unset → script uses 30s default
        },
    )
    assert rc == 0


def test_float_threshold_compares_with_awk(tmp_path: Path) -> None:
    """Sub-second tolerances work via awk float comparison."""
    hb = tmp_path / "hb"
    hb.touch()
    _age(hb, 4.5)
    env_file = tmp_path / "env"
    _write_env(
        env_file,
        MOUSEDROID_HEARTBEAT_PATH=str(hb),
        MOUSEDROID_HEARTBEAT_STALE_S="5.0",
        MOUSEDROID_START_GRACE_S="60",
        MOUSEDROID_START_GRACE_FILE=str(tmp_path / "start"),
    )
    assert _run(env_file) == 0
