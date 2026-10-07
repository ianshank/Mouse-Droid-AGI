"""Unit: values compose splices into a mount spec are refused before compose runs.

``docker-compose.jetson.yml`` mounts the named volume ``mousedroid_tensorrt_cache``
at ``${MOUSEDROID_JETSON__TENSORRT_CACHE_DIR:-...}``. Compose's short volume syntax
is ``VOLUME:TARGET[:MODE]``, so a value containing ``:`` injects a third field,
and a value such as ``/etc`` mounts the volume over an image directory. The cache
holds torch2trt engines that are unpickled at load time, so where it lands is not
cosmetic (F-052 task 5.6).

Compose is started from two places, and both are covered here, because guarding
only one would leave the other open: ``scripts/docker_deploy.sh`` and
``scripts/preflight_check.sh`` (the systemd unit's fatal ``ExecStartPre``). Both
call the one validator, ``scripts/validate_compose_env.sh``.

Real script, real fixture, specific messages — per ``test_deploy_remote_guard.py``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests._bash import requires_bash

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO_ROOT / "scripts"
_VALIDATOR = _SCRIPTS / "validate_compose_env.sh"
_DEPLOY = _SCRIPTS / "docker_deploy.sh"
_PREFLIGHT = _SCRIPTS / "preflight_check.sh"
_UNIT = _SCRIPTS / "mousedroid-docker.service"
_KEY = "MOUSEDROID_JETSON__TENSORRT_CACHE_DIR"

pytestmark = requires_bash()


def _validate(value: str | None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != _KEY}
    if value is not None:
        env[_KEY] = value
    return subprocess.run(
        ["bash", str(_VALIDATOR)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        None,  # unset: compose falls back to its own (valid) default
        "/opt/mousedroid/tensorrt_cache",  # the schema and compose default
        "/mnt/nvme/trt_cache",  # a relocation to a bigger disk
        "/data/cache",
        "/var/cache/mousedroid",
    ],
)
def test_safe_values_are_accepted(value: str | None) -> None:
    proc = _validate(value)

    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("/x:ro", "only [A-Za-z0-9._/-] is allowed"),  # injects a :MODE field
        ("/x:/y", "only [A-Za-z0-9._/-] is allowed"),  # injects a third field
        ("/etc", "must not be the root or a top-level directory"),
        ("/opt", "must not be the root or a top-level directory"),
        ("/", "must not be the root or a top-level directory"),
        ("/etc/mousedroid", "must not be inside /etc"),
        ("/usr/lib/trt", "must not be inside /usr"),
        ("tensorrt_cache", "must be an absolute path"),
        ("/opt/../etc", "must not contain empty, '.' or '..' components"),
        ("/opt//x", "must not contain empty, '.' or '..' components"),
        ("/opt/x/", "must not end in '/'"),
        ("/opt/my cache", "only [A-Za-z0-9._/-] is allowed"),
        ("/opt/$HOME", "only [A-Za-z0-9._/-] is allowed"),
    ],
)
def test_unsafe_values_are_refused_with_the_reason(value: str, reason: str) -> None:
    proc = _validate(value)

    assert proc.returncode == 1
    assert f"{_KEY}={value} refused: {reason}" in proc.stderr


# ---------------------------------------------------------------------------
# Entry point 1: docker_deploy.sh refuses before any compose command
# ---------------------------------------------------------------------------

_DOCKER_SHIM = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$DOCKER_LOG"
case "$1" in ps) echo "abc123" ;; exec) echo "OK" ;; esac
exit 0
"""


def _deploy(tmp_path: Path, cache_dir: str) -> tuple[subprocess.CompletedProcess[str], str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "docker"
    shim.write_text(_DOCKER_SHIM, encoding="utf-8")
    shim.chmod(0o755)
    docker_log = tmp_path / "docker.log"
    docker_log.write_text("", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    proc = subprocess.run(
        ["bash", str(_DEPLOY), "--no-build"],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "DOCKER_LOG": str(docker_log),
            "MOUSEDROID_INSTALL_DIR": str(tmp_path),
            "MOUSEDROID_CONFIG_DIR": str(tmp_path / "etc"),
            "MOUSEDROID_DOCKER_ENV_FILE": str(tmp_path / "absent.env"),
            "MOUSEDROID_HEALTH_TIMEOUT": "1",
            _KEY: cache_dir,
        },
        cwd=str(_REPO_ROOT),
    )
    return proc, docker_log.read_text(encoding="utf-8")


def test_docker_deploy_refuses_before_compose_ever_runs(tmp_path: Path) -> None:
    """The refusal must precede every compose invocation, not just ``up``.

    ``pull`` and ``build`` parse the file too, so the docker shim's log is
    checked for *any* compose call rather than for ``up`` alone.
    """
    proc, docker_calls = _deploy(tmp_path, "/x:ro")

    assert proc.returncode != 0
    assert f"{_KEY}=/x:ro refused" in proc.stderr
    assert "compose" not in docker_calls, f"compose ran despite the refusal:\n{docker_calls}"


def test_docker_deploy_proceeds_to_compose_with_a_safe_value(tmp_path: Path) -> None:
    """...and the guard is not refusing everything: a safe value reaches compose.

    Without this, the test above would also pass if the validator rejected
    every input, or if the script died for an unrelated reason first.
    """
    _, docker_calls = _deploy(tmp_path, "/opt/mousedroid/tensorrt_cache")

    assert "compose" in docker_calls, "a safe value never reached compose"


# ---------------------------------------------------------------------------
# Entry point 2: the systemd boot path, via preflight_check.sh
# ---------------------------------------------------------------------------


def _preflight(cache_dir: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_PREFLIGHT), "--skip-devices", "--skip-models"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, _KEY: cache_dir},
    )


def test_preflight_counts_an_unsafe_value_as_a_critical_failure(tmp_path: Path) -> None:
    proc = _preflight("/etc")
    output = proc.stdout + proc.stderr

    assert f"{_KEY}=/etc refused" in output
    assert "[FAIL] A value compose interpolates into a mount spec was refused" in output
    assert proc.returncode != 0


def test_preflight_passes_the_check_for_a_safe_value(tmp_path: Path) -> None:
    proc = _preflight("/opt/mousedroid/tensorrt_cache")

    assert "[OK] Compose-interpolated values are safe" in proc.stdout + proc.stderr


def test_the_unit_runs_preflight_fatally_and_before_compose() -> None:
    """The boot-path coverage claim, pinned against the unit file itself.

    ``preflight_check.sh`` only protects the boot path if the unit runs it as a
    FATAL ``ExecStartPre`` (no ``-`` prefix, which systemd treats as "ignore
    failure") and before the first line that runs compose.
    """
    exec_lines = [
        line.strip()
        for line in _UNIT.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("ExecStartPre=", "ExecStart="))
    ]
    preflight = [i for i, line in enumerate(exec_lines) if "preflight_check.sh" in line]
    compose = [i for i, line in enumerate(exec_lines) if "docker compose" in line]

    assert preflight, f"the unit no longer runs preflight_check.sh: {exec_lines}"
    first = exec_lines[preflight[0]]
    assert not first.startswith("ExecStartPre=-"), (
        f"preflight is non-fatal, so a refused value would not stop boot: {first}"
    )
    assert compose, exec_lines
    assert preflight[0] < compose[0], f"compose runs before preflight: {exec_lines}"
