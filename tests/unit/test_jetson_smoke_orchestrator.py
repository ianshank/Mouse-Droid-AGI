"""Regression tests for the host-detection guards in jetson_smoke_test.sh.

These tests fix smoke-report finding F-001 (the orchestrator emitting FAIL
instead of SKIP for Jetson-only checks on non-Jetson hosts) and F-002 (the
`/proc/meminfo` parse aborting the script via `set -e` on hosts where the
file is missing or lacks `MemAvailable`).

The script is invoked via `bash` with these env overrides (kept in sync
with the comment block at the top of ``scripts/jetson_smoke_test.sh``):

* ``MOUSEDROID_SMOKE_FORCE_PLATFORM`` — ``jetson`` | ``non-jetson`` | ``auto``
* ``MOUSEDROID_SMOKE_TEGRA_RELEASE_PATH`` — alternate tegra-release marker
* ``MOUSEDROID_SMOKE_MEMINFO_PATH`` — alternate /proc/meminfo path
* ``MOUSEDROID_SMOKE_THERMAL_PATH`` — alternate thermal-zone temp path
* ``MOUSEDROID_SMOKE_PYTHON`` — pinned Python interpreter (used here to
  force a Python that lacks CUDA/TensorRT so the forced-jetson assertions
  remain deterministic on Linux+GPU hosts)

These tests are entirely hermetic: no real GPU/CUDA/TensorRT is required,
and the test never touches /etc or /sys on the host. Hosts without
``bash`` on PATH (Windows minimal CI) are skipped cleanly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "jetson_smoke_test.sh"

_BASH = shutil.which("bash")
_BASH_AVAILABLE = pytest.mark.skipif(
    _BASH is None,
    reason="bash unavailable on this host",
)


# ``jetson_smoke_test.sh`` resolves Python via (in order):
#   1. ``MOUSEDROID_SMOKE_PYTHON`` env var
#   2. ``/opt/mousedroid/venv/bin/python`` (Jetson install path)
#   3. ``python3`` on PATH
# On Windows-Git-Bash without a ``python3`` shim AND without
# ``MOUSEDROID_SMOKE_PYTHON`` set, the script bails before reaching the
# assertion under test with ``ERROR: No Python runtime found for
# jetson_smoke_test.sh``. Skip the whole module cleanly in that case so
# the failures don't pollute the CI signal — the tests still RUN on
# Linux / Jetson hosts where ``python3`` is on PATH, or on any host
# where the operator points ``MOUSEDROID_SMOKE_PYTHON`` at a real
# Python interpreter.
_PYTHON_REACHABLE_FROM_BASH = (
    os.environ.get("MOUSEDROID_SMOKE_PYTHON") is not None
    or shutil.which("python3") is not None
    or Path("/opt/mousedroid/venv/bin/python").exists()
)
pytestmark = pytest.mark.skipif(
    sys.platform == "win32" and not _PYTHON_REACHABLE_FROM_BASH,
    reason=(
        "jetson_smoke_test.sh requires ``python3`` reachable from the bash "
        "subprocess; on Windows-Git-Bash without a ``python3`` shim the "
        "script bails before any assertion runs. Set "
        "MOUSEDROID_SMOKE_PYTHON to bypass on hosts where Python is "
        "reachable under a different name."
    ),
)

# A realistic /proc/meminfo fixture: MemTotal + MemAvailable, KB units.
# Values chosen to produce a deterministic "used%" (50%) so the test does
# not depend on the host's actual memory state.
_FAKE_MEMINFO_FULL = """\
MemTotal:       8000000 kB
MemFree:        2000000 kB
MemAvailable:   4000000 kB
Buffers:         100000 kB
Cached:         1500000 kB
"""

# Older-kernel / non-Linux-emulation variant: no MemAvailable. The script
# should fall back to MemFree and still PASS rather than aborting.
_FAKE_MEMINFO_NO_MEMAVAILABLE = """\
MemTotal:       8000000 kB
MemFree:        3000000 kB
"""


def _run_smoke(
    *args: str,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke ``jetson_smoke_test.sh`` with a pruned, deterministic env."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
    }
    if env_extra:
        env.update(env_extra)
    assert _BASH is not None  # mypy
    return subprocess.run(
        [_BASH, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# Structural assertions (host-agnostic, always run)
# ---------------------------------------------------------------------------


def test_smoke_script_exists() -> None:
    assert _SCRIPT.is_file(), f"missing: {_SCRIPT}"


def test_smoke_script_declares_is_jetson_host_helper() -> None:
    """F-001: the bash helper must be present and named to match the python one."""
    content = _SCRIPT.read_text(encoding="utf-8")
    assert "is_jetson_host()" in content, "missing is_jetson_host() bash helper"
    assert "MOUSEDROID_SMOKE_FORCE_PLATFORM" in content
    assert "/etc/nv_tegra_release" in content


def test_smoke_script_declares_meminfo_override() -> None:
    """F-002: the memory check must be overridable + guarded."""
    content = _SCRIPT.read_text(encoding="utf-8")
    assert "MOUSEDROID_SMOKE_MEMINFO_PATH" in content
    # MemFree fallback for kernels < 3.14 / non-Linux emulation.
    assert "MemFree" in content


# ---------------------------------------------------------------------------
# Behavioural tests (require bash)
# ---------------------------------------------------------------------------


@_BASH_AVAILABLE
def test_non_jetson_host_skips_cuda_tensorrt_thermal(tmp_path: Path) -> None:
    """F-001: SKIP (not FAIL) the three Jetson-only checks on non-Jetson hosts."""
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(_FAKE_MEMINFO_FULL, encoding="utf-8")

    result = _run_smoke(
        "system",
        env_extra={
            "MOUSEDROID_SMOKE_FORCE_PLATFORM": "non-jetson",
            "MOUSEDROID_SMOKE_MEMINFO_PATH": str(meminfo),
        },
    )
    stdout = result.stdout

    # The three Jetson-only checks must SKIP, not FAIL.
    assert "SKIP: torch.cuda.is_available" in stdout, stdout
    assert "SKIP: import tensorrt" in stdout, stdout
    assert "SKIP: thermal sensor read" in stdout, stdout

    # And memory must still PASS (50% used in the fixture).
    assert "PASS: memory check" in stdout, stdout
    assert "Failed:  0" in stdout, stdout
    assert "Skipped: 3" in stdout, stdout
    assert result.returncode == 0, (
        f"non-jetson run should exit 0 (no real failures); "
        f"got {result.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{result.stderr}"
    )


@_BASH_AVAILABLE
def test_force_jetson_host_attempts_cuda_tensorrt_thermal(tmp_path: Path) -> None:
    """F-001: forcing jetson on a dev box must NOT short-circuit — the three
    Jetson-only checks must actually run. We deliberately do *not* assert
    PASS/FAIL outcomes here, because a Linux host with an NVIDIA GPU,
    ``tensorrt`` installed, and a readable ``/sys/.../thermal_zone0/temp``
    would legitimately PASS the same checks that a Windows/macOS dev box
    fails. What matters for F-001 is that none of the three subchecks
    bypasses the real probe by emitting SKIP — i.e. the guard's "force
    jetson" branch routes through the actual implementation.

    Thermal is pinned to a missing path so its outcome is deterministic
    (FAIL) on every host; CUDA + TensorRT are left to whatever the host's
    Python interpreter provides.
    """
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(_FAKE_MEMINFO_FULL, encoding="utf-8")
    missing_thermal = tmp_path / "no-thermal-temp"  # deterministic FAIL

    result = _run_smoke(
        "system",
        env_extra={
            "MOUSEDROID_SMOKE_FORCE_PLATFORM": "jetson",
            "MOUSEDROID_SMOKE_MEMINFO_PATH": str(meminfo),
            "MOUSEDROID_SMOKE_THERMAL_PATH": str(missing_thermal),
        },
    )
    stdout = result.stdout

    # The "not running on a Jetson host" SKIP message must be absent for
    # every Jetson-only subcheck — that's the F-001 contract. Whether each
    # check then PASSes or FAILs is host-dependent and not asserted.
    assert "SKIP: torch.cuda.is_available" not in stdout, stdout
    assert "SKIP: import tensorrt" not in stdout, stdout
    assert "SKIP: thermal sensor read" not in stdout, stdout

    # Memory still PASSes because we point at a sane fixture.
    assert "PASS: memory check" in stdout, stdout

    # Thermal is pinned to a missing path, so it must FAIL deterministically
    # (proves the THERMAL_PATH override took effect under force-jetson).
    assert "FAIL: thermal sensor read" in stdout, stdout

    # Exit code mirrors the FAILURES counter, which is at least 1 (thermal).
    assert result.returncode >= 1, (
        f"forced-jetson with pinned-missing thermal should record at least one FAIL; "
        f"got rc={result.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{result.stderr}"
    )


@_BASH_AVAILABLE
def test_memory_check_skips_when_meminfo_unreadable(tmp_path: Path) -> None:
    """F-002 guard: missing meminfo must SKIP, not crash the script."""
    missing = tmp_path / "definitely-not-meminfo"  # does not exist

    result = _run_smoke(
        "system",
        env_extra={
            "MOUSEDROID_SMOKE_FORCE_PLATFORM": "non-jetson",
            "MOUSEDROID_SMOKE_MEMINFO_PATH": str(missing),
        },
    )
    stdout = result.stdout

    assert "SKIP: memory check" in stdout, stdout
    # Summary must be reached: the script no longer aborts on missing meminfo.
    assert "Smoke Test Summary" in stdout, stdout
    assert result.returncode == 0, (
        f"missing meminfo should SKIP cleanly (exit 0); got {result.returncode}"
        f"\nSTDOUT:\n{stdout}\nSTDERR:\n{result.stderr}"
    )


@_BASH_AVAILABLE
def test_memory_check_falls_back_to_memfree(tmp_path: Path) -> None:
    """F-002 fallback: kernels < 3.14 (and Git Bash on Windows) have no
    MemAvailable. The script must fall back to MemFree and still PASS."""
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(_FAKE_MEMINFO_NO_MEMAVAILABLE, encoding="utf-8")

    result = _run_smoke(
        "system",
        env_extra={
            "MOUSEDROID_SMOKE_FORCE_PLATFORM": "non-jetson",
            "MOUSEDROID_SMOKE_MEMINFO_PATH": str(meminfo),
        },
    )
    stdout = result.stdout

    assert "PASS: memory check" in stdout, stdout
    # Explicitly notes the degraded source so on-call sees "MemFree" not "MemAvailable".
    assert "MemFree" in stdout, stdout
    assert result.returncode == 0, (
        f"MemFree fallback should PASS; got {result.returncode}"
        f"\nSTDOUT:\n{stdout}\nSTDERR:\n{result.stderr}"
    )


@_BASH_AVAILABLE
def test_unknown_force_platform_value_warns_but_continues(tmp_path: Path) -> None:
    """A typo in MOUSEDROID_SMOKE_FORCE_PLATFORM must not silently flip behaviour.
    The script logs a WARN and falls back to real detection."""
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(_FAKE_MEMINFO_FULL, encoding="utf-8")
    # Point tegra-release marker at a path that does NOT exist so real
    # detection picks "non-jetson".
    missing_marker = tmp_path / "no-tegra-here"

    result = _run_smoke(
        "system",
        env_extra={
            "MOUSEDROID_SMOKE_FORCE_PLATFORM": "JETSON_PLEASE",  # typo
            "MOUSEDROID_SMOKE_MEMINFO_PATH": str(meminfo),
            "MOUSEDROID_SMOKE_TEGRA_RELEASE_PATH": str(missing_marker),
        },
    )

    assert "WARN: ignoring unknown MOUSEDROID_SMOKE_FORCE_PLATFORM" in result.stderr
    # Real detection then triggers SKIPs because the marker is missing.
    assert "SKIP: torch.cuda.is_available" in result.stdout
    assert result.returncode == 0


@_BASH_AVAILABLE
def test_empty_force_platform_treated_as_auto(tmp_path: Path) -> None:
    """An explicitly-empty MOUSEDROID_SMOKE_FORCE_PLATFORM (e.g. ``export X=``)
    must behave like unset/auto — no spurious WARN, real detection used.
    """
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(_FAKE_MEMINFO_FULL, encoding="utf-8")
    missing_marker = tmp_path / "no-tegra"

    result = _run_smoke(
        "system",
        env_extra={
            "MOUSEDROID_SMOKE_FORCE_PLATFORM": "",  # explicitly empty
            "MOUSEDROID_SMOKE_MEMINFO_PATH": str(meminfo),
            "MOUSEDROID_SMOKE_TEGRA_RELEASE_PATH": str(missing_marker),
        },
    )

    assert "WARN: ignoring unknown MOUSEDROID_SMOKE_FORCE_PLATFORM" not in result.stderr
    # Real detection fires: marker missing → non-Jetson → SKIPs.
    assert "SKIP: torch.cuda.is_available" in result.stdout
    assert result.returncode == 0


@_BASH_AVAILABLE
def test_force_jetson_echo_reports_override_tegra_path(tmp_path: Path) -> None:
    """M-4: the 'Host detected as Jetson (...)' echo must reflect the actual
    tegra path checked, including any MOUSEDROID_SMOKE_TEGRA_RELEASE_PATH
    override — not the hardcoded /etc/nv_tegra_release default. We force-jetson
    here so the test passes on Windows (where ``uname -s`` ≠ Linux); the
    echo path is the same code regardless of how on_jetson was decided.
    """
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(_FAKE_MEMINFO_FULL, encoding="utf-8")
    fake_marker = tmp_path / "fake_nv_tegra_release"
    fake_marker.write_text("# R36 (release), REVISION: 4.7 (test fixture)\n", encoding="utf-8")

    result = _run_smoke(
        "system",
        env_extra={
            "MOUSEDROID_SMOKE_FORCE_PLATFORM": "jetson",
            "MOUSEDROID_SMOKE_TEGRA_RELEASE_PATH": str(fake_marker),
            "MOUSEDROID_SMOKE_MEMINFO_PATH": str(meminfo),
        },
    )

    assert str(fake_marker) in result.stdout, (
        "log echo should report the actual tegra path that was configured, "
        "not the hardcoded /etc/nv_tegra_release default"
    )
    # /etc/nv_tegra_release is NOT what was checked — it must not appear in
    # the human-readable echo when an override is set.
    assert "/etc/nv_tegra_release" not in result.stdout


@_BASH_AVAILABLE
def test_memory_check_logs_warn_when_falling_back_to_memfree(tmp_path: Path) -> None:
    """The MemFree fallback must announce itself on stderr — silent
    degradation makes 'why is memory% high?' debugging impossible.
    """
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(_FAKE_MEMINFO_NO_MEMAVAILABLE, encoding="utf-8")

    result = _run_smoke(
        "system",
        env_extra={
            "MOUSEDROID_SMOKE_FORCE_PLATFORM": "non-jetson",
            "MOUSEDROID_SMOKE_MEMINFO_PATH": str(meminfo),
        },
    )
    assert "WARN: MemAvailable not present" in result.stderr, (
        "MemFree fallback must emit a WARN to stderr"
    )
    assert "PASS: memory check" in result.stdout


@_BASH_AVAILABLE
def test_thermal_path_overridable_for_tests(tmp_path: Path) -> None:
    """``MOUSEDROID_SMOKE_THERMAL_PATH`` override allows tests to point at a
    deterministic fixture instead of /sys/...thermal_zone0/temp. Without this
    override the path was hardcoded — flagged as M-3 in the scan. Force-jetson
    keeps the test platform-portable."""
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(_FAKE_MEMINFO_FULL, encoding="utf-8")
    thermal = tmp_path / "thermal_temp"
    thermal.write_text("42500\n", encoding="utf-8")  # 42.5°C

    result = _run_smoke(
        "system",
        env_extra={
            "MOUSEDROID_SMOKE_FORCE_PLATFORM": "jetson",
            "MOUSEDROID_SMOKE_MEMINFO_PATH": str(meminfo),
            "MOUSEDROID_SMOKE_THERMAL_PATH": str(thermal),
        },
    )
    # The thermal subcheck should PASS — file present + readable parses cleanly.
    # The temperature *value* echo depends on `bc` being available (which Git
    # Bash on Windows lacks), so we only assert the PASS marker, not the
    # specific computed temperature. On a real Jetson the formatted "42.5 C"
    # would also appear.
    assert "PASS: thermal sensor read" in result.stdout, result.stdout


@_BASH_AVAILABLE
def test_multi_config_csv_does_not_break_system_subcommand(tmp_path: Path) -> None:
    """``MOUSEDROID_JETSON_CONFIGS`` (CSV of YAML overlays) is parsed at
    script-start. The list isn't used by the ``system`` subcommand, but the
    parsing logic must not corrupt arg passing or fail on whitespace.
    """
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(_FAKE_MEMINFO_FULL, encoding="utf-8")
    # Three commas with whitespace + one empty entry — exercises the trimmer.
    csv = " /etc/mousedroid/a.yaml , /etc/mousedroid/b.yaml ,, /etc/mousedroid/c.yaml "

    result = _run_smoke(
        "system",
        env_extra={
            "MOUSEDROID_SMOKE_FORCE_PLATFORM": "non-jetson",
            "MOUSEDROID_SMOKE_MEMINFO_PATH": str(meminfo),
            "MOUSEDROID_JETSON_CONFIGS": csv,
        },
    )
    # Script must still run cleanly — the CSV parse loop is at script-top so
    # any bug there breaks every subcommand.
    assert "Smoke Test Summary" in result.stdout
    assert "SKIP: torch.cuda.is_available" in result.stdout
    assert result.returncode == 0
