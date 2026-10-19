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

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "jetson_smoke_test.sh"

_BASH = shutil.which("bash")
_BASH_AVAILABLE = pytest.mark.skipif(
    _BASH is None,
    reason="bash unavailable on this host",
)


pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "jetson_smoke_test.sh requires Linux bash. On Windows, WSL bash.EXE "
        "can fail with RPC errors (e.g. 0x8007072c)."
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


# ---------------------------------------------------------------------------
# F-059: the motor/power stages stack the bench ESP32 overlay
# ---------------------------------------------------------------------------

# One stub serves both stages, and each stage greps the output for the PASSED line of its own
# test(s): a bare exit 0 proves nothing, because pytest exits 0 when every test skips. Each
# stage's output (SMOKE_STUB_MOTOR_OUTPUT, SMOKE_STUB_POWER_OUTPUT) and the stub's exit status
# (SMOKE_STUB_RC) can be replaced to script a skip or a failure; unset, everything passes.
_STUB_PYTHON = """#!/usr/bin/env bash
if [[ "$1" == "-m" && "$2" == "pytest" && "$3" == "--version" ]]; then echo "pytest 0"; exit 0; fi
printf '%s\\n' "${MOUSEDROID_JETSON_CONFIGS:-}" >> "${SMOKE_STUB_CAPTURE}"
motor_passed="tests/hardware/test_esp32_loopback.py::test_send_velocity_moves_encoders PASSED
tests/hardware/test_esp32_loopback.py::test_emergency_stop_latency PASSED"
power_passed="tests/hardware/test_power_chain_smoke.py::test_power_chain_within_budget PASSED"
echo "${SMOKE_STUB_MOTOR_OUTPUT-${motor_passed}}"
echo "${SMOKE_STUB_POWER_OUTPUT-${power_passed}}"
exit "${SMOKE_STUB_RC:-0}"
"""
_BENCH = f"{_REPO_ROOT}/config/jetson_bench_esp32.yaml"
_PROD = f"{_REPO_ROOT}/config/jetson_production.yaml"


@_BASH_AVAILABLE
@pytest.mark.parametrize(
    ("given", "extra", "expected"),
    [
        ("", {}, f"{_PROD},{_BENCH}"),
        (
            "/etc/mousedroid/a.yaml,/etc/mousedroid/b.yaml",
            {},
            f"/etc/mousedroid/a.yaml,/etc/mousedroid/b.yaml,{_BENCH}",
        ),
        (f"{_PROD},{_BENCH}", {}, f"{_PROD},{_BENCH}"),  # no double append
        # The singular var is the base when the CSV is empty (parity with the Python loaders) ...
        (
            "",
            {"MOUSEDROID_JETSON_CONFIG": "/etc/mousedroid/single.yaml"},
            f"/etc/mousedroid/single.yaml,{_BENCH}",
        ),
        # ... and the CSV wins when both are set (same precedence as tests/_jetson_hardware.py).
        (
            "/etc/mousedroid/a.yaml",
            {"MOUSEDROID_JETSON_CONFIG": "/etc/mousedroid/single.yaml"},
            f"/etc/mousedroid/a.yaml,{_BENCH}",
        ),
        # Any other value of the override var is that file.
        (
            _PROD,
            {"MOUSEDROID_SMOKE_BENCH_ESP32_OVERLAY": "/etc/mousedroid/other.yaml"},
            f"{_PROD},/etc/mousedroid/other.yaml",
        ),
    ],
    ids=[
        "default-base",
        "csv-appended",
        "already-stacked",
        "singular-base",
        "csv-beats-singular",
        "override-path",
    ],
)
@pytest.mark.parametrize(
    ("stage", "pass_line", "stage_env"),
    [
        ("motor", "PASS: motor loopback smoke", {"MOUSEDROID_SMOKE_ALLOW_MOTION": "1"}),
        ("power", "PASS: power chain smoke", {}),
    ],
    ids=["motor", "power"],
)
def test_esp32_stages_stack_the_bench_overlay(
    tmp_path: Path,
    stage: str,
    pass_line: str,
    stage_env: dict[str, str],
    given: str,
    extra: dict[str, str],
    expected: str,
) -> None:
    """Both ESP32-owned stages hand pytest the config stack plus the bench overlay, once."""
    stub = tmp_path / "python"
    stub.write_text(_STUB_PYTHON, encoding="utf-8")
    stub.chmod(0o755)
    capture = tmp_path / "configs.txt"
    result = _run_smoke(
        stage,
        env_extra={
            "MOUSEDROID_SMOKE_PYTHON": str(stub),
            "SMOKE_STUB_CAPTURE": str(capture),
            "MOUSEDROID_JETSON_CONFIGS": given,
            **stage_env,
            **extra,
        },
    )
    assert pass_line in result.stdout, result.stdout
    assert capture.read_text(encoding="utf-8").splitlines() == [expected]


# What the stages record when the operator opts out of the overlay (a run with no chassis).
_OPT_OUT_REASON = (
    "bench ESP32 overlay opted out (MOUSEDROID_SMOKE_BENCH_ESP32_OVERLAY is empty): "
    "no chassis declared, stage not run"
)


@_BASH_AVAILABLE
@pytest.mark.parametrize(
    ("stage", "name", "stage_env"),
    [
        ("motor", "motor loopback smoke", {"MOUSEDROID_SMOKE_ALLOW_MOTION": "1"}),
        ("power", "power chain smoke", {}),
    ],
    ids=["motor", "power"],
)
def test_esp32_stages_skip_when_the_bench_overlay_is_opted_out(
    tmp_path: Path, stage: str, name: str, stage_env: dict[str, str]
) -> None:
    """MOUSEDROID_SMOKE_BENCH_ESP32_OVERLAY set to empty declares "no chassis" (ruling 31).

    The stage records SKIP with that reason and returns before pytest runs: on the production
    stack (esp32.enabled: false) pytest would only measure MockESP32Driver, and with the overlay
    stacked it would FAIL for want of a board. Either is wrong for a run that declared no chassis,
    so a health-only run stays green and honest. The stage never computes a config stack, which
    is why the opt-out has no row in the stacking test above.
    """
    stub = tmp_path / "python"
    stub.write_text(_STUB_PYTHON, encoding="utf-8")
    stub.chmod(0o755)
    capture = tmp_path / "configs.txt"
    result = _run_smoke(
        stage,
        env_extra={
            "MOUSEDROID_SMOKE_PYTHON": str(stub),
            "SMOKE_STUB_CAPTURE": str(capture),
            "MOUSEDROID_SMOKE_BENCH_ESP32_OVERLAY": "",
            **stage_env,
        },
    )
    assert f"SKIP: {name} {_OPT_OUT_REASON}" in result.stdout, result.stdout
    assert f"PASS: {name}" not in result.stdout, result.stdout
    assert "Failed:  0" in result.stdout, result.stdout
    assert not capture.exists(), "pytest ran for an opted-out stage; it must return before it"
    assert result.returncode == 0, result.stdout


# What `pytest -ra -v` prints for the power test when it skips: the run still exits 0.
_POWER_SKIP_REASON = (
    "esp32.enabled=false routes to MockESP32Driver; "
    "stack config/jetson_bench_esp32.yaml or unset MOUSEDROID_ESP32__ENABLED "
    "to probe the board"
)
_POWER_SKIPPED_OUTPUT = (
    "tests/hardware/test_power_chain_smoke.py::test_power_chain_within_budget SKIPPED [100%]\n"
    f"SKIPPED [1] tests/hardware/test_power_chain_smoke.py:21: {_POWER_SKIP_REASON}\n"
    "1 skipped in 0.01s"
)


@_BASH_AVAILABLE
@pytest.mark.parametrize(
    ("power_output", "reason"),
    [
        (_POWER_SKIPPED_OUTPUT, _POWER_SKIP_REASON),
        ("1 deselected in 0.01s", "power chain test did not pass"),
    ],
    ids=["skipped", "neither-passed-nor-skipped"],
)
def test_power_stage_fails_unless_its_test_passes(
    tmp_path: Path, power_output: str, reason: str
) -> None:
    """A clean pytest exit is not a measurement: the power stage needs its test to PASS (R3-2).

    pytest exits 0 when every test skips, and an exported MOUSEDROID_ESP32__ENABLED=false beats
    the bench overlay, so the power test skips with its remedy. The stage must not record PASS:
    it records FAIL and carries the skip reason (or a generic one) into the line.
    """
    stub = tmp_path / "python"
    stub.write_text(_STUB_PYTHON, encoding="utf-8")
    stub.chmod(0o755)
    result = _run_smoke(
        "power",
        env_extra={
            "MOUSEDROID_SMOKE_PYTHON": str(stub),
            "SMOKE_STUB_CAPTURE": str(tmp_path / "configs.txt"),
            "SMOKE_STUB_POWER_OUTPUT": power_output,
        },
    )
    assert f"FAIL: power chain smoke {reason}" in result.stdout, result.stdout
    assert "PASS: power chain smoke" not in result.stdout, result.stdout
    assert result.returncode == 1, result.stdout


# What the motor stage sees when its pytest run exits non-zero. The traceback echoes the failing
# statement's source BEFORE the E line, and the -ra FAILED summary truncates the message, so the
# stage must take the full E line (or the -ra SKIPPED reason), not the first line that matches.
_MOTOR_ASSERT_DETAIL = (
    "no FEEDBACK_BASE_INFO frame after 0.05 m/s on /dev/ttyUSB0 (command_set='legacy') "
    "in 5 poll(s); the encoder-less criterion needs waveshare_stock and a live board"
)
_MOTOR_ASSERTION_OUTPUT = (
    "tests/hardware/test_esp32_loopback.py::test_send_velocity_moves_encoders FAILED [ 50%]\n"
    ">           assert enc.imu_valid, (\n"
    '                f"no FEEDBACK_BASE_INFO frame after {test_vel:.2f} m/s on "\n'
    f"E           AssertionError: {_MOTOR_ASSERT_DETAIL}\n"
    "E           assert False\n"
    "FAILED tests/hardware/test_esp32_loopback.py::test_send_velocity_moves_encoders - "
    "AssertionError: no FEEDBACK_BASE_INFO frame after 0.05 m/s on /dev/ttyUSB0 (command_set=...\n"
    "=== 1 failed, 1 passed in 3.21s ==="
)
_MOTOR_INACTIVE_REASON = (
    "encoder loopback inactive; ensure rover power, motor/encoder wiring and both motion gates"
)
_MOTOR_INACTIVE_OUTPUT = (
    "tests/hardware/test_esp32_loopback.py::test_send_velocity_moves_encoders SKIPPED [ 50%]\n"
    "tests/hardware/test_esp32_loopback.py::test_emergency_stop_latency FAILED [100%]\n"
    f"SKIPPED [1] tests/hardware/test_esp32_loopback.py:160: {_MOTOR_INACTIVE_REASON}\n"
    "=== 1 failed, 1 skipped in 1.04s ==="
)
_MOTOR_NO_DEVICE_REASON = (
    "ESP32 serial device '/dev/ttyUSB0' not found. "
    "Set MOUSEDROID_ESP32_SERIAL_PORT or update esp32.serial_port in config."
)
_MOTOR_NO_DEVICE_OUTPUT = (
    f"SKIPPED [2] tests/hardware/test_esp32_loopback.py:70: {_MOTOR_NO_DEVICE_REASON}\n"
    "=== 1 failed, 2 skipped in 0.92s ==="
)


@_BASH_AVAILABLE
@pytest.mark.parametrize(
    ("motor_output", "detail"),
    [
        (_MOTOR_ASSERTION_OUTPUT, _MOTOR_ASSERT_DETAIL),
        (_MOTOR_INACTIVE_OUTPUT, _MOTOR_INACTIVE_REASON),
        (_MOTOR_NO_DEVICE_OUTPUT, _MOTOR_NO_DEVICE_REASON),
        ("=== 1 failed, 1 passed in 1.00s ===\nE       assert 0.5 <= 0.1", ""),
    ],
    ids=[
        "assertion-e-line",
        "encoder-loopback-inactive",
        "serial-device-missing",
        "no-known-cause",
    ],
)
def test_motor_stage_names_the_cause_when_pytest_exits_nonzero(
    tmp_path: Path, motor_output: str, detail: str
) -> None:
    """A non-zero pytest exit still carries its diagnosis into the FAIL line (review minor 3).

    The reason grep used to run only on a clean exit, where an assertion failure can never land:
    a silent encoder-less board then read as a bare "1 test(s) failed". Without a known cause the
    line stays exactly what it was.
    """
    stub = tmp_path / "python"
    stub.write_text(_STUB_PYTHON, encoding="utf-8")
    stub.chmod(0o755)
    result = _run_smoke(
        "motor",
        env_extra={
            "MOUSEDROID_SMOKE_PYTHON": str(stub),
            "SMOKE_STUB_CAPTURE": str(tmp_path / "configs.txt"),
            "MOUSEDROID_SMOKE_ALLOW_MOTION": "1",
            "SMOKE_STUB_MOTOR_OUTPUT": motor_output,
            "SMOKE_STUB_RC": "1",
        },
    )
    expected = "FAIL: motor loopback smoke 1 test(s) failed" + (f": {detail}" if detail else "")
    assert f"  {expected}\n" in result.stdout, result.stdout
    assert result.returncode == 1, result.stdout


# ---------------------------------------------------------------------------
# F-059 Task 4.1: the serial stage is an exclusive probe, not an open/write check
# ---------------------------------------------------------------------------

# Stand-in for pyserial, importable as ``serial`` ahead of the real one. It records every open
# (so a test sees the keyword arguments the stage really passed) and models the one behaviour
# the stage now depends on: with ``exclusive`` truthy and the port held, pyserial raises
# ``SerialException(errno, "Could not exclusively lock port ...")`` from the constructor
# (serialposix._reconfigure_port, flock(LOCK_EX | LOCK_NB)). SerialException is an IOError there.
_FAKE_SERIAL = r"""
import json
import os


class SerialException(IOError):
    pass


class Serial:
    def __init__(self, port=None, baudrate=9600, timeout=None, exclusive=None, **kwargs):
        record = {"port": port, "baudrate": baudrate, "timeout": timeout, "exclusive": exclusive}
        with open(os.environ["FAKE_SERIAL_LOG"], "a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")
        if exclusive and os.environ.get("FAKE_SERIAL_HELD") == "1":
            raise SerialException(
                11,
                f"Could not exclusively lock port {port}: "
                "[Errno 11] Resource temporarily unavailable",
            )

    def write(self, data):
        return len(data)

    def flush(self):
        pass

    def readline(self):
        return os.environ.get("FAKE_SERIAL_REPLY", "").encode()

    def close(self):
        pass
"""


def _run_serial_stage(
    tmp_path: Path, *, reply: str, held: bool = False
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, object]]]:
    """Run the real inline serial probe, under the test's own interpreter, against the fake port.

    Returns the finished process and every ``serial.Serial(...)`` open the probe made.
    """
    package = tmp_path / "fake_pyserial" / "serial"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(_FAKE_SERIAL, encoding="utf-8")
    node = tmp_path / "ttyFAKE0"
    node.write_text("", encoding="utf-8")  # the stage FAILs first if the device node is absent
    opens_log = tmp_path / "serial_opens.jsonl"
    result = _run_smoke(
        "serial",
        env_extra={
            "MOUSEDROID_SMOKE_PYTHON": sys.executable,
            "PYTHONPATH": str(tmp_path / "fake_pyserial"),
            "MOUSEDROID_ESP32__SERIAL_PORT": str(node),
            "FAKE_SERIAL_LOG": str(opens_log),
            "FAKE_SERIAL_REPLY": reply,
            "FAKE_SERIAL_HELD": "1" if held else "0",
        },
    )
    lines = opens_log.read_text(encoding="utf-8").splitlines() if opens_log.exists() else []
    return result, [json.loads(line) for line in lines]


@_BASH_AVAILABLE
def test_serial_stage_opens_the_port_exclusively_and_passes_on_a_reply(tmp_path: Path) -> None:
    """A live board answers the probe: the stage passes, having opened the port exclusively."""
    result, opens = _run_serial_stage(tmp_path, reply='{"T":1001}')
    assert "PASS: serial send/receive" in result.stdout, result.stdout
    assert [opened["exclusive"] for opened in opens] == [True], opens
    assert result.returncode == 0, result.stdout


@_BASH_AVAILABLE
def test_serial_stage_fails_fast_when_another_opener_holds_the_port(tmp_path: Path) -> None:
    """A port held elsewhere is a FAIL naming the lock, never a pass or a "no response".

    Before the stage opened exclusively it shared the UART with whoever owned it, and the
    outcome depended on who read a frame first.
    """
    result, _ = _run_serial_stage(tmp_path, reply='{"T":1001}', held=True)
    assert "FAIL: serial send/receive" in result.stdout, result.stdout
    assert "Could not exclusively lock port" in result.stdout, result.stdout
    assert "PASS:" not in result.stdout, result.stdout
    assert result.returncode == 1, result.stdout


@_BASH_AVAILABLE
def test_serial_stage_fails_on_silence(tmp_path: Path) -> None:
    """REVERSAL: a board that opens but never answers used to PASS as "serial open/write".

    The stage must be a probe: opening a port proves a device node, not a board. Silence is a
    FAIL whose reason names what to check (command set and baud first).
    """
    result, opens = _run_serial_stage(tmp_path, reply="")
    assert len(opens) == 1, opens
    assert "FAIL: serial send/receive" in result.stdout, result.stdout
    assert "No response from ESP32" in result.stdout, result.stdout
    assert "command_set=" in result.stdout, result.stdout
    assert "baud=" in result.stdout, result.stdout
    assert "serial open/write" not in result.stdout, result.stdout
    assert "PASS:" not in result.stdout, result.stdout
    assert result.returncode == 1, result.stdout
