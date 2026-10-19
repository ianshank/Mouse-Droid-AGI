"""Unit: ``scripts/flash_esp32.sh`` verifies every firmware part before it writes.

The script is the repair path for a WAVE ROVER whose ESP32 lost its stock
``General_Driver`` image. No firmware is committed (``*.bin`` is gitignored), so
the tracked ``deployments/esp32-firmware.json`` pins each of the four parts by
size and sha256, and the script must refuse -- before ``esptool.py`` is ever
invoked -- if any staged file differs from the record. Dry run is the default;
nothing is written without ``--write``.

These tests drive the real script against a throwaway firmware directory, a
throwaway record, and an ``esptool.py`` shim at the front of ``PATH`` that appends
its argv to a marker file (and fails on request, see ``ESPTOOL_SHIM_FAIL``). A test
that wants to prove "esptool never ran" asserts the marker file does not exist. No
hardware and no application code are involved, which makes this the unit tier per
``.claude/skills/test-tier-mirror/SKILL.md``.

``/dev/null`` is a character device, so ``--port /dev/null --write`` clears the
script's ``-c`` serial-port check without any adapter attached.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests._bash import requires_bash

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "flash_esp32.sh"
_RECORD = _REPO / "deployments" / "esp32-firmware.json"

# Skipped on Windows, where ``bash`` is the WSL shim (see tests/_bash.py for why
# ``which("bash")`` alone fails open there), and whenever the script or the record
# is absent -- a rename is caught by the path-asserting tests, not by every consumer.
pytestmark = requires_bash("python3", paths=(_SCRIPT, _RECORD))

_PARTS = (
    ("bootloader", "0x1000", b"B" * 32),
    ("partitions", "0x8000", b"P" * 16),
    ("boot_app0", "0xE000", b"A" * 8),
    ("app", "0x10000", b"F" * 64),
)

_Staged = tuple[Path, Path, Path, dict[str, str]]

# The tracked record, pinned in full. The sha256 values were computed from the Waveshare factory
# package (flash_download_tool_3.9.5/bin), so editing the record means editing this on purpose.
_FACTORY_PARTS = (
    {
        "name": "bootloader",
        "file": "General_Driver.ino.bootloader.bin",
        "offset": "0x1000",
        "size": 18992,
        "sha256": "644de0067047e22380034b8989c39e5d2882f7538c698788866ca5130427322e",
    },
    {
        "name": "partitions",
        "file": "General_Driver.ino.partitions.bin",
        "offset": "0x8000",
        "size": 3072,
        "sha256": "148b959cbff1c38aa8e1d5c0ba9d612c54997b945e56a63f41223eef650653a1",
    },
    {
        "name": "boot_app0",
        "file": "boot_app0.bin",
        "offset": "0xE000",
        "size": 8192,
        "sha256": "f94c5d786a7a8fab06ac5d10e33bf37711a6697636dc037559ea19cc410a17f0",
    },
    {
        "name": "app",
        "file": "General_Driver.ino.bin",
        "offset": "0x10000",
        "size": 986928,
        "sha256": "ebedf216d23402381a16f44373721c8f2d90dfe6ef020cb1f33f73c8f95ee4a4",
    },
)

# Executables the script itself runs. A PATH made only of these has no esptool.py and no pip.
_SCRIPT_TOOLS = (
    "bash",
    "env",
    "python3",
    "dirname",
    "awk",
    "sha256sum",
    "shasum",
    "wc",
    "tr",
    "sed",
)

# Logs its argv to the marker file. ESPTOOL_SHIM_FAIL=1 fails every call; any other non-empty
# value fails only the calls whose argv contains that word (for example verify_flash).
_ESPTOOL_SHIM = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "$ESPTOOL_MARKER"
case "${ESPTOOL_SHIM_FAIL:-}" in
    "") exit 0 ;;
    1) exit 1 ;;
esac
case " $* " in *" ${ESPTOOL_SHIM_FAIL} "*) exit 1 ;; esac
exit 0
"""

# Stands in for python3 via MOUSEDROID_PYTHON: the validator call exits 0 but prints one part.
_FAKE_PYTHON = r"""#!/usr/bin/env bash
if [ "$1" = "-c" ]; then echo 115200; exit 0; fi
echo "0x1000 bootloader.bin 0000000000000000000000000000000000000000000000000000000000000000 32"
"""


@pytest.fixture
def staged(tmp_path: Path) -> _Staged:
    """A staged firmware dir + matching record + esptool shim.

    Returns:
        ``(bin_dir, record, marker, env)``. ``marker`` exists only if the shim ran.
    """
    bin_dir = tmp_path / "firmware"
    bin_dir.mkdir()
    parts = []
    for name, offset, payload in _PARTS:
        (bin_dir / f"{name}.bin").write_bytes(payload)
        parts.append(
            {
                "name": name,
                "file": f"{name}.bin",
                "offset": offset,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    record = tmp_path / "record.json"
    record.write_text(
        json.dumps({"platform": "esp32", "connect_baud": 115200, "parts": parts}), encoding="utf-8"
    )
    shims = tmp_path / "shims"
    shims.mkdir()
    marker = tmp_path / "esptool-called.txt"
    shim = shims / "esptool.py"
    shim.write_text(_ESPTOOL_SHIM, encoding="utf-8")
    shim.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{shims}{os.pathsep}{os.environ.get('PATH', '')}",
        "ESPTOOL_MARKER": str(marker),
    }
    return bin_dir, record, marker, env


def _run(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=_REPO,
    )


def _args(staged: _Staged, *flags: str, port: str = "/dev/null") -> list[str]:
    """Command line for the staged set; put ``--write`` and friends in ``flags``."""
    bin_dir, record, _marker, _env = staged
    return ["--port", port, "--bin-dir", str(bin_dir), "--record", str(record), *flags]


def _printed_argv(stdout: str, kind: str) -> str:
    """The esptool argv the script printed for ``kind`` (write or verify), minus the program."""
    match = re.search(rf"^esptool argv \({kind}\):\s+esptool\.py (.*)$", stdout, flags=re.MULTILINE)
    assert match, f"no printed {kind} argv in: {stdout}"
    return match.group(1)


def test_dry_run_verifies_every_part_and_prints_the_factory_argv(staged: _Staged) -> None:
    bin_dir, record, marker, env = staged
    r = _run(env, "--port", "/dev/ttyUSB0", "--bin-dir", str(bin_dir), "--record", str(record))
    assert r.returncode == 0, r.stderr
    assert "verified app.bin @ 0x10000" in r.stdout
    expected = (
        f"--chip esp32 --port /dev/ttyUSB0 --baud 115200 --before default_reset --after no_reset "
        f"write_flash --flash_mode keep --flash_freq keep --flash_size keep --compress "
        f"0x1000 {bin_dir}/bootloader.bin 0x8000 {bin_dir}/partitions.bin "
        f"0xE000 {bin_dir}/boot_app0.bin 0x10000 {bin_dir}/app.bin"
    )
    assert expected in r.stdout
    assert "verify_flash 0x1000" in r.stdout
    assert "DRY RUN" in r.stdout
    assert not marker.exists(), "dry run must never invoke esptool"


def test_a_hash_mismatch_refuses_before_any_esptool_call(staged: _Staged) -> None:
    bin_dir, record, marker, env = staged
    (bin_dir / "app.bin").write_bytes(b"X" + b"F" * 63)  # same size, different bytes
    r = _run(
        env, "--port", "/dev/null", "--bin-dir", str(bin_dir), "--record", str(record), "--write"
    )
    assert r.returncode != 0
    assert "sha256 mismatch for app.bin" in r.stderr
    assert not marker.exists(), "esptool ran despite a hash mismatch"


def test_write_runs_write_flash_then_verify_flash(staged: _Staged) -> None:
    _bin_dir, _record, marker, env = staged
    r = _run(env, *_args(staged, "--write"))
    assert r.returncode == 0, r.stderr
    calls = marker.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert " write_flash " in calls[0]
    assert " verify_flash " in calls[1]
    # What the operator was shown is exactly what ran.
    assert calls == [_printed_argv(r.stdout, "write"), _printed_argv(r.stdout, "verify")]


def test_the_tracked_record_is_the_factory_image_set() -> None:
    rec = json.loads(_RECORD.read_text(encoding="utf-8"))
    assert [p["offset"] for p in rec["parts"]] == ["0x1000", "0x8000", "0xE000", "0x10000"]
    assert all(re.fullmatch(r"[0-9a-f]{64}", p["sha256"]) for p in rec["parts"])
    assert rec["connect_baud"] == 115200
    assert rec["parts"][3]["sha256"].startswith("ebedf216d23402381a16f44373721c8f")
    assert any("combine/target.bin" in n for n in rec["never_flash"])


def test_script_parses_and_is_strict_mode() -> None:
    assert subprocess.run(["bash", "-n", str(_SCRIPT)], capture_output=True).returncode == 0
    assert "set -euo pipefail" in _SCRIPT.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Refusal paths. The record names four parts; the script must check all four,
# and each way a part can be wrong must stop it before esptool runs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [name for name, _, _ in _PARTS])
def test_every_part_is_hash_checked_not_just_the_last(staged: _Staged, name: str) -> None:
    bin_dir, record, marker, env = staged
    original = (bin_dir / f"{name}.bin").read_bytes()
    (bin_dir / f"{name}.bin").write_bytes(b"X" + original[1:])  # same size, different bytes
    r = _run(
        env, "--port", "/dev/null", "--bin-dir", str(bin_dir), "--record", str(record), "--write"
    )
    assert r.returncode != 0
    assert f"sha256 mismatch for {name}.bin" in r.stderr
    assert not marker.exists(), f"esptool ran despite a corrupted {name}.bin"


def test_a_missing_part_refuses_before_any_esptool_call(staged: _Staged) -> None:
    bin_dir, record, marker, env = staged
    (bin_dir / "boot_app0.bin").unlink()
    r = _run(
        env, "--port", "/dev/null", "--bin-dir", str(bin_dir), "--record", str(record), "--write"
    )
    assert r.returncode != 0
    assert "part missing" in r.stderr
    assert "boot_app0.bin" in r.stderr
    assert not marker.exists(), "esptool ran with a part missing"


def test_a_size_mismatch_refuses_before_any_esptool_call(staged: _Staged) -> None:
    bin_dir, record, marker, env = staged
    (bin_dir / "partitions.bin").write_bytes(b"P" * 17)  # one byte longer than the record says
    r = _run(
        env, "--port", "/dev/null", "--bin-dir", str(bin_dir), "--record", str(record), "--write"
    )
    assert r.returncode != 0
    assert "size mismatch for partitions.bin" in r.stderr
    assert not marker.exists(), "esptool ran despite a size mismatch"


def test_write_refuses_a_port_that_is_not_a_serial_device(staged: _Staged) -> None:
    bin_dir, record, marker, env = staged
    r = _run(
        env,
        "--port",
        str(bin_dir / "no-such-tty"),
        "--bin-dir",
        str(bin_dir),
        "--record",
        str(record),
        "--write",
    )
    assert r.returncode != 0
    assert "serial port not found" in r.stderr
    assert not marker.exists(), "esptool ran against a port that is not a character device"


# ---------------------------------------------------------------------------
# Interface: the documented flags.
# ---------------------------------------------------------------------------


def test_baud_override_reaches_the_argv(staged: _Staged) -> None:
    bin_dir, record, _marker, env = staged
    r = _run(
        env,
        "--port",
        "/dev/ttyUSB0",
        "--bin-dir",
        str(bin_dir),
        "--record",
        str(record),
        "--baud",
        "460800",
    )
    assert r.returncode == 0, r.stderr
    assert "--baud 460800" in r.stdout
    assert "--baud 115200" not in r.stdout


def test_help_prints_the_usage_header_and_stops_before_the_code(staged: _Staged) -> None:
    """``--help`` slices the comment header of the script; keep the slice on the header."""
    _bin_dir, _record, marker, env = staged
    r = _run(env, "--help")
    assert r.returncode == 0, r.stderr
    assert "Usage: bash scripts/flash_esp32.sh --port" in r.stdout
    assert "--write" in r.stdout
    assert "--dry-run" in r.stdout
    assert "set -euo pipefail" not in r.stdout, "--help ran past the comment header into the code"
    assert not marker.exists()


# ---------------------------------------------------------------------------
# The record validator is Python run through a command substitution. The script
# must honour its exit status, not merely count the lines it managed to print:
# `mapfile < <(python ...) || die` tests the status of mapfile, never of Python,
# so only the part count stood between a malformed record and a partial flash.
# ---------------------------------------------------------------------------


def test_a_record_the_validator_rejects_dies_before_any_esptool_call(staged: _Staged) -> None:
    """The validator prints three parts, then rejects the fourth (upper-case sha256)."""
    _bin_dir, record, marker, env = staged
    rec = json.loads(record.read_text(encoding="utf-8"))
    rec["parts"][3]["sha256"] = rec["parts"][3]["sha256"].upper()
    record.write_text(json.dumps(rec), encoding="utf-8")
    r = _run(env, *_args(staged, "--write"))
    assert r.returncode != 0
    assert "bad sha256 for app" in r.stderr
    assert "could not read parts" in r.stderr
    assert not marker.exists(), "esptool ran with a record the validator rejected"


def test_a_three_part_record_dies_in_the_validator_before_any_esptool_call(
    staged: _Staged,
) -> None:
    _bin_dir, record, marker, env = staged
    rec = json.loads(record.read_text(encoding="utf-8"))
    rec["parts"].pop()
    record.write_text(json.dumps(rec), encoding="utf-8")
    r = _run(env, *_args(staged, "--write"))
    assert r.returncode != 0
    assert "record must list exactly 4 parts, got 3" in r.stderr
    assert "could not read parts" in r.stderr
    assert not marker.exists(), "esptool ran with a three-part record"


def test_the_part_count_guard_holds_even_if_the_validator_is_bypassed(
    staged: _Staged, tmp_path: Path
) -> None:
    """Defence in depth: an interpreter that exits 0 but prints one part still cannot flash."""
    _bin_dir, _record, marker, env = staged
    fake = tmp_path / "fake-python"
    fake.write_text(_FAKE_PYTHON, encoding="utf-8")
    fake.chmod(0o755)
    r = _run({**env, "MOUSEDROID_PYTHON": str(fake)}, *_args(staged, "--write"))
    assert r.returncode != 0
    assert "expected 4 parts" in r.stderr
    assert not marker.exists(), "esptool ran after a short parts list"


# ---------------------------------------------------------------------------
# Command line: --dry-run is sticky, and a flag missing its value says so.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [("--dry-run", "--write"), ("--write", "--dry-run")],
    ids=["dry-run-first", "write-first"],
)
def test_dry_run_wins_over_write_in_either_order(staged: _Staged, flags: tuple[str, ...]) -> None:
    _bin_dir, _record, marker, env = staged
    r = _run(env, *_args(staged, *flags))
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout
    assert not marker.exists(), "--write overrode --dry-run"


@pytest.mark.parametrize("flag", ["--port", "--bin-dir", "--record", "--baud"])
def test_a_trailing_value_flag_names_itself(staged: _Staged, flag: str) -> None:
    _bin_dir, _record, marker, env = staged
    r = _run(env, flag)
    assert r.returncode != 0
    assert f"{flag} needs a value" in r.stderr
    assert "unbound variable" not in r.stderr
    assert not marker.exists()


# ---------------------------------------------------------------------------
# After the verification: a failing esptool must stop the script, and a missing
# esptool must be reported, never installed.
# ---------------------------------------------------------------------------


def test_a_failed_write_flash_stops_before_verify_and_the_success_banner(staged: _Staged) -> None:
    _bin_dir, _record, marker, env = staged
    r = _run({**env, "ESPTOOL_SHIM_FAIL": "1"}, *_args(staged, "--write"))
    assert r.returncode != 0
    calls = marker.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1
    assert " write_flash " in calls[0]
    assert "Flash complete" not in r.stdout


def test_a_failed_verify_flash_is_not_reported_as_success(staged: _Staged) -> None:
    _bin_dir, _record, marker, env = staged
    r = _run({**env, "ESPTOOL_SHIM_FAIL": "verify_flash"}, *_args(staged, "--write"))
    assert r.returncode != 0
    calls = marker.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert " verify_flash " in calls[1]
    assert "Flash complete" not in r.stdout


def test_a_missing_esptool_is_reported_and_never_installed(staged: _Staged, tmp_path: Path) -> None:
    """With no esptool.py on PATH the script says how to install it and installs nothing."""
    _bin_dir, _record, _marker, env = staged
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in _SCRIPT_TOOLS:
        found = shutil.which(name)
        if found:
            (tools / name).symlink_to(found)
    pip_marker = tmp_path / "pip-called.txt"
    for name in ("pip", "pip3"):
        pip = tools / name
        pip.write_text(f'#!/usr/bin/env bash\necho "$*" >> "{pip_marker}"\n', encoding="utf-8")
        pip.chmod(0o755)
    r = _run({**env, "PATH": str(tools)}, *_args(staged, "--write"))
    assert r.returncode != 0
    assert "esptool.py not on PATH" in r.stderr
    assert "pip install esptool" in r.stderr
    assert not pip_marker.exists(), "the script ran pip"


# ---------------------------------------------------------------------------
# The tracked record itself: pinned in full, read by the real script, and its
# esptool block tied to the flags the script passes.
# ---------------------------------------------------------------------------


def test_the_tracked_record_pins_the_exact_factory_image_set() -> None:
    rec = json.loads(_RECORD.read_text(encoding="utf-8"))
    assert rec["parts"] == list(_FACTORY_PARTS)
    assert rec["connect_baud"] == 115200


def test_the_records_esptool_block_matches_the_flags_the_script_passes(staged: _Staged) -> None:
    """The flags live in the script and the record documents them; pin the two together."""
    _bin_dir, _record, _marker, env = staged
    block = json.loads(_RECORD.read_text(encoding="utf-8"))["esptool"]
    assert block["compress"] is True
    assert block["verify"] is True
    r = _run(env, *_args(staged, port="/dev/ttyUSB0"))
    assert r.returncode == 0, r.stderr
    before, after = block["before"], block["after"]
    mode, freq, size = block["flash_mode"], block["flash_freq"], block["flash_size"]
    assert f"--before {before} --after {after} write_flash" in r.stdout
    assert f"write_flash --flash_mode {mode} --flash_freq {freq}" in r.stdout
    assert f"--flash_size {size} --compress" in r.stdout
    assert "verify_flash" in r.stdout


def test_the_script_reads_the_tracked_record_and_stops_at_the_first_missing_part(
    staged: _Staged, tmp_path: Path
) -> None:
    _bin_dir, _record, marker, env = staged
    empty = tmp_path / "empty-firmware"
    empty.mkdir()
    r = _run(
        env, "--port", "/dev/null", "--bin-dir", str(empty), "--record", str(_RECORD), "--write"
    )
    assert r.returncode != 0
    assert f"part missing: {empty}/General_Driver.ino.bootloader.bin" in r.stderr
    assert not marker.exists()
