"""Unit: ``scripts/flash_esp32.sh`` verifies every firmware part before it writes.

The script is the repair path for a WAVE ROVER whose ESP32 lost its stock
``General_Driver`` image. No firmware is committed (``*.bin`` is gitignored), so
the tracked ``deployments/esp32-firmware.json`` pins each of the four parts by
size and sha256, and the script must refuse -- before ``esptool.py`` is ever
invoked -- if any staged file differs from the record. Dry run is the default;
nothing is written without ``--write``.

These tests drive the real script against a throwaway firmware directory, a
throwaway record, and an ``esptool.py`` shim at the front of ``PATH`` that only
appends its argv to a marker file. A test that wants to prove "esptool never ran"
asserts the marker file does not exist. No hardware and no application code are
involved, which makes this the unit tier per
``.claude/skills/test-tier-mirror/SKILL.md``.

``/dev/null`` is a character device, so ``--port /dev/null --write`` clears the
script's ``-c`` serial-port check without any adapter attached.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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
    shim.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "$ESPTOOL_MARKER"\nexit 0\n',
        encoding="utf-8",
    )
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


def test_write_invokes_esptool_with_the_verified_argv(staged: _Staged) -> None:
    bin_dir, record, marker, env = staged
    r = _run(
        env, "--port", "/dev/null", "--bin-dir", str(bin_dir), "--record", str(record), "--write"
    )
    assert r.returncode == 0, r.stderr
    calls = marker.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert " write_flash " in calls[0]
    assert " verify_flash " in calls[1]


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
    assert "set -euo pipefail" not in r.stdout, "--help ran past the comment header into the code"
    assert not marker.exists()
