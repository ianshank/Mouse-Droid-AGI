"""CLI smoke tests for scripts/check_usbc_devices.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "check_usbc_devices.py"


def _write_config(tmp_path: Path, by_id_root: Path) -> Path:
    cfg = tmp_path / "smoke.yaml"
    # by_id_root must be passed as a POSIX-style path so Pydantic on Linux
    # parses it without backslash-escape issues. Path.as_posix() handles both
    # Windows test hosts and the Jetson target.
    cfg.write_text(
        f"""
mock_hardware: true
usbc_discovery:
  enabled: true
  by_id_root: "{by_id_root.as_posix()}"
  required_endpoints:
    - name: rover_esp32
      by_id_glob: "*CP2102N*-if00-port0"
""".lstrip(),
        encoding="utf-8",
    )
    return cfg


def test_cli_exits_zero_when_all_present(tmp_path: Path) -> None:
    by_id = tmp_path / "by-id"
    by_id.mkdir()
    (by_id / "usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_X-if00-port0").touch()
    cfg = _write_config(tmp_path, by_id)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(cfg), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["rover_esp32"]["status"] == "present"


def test_cli_exits_nonzero_when_required_missing(tmp_path: Path) -> None:
    by_id = tmp_path / "by-id"
    by_id.mkdir()
    cfg = _write_config(tmp_path, by_id)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(cfg)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "rover_esp32" in result.stdout + result.stderr


def test_cli_short_circuits_when_discovery_disabled(tmp_path: Path) -> None:
    cfg = tmp_path / "smoke.yaml"
    cfg.write_text("mock_hardware: true\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(cfg)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usbc_discovery" in result.stderr  # message identifies the gate
    assert "nothing to check" in result.stderr
