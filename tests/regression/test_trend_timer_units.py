"""Regression pins for the trend timer units (F-018, WS-4.3).

The load-bearing contract: the timer-driven preflight must NEVER open an
exclusive device while the orchestrator container runs. The container owns
camera / LiDAR / ESP32 UART / audio; a concurrent open corrupts both readers.
So the unit's default ``--checks`` list is pinned to the non-exclusive subset
and the exclusive check names are pinned OUT of the ExecStart line entirely.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVICE = _REPO_ROOT / "scripts" / "mousedroid-trend.service"
_TIMER = _REPO_ROOT / "scripts" / "mousedroid-trend.timer"

# The only checks a live-rover timer may run (config + plain-file reads).
_SAFE_CHECKS = {"config", "host_env_keys"}
# Exclusive-device checks that must never appear in the timer's ExecStart.
_EXCLUSIVE_CHECKS = ("camera", "lidar", "esp32", "microphone", "speaker")


def _service_text() -> str:
    return _SERVICE.read_text(encoding="utf-8")


class TestServiceUnit:
    def test_unit_files_exist(self) -> None:
        assert _SERVICE.is_file()
        assert _TIMER.is_file()

    def test_oneshot_with_env_file_and_syslog_id(self) -> None:
        text = _service_text()
        assert "Type=oneshot" in text
        assert "EnvironmentFile=-" in text, "missing-env-file must be tolerated (dash prefix)"
        assert "SyslogIdentifier=mousedroid-trend" in text

    def test_default_checks_are_a_safe_subset(self) -> None:
        text = _service_text()
        match = re.search(r"Environment=MOUSEDROID_TREND_CHECKS=(\S+)", text)
        assert match, "the default check list must be declared as an Environment= line"
        declared = set(match.group(1).split(","))
        assert declared <= _SAFE_CHECKS, (
            f"timer default checks {declared} exceed the non-exclusive set "
            f"{_SAFE_CHECKS} - see the unit header contract"
        )

    @pytest.mark.parametrize("check", _EXCLUSIVE_CHECKS)
    def test_exclusive_checks_never_in_exec_start(self, check: str) -> None:
        exec_lines = [line for line in _service_text().splitlines() if line.startswith("ExecStart")]
        assert exec_lines, "service must declare ExecStart"
        for line in exec_lines:
            assert check not in line, (
                f"exclusive check {check!r} in ExecStart - the orchestrator "
                "container owns that device; timer probes would corrupt both readers"
            )

    def test_journal_flags_threaded(self) -> None:
        text = _service_text()
        assert "--journal-path" in text
        assert "--trend" in text
        assert "--journal-max-bytes" in text, "SD-card growth must be capped"

    def test_journal_path_is_separate_from_full_validation(self) -> None:
        # Timer runs (2 fast checks) and full-validation runs (all checks)
        # must not share a journal (bogus elapsed-time trend comparisons).
        # Only directive lines matter — the header prose may reference the
        # full-validation harness for context.
        directives = [
            line
            for line in _service_text().splitlines()
            if line.startswith(("Environment=", "ExecStart"))
        ]
        joined = "\n".join(directives)
        assert "trend/preflight.jsonl" in joined
        assert "jetson_full_validation" not in joined

    def test_no_hardcoded_paths_in_exec_start(self) -> None:
        exec_lines = [
            line for line in _service_text().splitlines() if line.startswith("ExecStart=")
        ]
        for line in exec_lines:
            assert "${MOUSEDROID_INSTALL_DIR}" in line, "ExecStart must use the env indirection"


class TestTimerUnit:
    def test_cadence_declared(self) -> None:
        text = _TIMER.read_text(encoding="utf-8")
        assert "OnBootSec=" in text
        assert "OnUnitActiveSec=" in text
        assert "WantedBy=timers.target" in text
