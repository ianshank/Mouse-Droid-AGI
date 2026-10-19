"""Structural tests for the Jetson full smoke harness shell wrapper and its stage script.

Text pins, so they run on every platform. The behaviour behind them is driven for real in
``test_jetson_smoke_orchestrator.py`` (the stage script against a stubbed interpreter and a
fake ``serial``), which needs bash and so skips on Windows.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_HARNESS = Path("scripts/jetson_full_smoke_run.sh")
_SMOKE = Path("scripts/jetson_smoke_test.sh")


@pytest.fixture(scope="module")
def harness_text() -> str:
    if not _HARNESS.exists():
        pytest.skip(f"{_HARNESS} not present in this checkout")
    return _HARNESS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def smoke_text() -> str:
    if not _SMOKE.exists():
        pytest.skip(f"{_SMOKE} not present in this checkout")
    return _SMOKE.read_text(encoding="utf-8")


def _bash_function(text: str, name: str) -> str:
    """Code of the top-level shell function ``name``: its opening line to the first bare ``}``.

    Whole-line ``#`` comments are dropped, so prose that mentions a token cannot satisfy a pin
    on the code that must contain it.
    """
    lines = text.splitlines()
    start = lines.index(f"{name}() {{")
    body = lines[start : lines.index("}", start)]
    return "\n".join(line for line in body if not line.lstrip().startswith("#"))


def test_harness_defines_blocking_override_resolution(harness_text: str) -> None:
    assert "resolve_blocking()" in harness_text
    assert "MOUSEDROID_SMOKE_BLOCKING_${upper}" in harness_text
    assert "invalid; using default" in harness_text


def test_harness_exports_stage_timeout_to_wrapped_commands(harness_text: str) -> None:
    assert "MOUSEDROID_SMOKE_STAGE_TIMEOUT" in harness_text
    assert "timeout --signal=INT --kill-after=10" in harness_text
    assert "timeout --signal=INT --kill-after=5" in harness_text


def test_harness_marks_nonblocking_timeouts_as_expected_fail(harness_text: str) -> None:
    assert 'why="rc=${rc} (non-blocking)"' in harness_text
    assert 'why="rc=${rc} (timeout after ${tmo}s, non-blocking)"' in harness_text
    assert 'record "${label}" "EXPECTED-FAIL" "${why}"' in harness_text


def test_harness_summary_includes_stage_table(harness_text: str) -> None:
    assert 'echo "| Stage | Status | Note |"' in harness_text
    assert 'echo "|-------|--------|------|"' in harness_text
    assert 'for entry in "${RESULTS[@]}"; do' in harness_text


def test_harness_summary_includes_voice_remediation(harness_text: str) -> None:
    assert 'echo "## Rocky voice prerequisites"' in harness_text
    assert "voice_remediation" in harness_text
    assert "voice.tts_model_path" in harness_text


# ---------------------------------------------------------------------------
# F-059 Task 4.1: the serial stage is an exclusive probe
# ---------------------------------------------------------------------------


def test_serial_stage_opens_the_port_exclusively(smoke_text: str) -> None:
    """The stage's inline probe takes the lock the driver takes, so a held port FAILs fast.

    Without ``exclusive=True`` it opens a port the orchestrator already owns, writes into the
    shared UART and reads half of the frames: a live board then looks silent or garbled.
    """
    code = _bash_function(smoke_text, "test_serial")
    assert re.search(r"serial\.Serial\(.*\bexclusive=True\b", code), code


def test_serial_stage_fails_on_silence(smoke_text: str) -> None:
    """REVERSAL: a silent board used to record ``PASS: serial open/write (no response: ...)``.

    That pass could not fail: the stage opened the port, wrote, got nothing back and said the
    port opened, which is true of a board with no firmware, a mismatched baud or no board at
    all. A probe needs a reply, so silence is now a FAIL and the ``WARN:`` path is gone.
    """
    code = _bash_function(smoke_text, "test_serial")
    assert "FAIL:No response from ESP32" in code, code
    assert "WARN:" not in code, code
    assert "serial open/write" not in code, code
    assert code.count("record_pass") == 1, "only a reply may pass the serial stage"
