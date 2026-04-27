"""Structural tests for the Jetson full smoke harness shell wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

_HARNESS = Path("scripts/jetson_full_smoke_run.sh")


@pytest.fixture(scope="module")
def harness_text() -> str:
    if not _HARNESS.exists():
        pytest.skip(f"{_HARNESS} not present in this checkout")
    return _HARNESS.read_text(encoding="utf-8")


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
    assert 'voice_remediation' in harness_text
    assert 'voice.tts_model_path' in harness_text