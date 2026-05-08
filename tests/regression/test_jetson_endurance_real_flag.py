"""Regression tests for the `MOUSEDROID_ENDURANCE_FORCE_REAL` opt-in.

The opt-in is module-level: setting `MOUSEDROID_ENDURANCE_FORCE_REAL=1` at
process start flips `MOUSEDROID_MOCK_HARDWARE=false` *before* the conftest
autouse fixture can default it to "true". These tests pin that behaviour
+ the metrics snapshot path so a future refactor cannot silently regress
either guarantee.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def _reload_endurance(monkeypatch: pytest.MonkeyPatch) -> object:
    """Force-reload the endurance test module under fresh env vars."""
    import tests.performance.test_jetson_endurance as endurance_mod

    return importlib.reload(endurance_mod)


def test_force_real_env_true_flips_mock_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """`=1` flips MOUSEDROID_MOCK_HARDWARE to "false" at module import."""
    monkeypatch.setenv("MOUSEDROID_ENDURANCE_FORCE_REAL", "1")
    monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "true")

    mod = _reload_endurance(monkeypatch)

    assert mod._FORCE_REAL is True
    # The module-level flip should have overwritten the conftest default.
    import os

    assert os.environ["MOUSEDROID_MOCK_HARDWARE"] == "false"


@pytest.mark.parametrize("value", ["true", "yes", "1", "TRUE", "Yes"])
def test_force_real_truthy_values_accepted(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Any case-insensitive truthy spelling enables the opt-in."""
    monkeypatch.setenv("MOUSEDROID_ENDURANCE_FORCE_REAL", value)
    mod = _reload_endurance(monkeypatch)
    assert mod._FORCE_REAL is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_force_real_falsy_values_disable_optin(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Empty / "0" / "false" / "no" all keep the opt-in disabled."""
    monkeypatch.setenv("MOUSEDROID_ENDURANCE_FORCE_REAL", value)
    mod = _reload_endurance(monkeypatch)
    assert mod._FORCE_REAL is False


def test_metrics_dir_default_under_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default metrics dir lives under `reports/endurance` for git tracking."""
    monkeypatch.delenv("MOUSEDROID_ENDURANCE_REPORT_DIR", raising=False)
    mod = _reload_endurance(monkeypatch)
    assert mod._METRICS_DIR.parts[-2:] == ("reports", "endurance")


def test_metrics_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`MOUSEDROID_ENDURANCE_REPORT_DIR` redirects the snapshot path."""
    custom = tmp_path / "custom-endurance"
    monkeypatch.setenv("MOUSEDROID_ENDURANCE_REPORT_DIR", str(custom))
    mod = _reload_endurance(monkeypatch)
    metrics_dir = mod._METRICS_DIR
    assert metrics_dir == custom


def test_metrics_snapshot_writes_valid_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`_write_metrics_snapshot` produces valid JSON with the expected keys."""
    monkeypatch.setenv("MOUSEDROID_ENDURANCE_REPORT_DIR", str(tmp_path))
    monkeypatch.delenv("MOUSEDROID_ENDURANCE_FORCE_REAL", raising=False)
    mod = _reload_endurance(monkeypatch)

    out = mod._write_metrics_snapshot(
        duration_s=60.0,
        p95_ms=12.3,
        p99_ms=18.7,
        rss_start_mb=120.0,
        rss_end_mb=125.0,
        tick_count=1800,
        error_count=0,
        max_gpu_temp_c=58.0,
    )

    assert out is not None
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    for key in (
        "stamp",
        "force_real",
        "duration_s",
        "p95_ms",
        "p99_ms",
        "rss_start_mb",
        "rss_end_mb",
        "tick_count",
        "error_count",
        "max_gpu_temp_c",
    ):
        assert key in payload, f"missing key {key} in metrics snapshot"

    assert payload["force_real"] is False
    assert payload["p95_ms"] == 12.3
    assert payload["tick_count"] == 1800


def test_metrics_snapshot_is_non_fatal_on_oserror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the metrics dir is unwritable, snapshot returns None — never raises."""
    target = tmp_path / "blocked"
    target.write_text("this is a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("MOUSEDROID_ENDURANCE_REPORT_DIR", str(target))

    mod = _reload_endurance(monkeypatch)

    out = mod._write_metrics_snapshot(
        duration_s=1.0,
        p95_ms=1.0,
        p99_ms=2.0,
        rss_start_mb=0.0,
        rss_end_mb=0.0,
        tick_count=10,
        error_count=0,
        max_gpu_temp_c=0.0,
    )
    assert out is None
