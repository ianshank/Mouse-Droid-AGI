"""Unit tests for the SUMMARY.md render shim (F-018, WS-4.2)."""

from __future__ import annotations

from pathlib import Path

from tests._script_loader import load_script_module

_render = load_script_module("render_validation_summary")


def _write_inputs(tmp_path: Path, *, with_trend: bool) -> tuple[Path, Path, Path]:
    results = tmp_path / "results.psv"
    results.write_text("PASS|preflight (real)|\nWARN|serial smoke|dead ESP32\n", encoding="utf-8")
    log = tmp_path / "phase2_preflight.log"
    log_text = "json...\n"
    if with_trend:
        log_text += "Trend: 1 regression(s) vs previous run:\n  - check 'camera' newly FAILing\n"
    log.write_text(log_text, encoding="utf-8")
    out = tmp_path / "SUMMARY.md"
    return results, log, out


def _run(results: Path, log: Path | None, out: Path) -> int:
    argv = [
        "--results-file",
        str(results),
        "--stamp",
        "20260703T000000Z",
        "--repo",
        "/opt/mousedroid",
        "--config",
        "config/jetson_production.yaml",
        "--telemetry-url",
        "http://127.0.0.1:8080",
        "--out",
        str(out),
    ]
    if log is not None:
        argv.extend(["--preflight-log", str(log)])
    return _render.main(argv)


def test_renders_summary_with_trend(tmp_path: Path) -> None:
    results, log, out = _write_inputs(tmp_path, with_trend=True)
    assert _run(results, log, out) == 0
    text = out.read_text(encoding="utf-8")
    assert "| WARN | serial smoke | dead ESP32 |" in text
    assert "newly FAILing" in text


def test_absent_trend_yields_placeholder(tmp_path: Path) -> None:
    results, log, out = _write_inputs(tmp_path, with_trend=False)
    assert _run(results, log, out) == 0
    assert "no trend data recorded this run" in out.read_text(encoding="utf-8")


def test_missing_log_file_is_tolerated(tmp_path: Path) -> None:
    results, _, out = _write_inputs(tmp_path, with_trend=False)
    assert _run(results, tmp_path / "absent.log", out) == 0
    assert out.is_file()


def test_no_log_flag_is_tolerated(tmp_path: Path) -> None:
    results, _, out = _write_inputs(tmp_path, with_trend=False)
    assert _run(results, None, out) == 0
    assert "## Trend" in out.read_text(encoding="utf-8")
