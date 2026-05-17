"""Smoke-pass: validation.preflight.run_preflight unit tests."""

from __future__ import annotations

import unittest.mock

import pytest

from mousedroid.config.schema import Settings
from mousedroid.validation.preflight import (
    PreflightCheckResult,
    PreflightReport,
    PreflightStatus,
    run_preflight,
)


@pytest.mark.asyncio
async def test_run_preflight_returns_report_with_all_checks_listed() -> None:
    """Every preflight check appears in the report, even on mock_hardware."""
    cfg = Settings(mock_hardware=True)
    report = await run_preflight(cfg)
    assert isinstance(report, PreflightReport)
    expected_checks = {"camera", "microphone", "speaker", "lidar", "esp32", "config"}
    assert expected_checks.issubset({c.name for c in report.checks})


@pytest.mark.asyncio
async def test_run_preflight_ok_when_all_checks_pass() -> None:
    """All-OK checks yield ``PreflightStatus.OK`` for the report."""
    cfg = Settings(mock_hardware=True)
    report = await run_preflight(cfg)
    assert report.overall_status == PreflightStatus.OK


def test_overall_status_degraded_on_any_warn() -> None:
    """One WARN check → overall DEGRADED (not FAIL)."""
    report = PreflightReport(
        checks=[
            PreflightCheckResult(
                name="camera",
                status=PreflightStatus.OK,
                detail="ok",
                elapsed_s=0.01,
            ),
            PreflightCheckResult(
                name="lidar",
                status=PreflightStatus.WARN,
                detail="low coverage",
                elapsed_s=0.02,
            ),
        ],
    )
    assert report.overall_status == PreflightStatus.DEGRADED


def test_overall_status_failed_on_any_fail() -> None:
    """One FAIL check → overall FAIL regardless of other passes."""
    report = PreflightReport(
        checks=[
            PreflightCheckResult(
                name="camera",
                status=PreflightStatus.OK,
                detail="ok",
                elapsed_s=0.01,
            ),
            PreflightCheckResult(
                name="esp32",
                status=PreflightStatus.FAIL,
                detail="no device",
                elapsed_s=0.0,
            ),
        ],
    )
    assert report.overall_status == PreflightStatus.FAIL


@pytest.mark.asyncio
async def test_run_preflight_per_check_exception_swallowed_as_fail() -> None:
    """A check helper raising mid-run becomes a FAIL entry, not a bubbled exception."""
    cfg = Settings(mock_hardware=True)
    from mousedroid.validation import preflight as _p

    async def _boom(_cfg: Settings) -> PreflightCheckResult:
        raise RuntimeError("simulated camera driver crash")

    patched = {**_p._CHECK_DISPATCH, "camera": _boom}
    with unittest.mock.patch.object(_p, "_CHECK_DISPATCH", patched):
        report = await run_preflight(cfg)

    camera_result = next(c for c in report.checks if c.name == "camera")
    assert camera_result.status == PreflightStatus.FAIL
    assert "RuntimeError" in camera_result.detail


@pytest.mark.asyncio
async def test_run_preflight_filter_by_check_names() -> None:
    """``check_names`` filter restricts the dispatch to the requested subset."""
    cfg = Settings(mock_hardware=True)
    report = await run_preflight(cfg, check_names={"camera", "esp32"})
    assert {c.name for c in report.checks} == {"camera", "esp32"}


@pytest.mark.asyncio
async def test_run_preflight_records_elapsed_per_check() -> None:
    """Each check captures elapsed wall-clock seconds (observability)."""
    cfg = Settings(mock_hardware=True)
    report = await run_preflight(cfg)
    assert all(c.elapsed_s >= 0.0 for c in report.checks)
    # Total elapsed is bounded below by the sum of per-check elapsed times
    # (checks run sequentially today). Small epsilon for clock jitter.
    assert report.total_elapsed_s + 0.001 >= sum(c.elapsed_s for c in report.checks)


def test_preflight_report_render_text_includes_overall_status() -> None:
    """``PreflightReport.render_text()`` includes overall status + per-check lines."""
    report = PreflightReport(
        checks=[
            PreflightCheckResult(
                name="camera",
                status=PreflightStatus.OK,
                detail="ok",
                elapsed_s=0.01,
            ),
        ],
    )
    text = report.render_text()
    assert "ok" in text.lower()
    assert "camera" in text
