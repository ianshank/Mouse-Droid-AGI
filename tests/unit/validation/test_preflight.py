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


def test_detect_csi_ribbon_disconnect_returns_none_when_video_node_present() -> None:
    """``/dev/video*`` present → not a ribbon issue."""
    from mousedroid.validation.preflight import _detect_csi_ribbon_disconnect

    assert (
        _detect_csi_ribbon_disconnect(
            video_nodes=["/dev/video0"],
            modules_text="imx708 20480 0 - Live 0xfff0000000\n",
        )
        is None
    )


def test_detect_csi_ribbon_disconnect_flags_loaded_imx_module() -> None:
    """Sensor module loaded + no /dev/video* → returns ribbon-disconnect diagnostic."""
    from mousedroid.validation.preflight import _detect_csi_ribbon_disconnect

    msg = _detect_csi_ribbon_disconnect(
        video_nodes=[],
        modules_text="imx708 20480 0 - Live 0xfff0000000\nother_mod 1024 0\n",
    )
    assert msg is not None
    assert "imx708" in msg
    assert "ribbon" in msg.lower()


def test_detect_csi_ribbon_disconnect_silent_when_no_sensor_module() -> None:
    """No sensor module loaded → not a ribbon issue; return None."""
    from mousedroid.validation.preflight import _detect_csi_ribbon_disconnect

    assert (
        _detect_csi_ribbon_disconnect(
            video_nodes=[],
            modules_text="other_mod 1024 0\nyet_another 2048 0\n",
        )
        is None
    )


def test_detect_csi_ribbon_disconnect_flags_ov_and_ar0_prefixes() -> None:
    """Other Jetson sensor prefixes (ov*, ar0*) also surface the diagnostic."""
    from mousedroid.validation.preflight import _detect_csi_ribbon_disconnect

    msg = _detect_csi_ribbon_disconnect(
        video_nodes=[],
        modules_text="ov5693 16384 0\nar0234 24576 0\n",
    )
    assert msg is not None
    assert "ov5693" in msg or "ar0234" in msg


@pytest.mark.asyncio
async def test_check_camera_returns_warn_when_ribbon_disconnect_detected() -> None:
    """Real-hardware path: ribbon-disconnect detector hit → WARN, not FAIL.

    Pins the operator-actionable WARN signal end-to-end (the unit tests
    above prove ``_detect_csi_ribbon_disconnect`` returns the right
    string; this test proves ``_check_camera`` surfaces it as WARN with
    the message body intact rather than swallowing it or returning FAIL).
    """
    from mousedroid.validation import preflight as _p

    # Construct via ``mock_hardware=True`` (bypasses the "at least one
    # distance sensor required" root validator) then flip the flag so
    # ``_check_camera`` exercises the real-hardware branch.
    cfg = Settings(mock_hardware=True)
    cfg.mock_hardware = False

    with unittest.mock.patch.object(
        _p,
        "_detect_csi_ribbon_disconnect",
        return_value="CSI ribbon appears disconnected: imx708 loaded but no /dev/video*",
    ):
        result = await _p._check_camera(cfg)

    assert result.name == "camera"
    assert result.status == PreflightStatus.WARN
    assert "imx708" in result.detail
    assert "ribbon" in result.detail.lower()


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
