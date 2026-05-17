"""Smoke-pass: validate_all_pillars dispatcher tests."""

from __future__ import annotations

import unittest.mock

import pytest

from mousedroid.config.schema import Settings
from mousedroid.validation.pillars import (
    PillarReport,
    PillarResult,
    PillarStatus,
    validate_all_pillars,
)


@pytest.mark.asyncio
async def test_validate_all_pillars_returns_report() -> None:
    """The dispatcher returns a ``PillarReport`` (dry-run keeps it fast)."""
    cfg = Settings(mock_hardware=True)
    report = await validate_all_pillars(cfg, dry_run=True)
    assert isinstance(report, PillarReport)


@pytest.mark.asyncio
async def test_dry_run_lists_all_10_pillars_as_skipped() -> None:
    """``dry_run=True`` lists all 10 canonical pillars as SKIPPED entries."""
    cfg = Settings(mock_hardware=True)
    report = await validate_all_pillars(cfg, dry_run=True)
    names = {r.name for r in report.results}
    assert names == {
        "safety",
        "world_model",
        "memory",
        "cognitive",
        "reward",
        "curiosity",
        "continual",
        "meta",
        "scaling",
        "growth",
    }
    assert all(r.status == PillarStatus.SKIPPED for r in report.results)


@pytest.mark.asyncio
async def test_pillar_names_filter_restricts_dispatch() -> None:
    """``pillar_names`` filter only runs the requested subset."""
    cfg = Settings(mock_hardware=True)
    report = await validate_all_pillars(
        cfg,
        pillar_names={"safety", "world_model"},
        dry_run=True,
    )
    assert {r.name for r in report.results} == {"safety", "world_model"}


@pytest.mark.asyncio
async def test_per_pillar_exception_swallowed_as_fail() -> None:
    """A pillar check raising mid-run becomes a FAIL entry, not a bubbled exception."""
    cfg = Settings(mock_hardware=True)
    from mousedroid.validation import pillars as _p

    async def _boom(_cfg: Settings) -> PillarResult:
        raise RuntimeError("simulated pillar crash")

    patched = {**_p._PILLAR_DISPATCH, "safety": _boom}
    with unittest.mock.patch.object(_p, "_PILLAR_DISPATCH", patched):
        report = await validate_all_pillars(cfg, pillar_names={"safety"})

    safety = next(r for r in report.results if r.name == "safety")
    assert safety.status == PillarStatus.FAIL
    assert "RuntimeError" in safety.detail


def test_overall_status_ok_when_all_results_ok() -> None:
    """Aggregate is OK only when every pillar is OK (or SKIPPED)."""
    report = PillarReport(
        results=[
            PillarResult(name="safety", status=PillarStatus.OK, elapsed_s=0.01),
            PillarResult(name="memory", status=PillarStatus.SKIPPED, elapsed_s=0.0),
        ],
    )
    assert report.overall_status == PillarStatus.OK


def test_overall_status_degraded_on_warn() -> None:
    """A WARN entry promotes the aggregate to DEGRADED."""
    report = PillarReport(
        results=[
            PillarResult(name="safety", status=PillarStatus.OK, elapsed_s=0.01),
            PillarResult(name="lidar", status=PillarStatus.WARN, elapsed_s=0.02),
        ],
    )
    assert report.overall_status == PillarStatus.DEGRADED


def test_overall_status_fail_on_any_fail() -> None:
    """A FAIL entry forces aggregate FAIL regardless of others."""
    report = PillarReport(
        results=[
            PillarResult(name="safety", status=PillarStatus.OK, elapsed_s=0.01),
            PillarResult(name="memory", status=PillarStatus.FAIL, elapsed_s=0.01),
        ],
    )
    assert report.overall_status == PillarStatus.FAIL


def test_render_text_includes_overall_and_per_pillar_lines() -> None:
    """``render_text`` emits the overall + every pillar line."""
    report = PillarReport(
        results=[PillarResult(name="safety", status=PillarStatus.OK, detail="ok", elapsed_s=0.01)],
    )
    text = report.render_text()
    assert "overall=ok" in text
    assert "safety" in text


def test_model_dump_json_round_trips() -> None:
    """Pydantic JSON round-trip preserves every PillarResult field."""
    report = PillarReport(
        results=[PillarResult(name="safety", status=PillarStatus.OK, detail="ok", elapsed_s=0.01)],
        total_elapsed_s=0.01,
    )
    decoded = PillarReport.model_validate_json(report.model_dump_json())
    assert decoded == report


@pytest.mark.asyncio
async def test_dispatch_records_elapsed_per_pillar() -> None:
    """Each pillar result captures elapsed wall-clock seconds."""
    cfg = Settings(mock_hardware=True)
    report = await validate_all_pillars(cfg, pillar_names={"safety"}, dry_run=False)
    assert all(r.elapsed_s >= 0.0 for r in report.results)
