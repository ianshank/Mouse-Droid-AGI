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


def test_pattern_b_pillar_skips_gracefully_when_pytest_absent() -> None:
    """Pattern-B pillars SKIP (not crash) when pytest isn't installed.

    Pins the production-container fix — ``validate_pillars`` must remain
    importable + invokable on a runtime with no dev extras. Mocks
    ``importlib.util.find_spec`` to simulate pytest absence and verifies
    the pillar lands as SKIPPED with the documented diagnostic.
    """
    from mousedroid.validation import pillars as _p

    real_find_spec = _p.importlib.util.find_spec

    def _fake_find_spec(name: str) -> object | None:
        if name == "pytest":
            return None
        return real_find_spec(name)

    with unittest.mock.patch.object(_p.importlib.util, "find_spec", _fake_find_spec):
        result = _p._run_pytest_delegated(
            "continual",
            _p._PYTEST_DELEGATION_PATHS["continual"],
        )

    assert result.status == PillarStatus.SKIPPED
    assert "pytest not installed" in result.detail
    # The reason must be machine-readable so --strict-skips can fail on it
    # WITHOUT false-failing legitimate config-disabled skips.
    assert result.skip_reason == "environment"


@pytest.mark.asyncio
async def test_dry_run_skips_are_tagged_dry_run() -> None:
    """Dry-run skips carry ``skip_reason="dry_run"`` — never "environment"."""
    cfg = Settings(mock_hardware=True)
    report = await validate_all_pillars(cfg, dry_run=True)
    assert {r.skip_reason for r in report.results} == {"dry_run"}


@pytest.mark.asyncio
async def test_config_disabled_pillars_are_tagged_config_disabled() -> None:
    """``memory``/``curiosity`` SKIP as ``config_disabled`` when memory is off.

    This is the production-overlay shape (no ``memory:`` block →
    ``MemoryConfig.enabled`` defaults False). Strict mode must treat these as
    legitimate passes, so the reason must NOT be "environment".
    """
    cfg = Settings(mock_hardware=True)
    assert cfg.memory.enabled is False, "precondition: memory disabled by default"
    report = await validate_all_pillars(
        cfg,
        pillar_names={"memory", "curiosity"},
        dry_run=False,
    )
    assert {r.name for r in report.results} == {"memory", "curiosity"}
    for result in report.results:
        assert result.status == PillarStatus.SKIPPED
        assert result.skip_reason == "config_disabled"


def test_non_skip_results_have_no_skip_reason() -> None:
    """OK/FAIL results leave ``skip_reason`` as None (additive-field default)."""
    ok = PillarResult(name="safety", status=PillarStatus.OK, detail="x", elapsed_s=0.0)
    fail = PillarResult(name="safety", status=PillarStatus.FAIL, detail="x", elapsed_s=0.0)
    assert ok.skip_reason is None
    assert fail.skip_reason is None
