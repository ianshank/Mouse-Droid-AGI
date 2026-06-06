"""Unit tests for the validation report store + regression detection."""

from __future__ import annotations

from pathlib import Path

from mousedroid.config.schema import HarnessJournalConfig
from mousedroid.harness.journal.jsonl_journal import JSONLJournal
from mousedroid.harness.journal.protocol import JournalEntry
from mousedroid.validation.preflight import (
    PreflightCheckResult,
    PreflightReport,
    PreflightStatus,
)
from mousedroid.validation.report_store import (
    EVENT_PREFLIGHT_REPORT,
    StoredReport,
    detect_regressions,
    read_report_history,
    record_report,
)


def _report(
    *checks: tuple[str, PreflightStatus, float],
    total: float = 1.0,
) -> PreflightReport:
    return PreflightReport(
        checks=[PreflightCheckResult(name=n, status=s, elapsed_s=e) for n, s, e in checks],
        total_elapsed_s=total,
    )


def _journal(tmp_path: Path) -> JSONLJournal:
    return JSONLJournal(
        HarnessJournalConfig(backend="jsonl", path=tmp_path / "validation.jsonl"),
    )


async def test_record_then_read_roundtrip(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    await journal.start()
    try:
        report = _report(("camera", PreflightStatus.OK, 0.04))
        stored = await record_report(journal, report, run_id="run-1", git_sha="abc123")
        assert stored.run_id == "run-1"
        assert stored.git_sha == "abc123"
        history = await read_report_history(journal)
    finally:
        await journal.stop()

    assert len(history) == 1
    assert history[0].overall_status == "ok"
    assert history[0].check_by_name("camera") is not None
    assert history[0].check_by_name("nonexistent") is None


async def test_read_history_skips_non_report_entries(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    await journal.start()
    try:
        await journal.append(JournalEntry(event="tick", payload={"i": 1}))
        await record_report(journal, _report(("config", PreflightStatus.OK, 0.0)), run_id="r")
        await journal.append(JournalEntry(event="other", payload={"x": 2}))
        history = await read_report_history(journal)
    finally:
        await journal.stop()
    assert len(history) == 1


async def test_read_history_tolerates_malformed_report_payload(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    await journal.start()
    try:
        # A report-tagged entry with a payload that fails StoredReport validation.
        await journal.append(
            JournalEntry(event=EVENT_PREFLIGHT_REPORT, payload={"garbage": True}),
        )
        await record_report(journal, _report(("config", PreflightStatus.OK, 0.0)), run_id="r")
        history = await read_report_history(journal)
    finally:
        await journal.stop()
    assert len(history) == 1  # malformed entry skipped, good one kept


class TestDetectRegressions:
    def _stored(
        self,
        *,
        at: int,
        status: str,
        total: float,
        checks: list[tuple[str, str, float]],
    ) -> StoredReport:
        from mousedroid.validation.report_store import CheckSnapshot

        return StoredReport(
            recorded_at_ns=at,
            run_id=f"run-{at}",
            overall_status=status,
            total_elapsed_s=total,
            checks=[CheckSnapshot(name=n, status=s, elapsed_s=e) for n, s, e in checks],
        )

    def test_insufficient_history_not_compared(self) -> None:
        result = detect_regressions([])
        assert result.compared is False
        assert result.has_regressions is False
        assert "insufficient history" in result.render_text()

    def test_no_regression_when_stable(self) -> None:
        prev = self._stored(at=1, status="ok", total=1.0, checks=[("camera", "ok", 0.1)])
        curr = self._stored(at=2, status="ok", total=1.0, checks=[("camera", "ok", 0.1)])
        result = detect_regressions([prev, curr])
        assert result.compared is True
        assert result.has_regressions is False

    def test_status_downgrade_flagged(self) -> None:
        prev = self._stored(at=1, status="ok", total=1.0, checks=[])
        curr = self._stored(at=2, status="fail", total=1.0, checks=[])
        result = detect_regressions([prev, curr])
        assert any("downgraded" in r for r in result.regressions)

    def test_new_failing_check_flagged(self) -> None:
        prev = self._stored(at=1, status="ok", total=1.0, checks=[("lidar", "ok", 0.1)])
        curr = self._stored(at=2, status="fail", total=1.0, checks=[("lidar", "fail", 0.1)])
        result = detect_regressions([prev, curr])
        assert any("newly FAILing" in r for r in result.regressions)

    def test_check_absent_in_previous_run_is_handled(self) -> None:
        # esp32 is new this run (prior is None → no elapsed comparison); camera
        # exists in both. Covers both branches of the per-check prior lookup.
        prev = self._stored(at=1, status="ok", total=1.0, checks=[("camera", "ok", 0.1)])
        curr = self._stored(
            at=2,
            status="fail",
            total=1.0,
            checks=[("camera", "ok", 0.1), ("esp32", "fail", 0.2)],
        )
        result = detect_regressions([prev, curr])
        assert any("esp32" in r and "newly FAILing" in r for r in result.regressions)

    def test_latency_creep_flagged_above_ratio_and_floor(self) -> None:
        prev = self._stored(at=1, status="ok", total=1.0, checks=[("camera", "ok", 0.1)])
        curr = self._stored(at=2, status="ok", total=1.0, checks=[("camera", "ok", 0.4)])
        result = detect_regressions([prev, curr])
        assert any("slowed" in r and "camera" in r for r in result.regressions)

    def test_tiny_slowdown_below_floor_ignored(self) -> None:
        # 0.001 -> 0.01 is 10x ratio but only +0.009s, under the 0.05s floor.
        prev = self._stored(at=1, status="ok", total=1.0, checks=[("config", "ok", 0.001)])
        curr = self._stored(at=2, status="ok", total=1.0, checks=[("config", "ok", 0.01)])
        result = detect_regressions([prev, curr])
        assert result.has_regressions is False

    def test_total_slowdown_flagged(self) -> None:
        prev = self._stored(at=1, status="ok", total=1.0, checks=[])
        curr = self._stored(at=2, status="ok", total=2.0, checks=[])
        result = detect_regressions([prev, curr])
        assert any("total elapsed slowed" in r for r in result.regressions)

    def test_unordered_history_uses_two_newest(self) -> None:
        a = self._stored(at=3, status="fail", total=1.0, checks=[])
        b = self._stored(at=1, status="ok", total=1.0, checks=[])
        c = self._stored(at=2, status="ok", total=1.0, checks=[])
        # Newest two by recorded_at_ns are c(2) then a(3): ok -> fail downgrade.
        result = detect_regressions([a, b, c])
        assert any("downgraded" in r for r in result.regressions)

    def test_custom_thresholds_are_honoured(self) -> None:
        # +0.15s is below a 0.2s floor -> not flagged even at a low ratio.
        prev = self._stored(at=1, status="ok", total=1.0, checks=[("camera", "ok", 0.1)])
        curr = self._stored(at=2, status="ok", total=1.0, checks=[("camera", "ok", 0.25)])
        loose = detect_regressions([prev, curr], slow_ratio=1.1, slow_floor_s=0.2)
        strict = detect_regressions([prev, curr], slow_ratio=1.1, slow_floor_s=0.01)
        assert loose.has_regressions is False
        assert strict.has_regressions is True

    def test_render_text_no_regression_branch(self) -> None:
        prev = self._stored(at=1, status="ok", total=1.0, checks=[])
        curr = self._stored(at=2, status="ok", total=1.0, checks=[])
        text = detect_regressions([prev, curr]).render_text()
        assert "no regression" in text

    def test_render_text_lists_each_regression(self) -> None:
        prev = self._stored(at=1, status="ok", total=1.0, checks=[])
        curr = self._stored(at=2, status="fail", total=3.0, checks=[])
        text = detect_regressions([prev, curr]).render_text()
        assert "regression(s)" in text
        assert text.count("  - ") >= 1


async def test_record_and_read_are_safe_on_null_journal() -> None:
    """NullJournal (harness disabled) silently no-ops; history reads empty."""
    from mousedroid.harness.journal.null_journal import NullJournal

    journal = NullJournal()
    await journal.start()
    try:
        stored = await record_report(
            journal,
            _report(("config", PreflightStatus.OK, 0.0)),
            run_id="null-run",
        )
        assert stored.run_id == "null-run"  # return value still well-formed
        history = await read_report_history(journal)
    finally:
        await journal.stop()
    assert history == []
