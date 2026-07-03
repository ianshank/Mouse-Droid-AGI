"""Unit tests for the pure SUMMARY.md renderer (F-018, WS-4.2)."""

from __future__ import annotations

from mousedroid.validation.report_store import RegressionReport
from mousedroid.validation.summary import (
    SummaryRow,
    extract_trend_block,
    parse_result_rows,
    render_summary,
)


class TestParseResultRows:
    def test_parses_status_name_note(self) -> None:
        rows = parse_result_rows(["PASS|preflight (real)|", "WARN|serial smoke|dead ESP32"])
        assert rows == [
            SummaryRow(status="PASS", name="preflight (real)", note=""),
            SummaryRow(status="WARN", name="serial smoke", note="dead ESP32"),
        ]

    def test_note_may_contain_delimiters(self) -> None:
        rows = parse_result_rows(["FAIL|x|a|b|c"])
        assert rows[0].note == "a|b|c"

    def test_malformed_rows_are_skipped_not_raised(self) -> None:
        rows = parse_result_rows(["not-a-row", "", "PASS|ok|"])
        assert [r.name for r in rows] == ["ok"]


class TestExtractTrendBlock:
    def test_absent_trend_returns_none(self) -> None:
        assert extract_trend_block("preflight output\nno trend here\n") is None

    def test_ok_line_extracted(self) -> None:
        log = "checks...\nTrend: OK — no regression vs previous run.\ntrailing\n"
        assert extract_trend_block(log) == "Trend: OK — no regression vs previous run."

    def test_regression_bullets_extracted(self) -> None:
        # Feed the EXACT shape RegressionReport.render_text() emits, so this
        # test breaks if the upstream format drifts.
        rendered = RegressionReport(
            compared=True,
            regressions=["check 'camera' newly FAILing", "total elapsed slowed"],
        ).render_text()
        log = f"json output...\n{rendered}\nunrelated tail\n"
        block = extract_trend_block(log)
        assert block is not None
        assert block.splitlines() == rendered.splitlines()

    def test_insufficient_history_passes_through(self) -> None:
        rendered = RegressionReport(compared=False).render_text()
        assert extract_trend_block(f"x\n{rendered}\n") == rendered


_ROWS = [
    SummaryRow(status="PASS", name="a"),
    SummaryRow(status="WARN", name="b", note="n"),
    SummaryRow(status="FAIL", name="c"),
]


class TestRenderSummary:
    def _render(self, trend_block: str | None) -> str:
        return render_summary(
            _ROWS,
            stamp="20260703T000000Z",
            repo="/opt/mousedroid",
            config="config/jetson_production.yaml",
            telemetry_url="http://127.0.0.1:8080",
            trend_block=trend_block,
        )

    def test_totals_and_table(self) -> None:
        out = self._render(None)
        assert "- Totals: PASS=1 WARN=1 FAIL=1" in out
        assert "| WARN | b | n |" in out

    def test_trend_section_with_block(self) -> None:
        out = self._render("Trend: 1 regression(s) vs previous run:\n  - check 'x' newly FAILing")
        assert "## Trend" in out
        assert "newly FAILing" in out

    def test_trend_section_placeholder_when_absent(self) -> None:
        out = self._render(None)
        assert "## Trend" in out
        assert "no trend data recorded this run" in out
