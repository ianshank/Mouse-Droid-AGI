"""Pure SUMMARY.md renderer for the full-validation harness (F-018, WS-4.2).

``scripts/jetson_full_validation.sh`` used to build its Phase-4 SUMMARY.md as
an inline, untested bash heredoc. This module owns that rendering as a pure,
mypy-strict function of its inputs so the summary logic sits under the
coverage gate; the bash script feeds it the pipe-delimited ``RESULTS[]`` rows
(via ``scripts/render_validation_summary.py``) and keeps a minimal fallback
table for python-less hosts.

Also extracts the ``--trend`` block that ``RegressionReport.render_text()``
(see :mod:`mousedroid.validation.report_store`) prints into the Phase-2
preflight log, so run-over-run regressions surface in the operator-facing
SUMMARY instead of only in a log nobody tails.
"""

from __future__ import annotations

from dataclasses import dataclass

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

# The RESULTS[] row convention owned by jetson_full_validation.sh: pipe-
# delimited STATUS|name|note with note optional (single definition point for
# the delimiter — the bash side writes it, this side parses it).
_ROW_DELIMITER = "|"
_TREND_PREFIX = "Trend:"
_TREND_BULLET = "  - "
_NO_TREND_PLACEHOLDER = "Trend: no trend data recorded this run."


@dataclass(frozen=True)
class SummaryRow:
    """One parsed ``RESULTS[]`` row (STATUS|name|note)."""

    status: str
    name: str
    note: str = ""


def parse_result_rows(lines: list[str]) -> list[SummaryRow]:
    """Parse pipe-delimited result rows; malformed lines are skipped loudly.

    A malformed row (no delimiter) is logged and dropped rather than raised —
    a corrupted results file must still yield a summary of the valid rows.
    """
    rows: list[SummaryRow] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        # STATUS|name|note is a fixed 3-field row shape (maxsplit=2 keeps
        # delimiters inside the note intact) — structural, not tunable.
        parts = line.split(_ROW_DELIMITER, 2)  # hardcoded-ok: row-format field count
        if len(parts) < 2:  # hardcoded-ok: STATUS+name are the mandatory fields
            _log.warning("summary_row_malformed", row=line)
            continue
        rows.append(
            SummaryRow(
                status=parts[0].strip(),
                name=parts[1].strip(),
                note=parts[2].strip() if len(parts) > 2 else "",  # hardcoded-ok: optional field
            )
        )
    return rows


def extract_trend_block(preflight_log_text: str) -> str | None:
    """Pull the ``Trend: ...`` block a ``--trend`` preflight run printed.

    Returns the ``Trend:`` line plus its immediately-following ``  - `` bullet
    lines (the exact shape ``RegressionReport.render_text()`` emits), or
    ``None`` when the log carries no trend output.
    """
    lines = preflight_log_text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(_TREND_PREFIX):
            block = [line]
            for follower in lines[i + 1 :]:
                if follower.startswith(_TREND_BULLET):
                    block.append(follower)
                else:
                    break
            return "\n".join(block)
    return None


def _escape_cell(text: str) -> str:
    """Escape markdown-table-breaking characters in a free-text cell.

    ``parse_result_rows`` deliberately preserves ``|`` inside the note
    (maxsplit=2), so a pipe-bearing note is a realistic input — escape it
    rather than let it split the row into spurious columns. Newlines would
    end the row entirely, so they collapse to a single space.
    """
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_summary(
    rows: list[SummaryRow],
    *,
    stamp: str,
    repo: str,
    config: str,
    telemetry_url: str,
    trend_block: str | None,
) -> str:
    """Render the operator-facing SUMMARY.md (existing table + Trend section)."""
    passes = sum(1 for r in rows if r.status == "PASS")
    warns = sum(1 for r in rows if r.status == "WARN")
    failures = sum(1 for r in rows if r.status == "FAIL")

    lines = [
        "# Jetson full-validation summary",
        "",
        f"- UTC: {stamp}",
        f"- Repo: {repo}",
        f"- Config: {config}",
        f"- Telemetry: {telemetry_url}",
        f"- Totals: PASS={passes} WARN={warns} FAIL={failures}",
        "",
        "| Status | Check | Note |",
        "|--------|-------|------|",
    ]
    lines.extend(
        f"| {_escape_cell(r.status)} | {_escape_cell(r.name)} | {_escape_cell(r.note)} |"
        for r in rows
    )
    lines.extend(
        [
            "",
            "## Trend",
            "",
            trend_block if trend_block is not None else _NO_TREND_PLACEHOLDER,
            "",
        ]
    )
    _log.debug(
        "validation_summary_rendered",
        rows=len(rows),
        has_trend=trend_block is not None,
    )
    return "\n".join(lines)
