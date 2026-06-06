"""Persist validation reports to the harness journal + detect regressions.

A single :func:`mousedroid.validation.preflight.run_preflight` call answers
"is the rover healthy *right now*?" — a binary PASS/FAIL. It cannot answer
"is the rover *degrading*?": a camera whose capture time has crept from 40 ms
to 400 ms over a week still passes every individual run, but is one bad boot
from a FAIL. This module closes that gap by appending each run to the existing
append-only harness journal (:mod:`mousedroid.harness.journal`) and comparing
the latest run against its predecessor.

Why reuse the journal rather than a new store: the JSONL/LMDB journal backends
already solve non-blocking append, bounded-memory iteration, and durable
ordering. We add a thin typed envelope on top — no parallel persistence path.

Architecture invariants (per CLAUDE.md):

* Asyncio-only — append + read go through the async journal protocol.
* No hardcoded thresholds — the regression sensitivity (``slow_ratio`` /
  ``slow_floor_s``) are explicit parameters with documented defaults; the
  caller (CLI / operator) owns the policy.
* Backwards compatible — a journal containing only non-report entries reads
  back as an empty history; existing journals are untouched.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from mousedroid.harness.journal.protocol import JournalEntry
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.harness.journal.protocol import JournalProtocol
    from mousedroid.validation.preflight import PreflightReport

_log = get_logger(__name__)

# Stable event id stamped on every report entry so :func:`read_report_history`
# can filter the journal without colliding with orchestrator tick entries.
EVENT_PREFLIGHT_REPORT = "preflight_report"

# Ordinal severity used to detect an aggregate-status *downgrade* across runs.
# Higher = worse. ``degraded`` sits between ``ok`` and ``fail``. Definitional
# ladder, not a runtime-tunable.
_STATUS_RANK: dict[str, int] = {"ok": 0, "degraded": 1, "warn": 1, "fail": 2}  # hardcoded-ok

# Minimum stored runs required to compute a trend (need a prev + curr).
_MIN_RUNS_TO_COMPARE = 2  # hardcoded-ok: structural minimum, not tunable


class CheckSnapshot(BaseModel):
    """One check's outcome, flattened for storage + trend comparison."""

    name: str
    status: str
    elapsed_s: float = Field(ge=0.0)


class StoredReport(BaseModel):
    """A preflight run flattened into a journal-storable, comparable envelope."""

    recorded_at_ns: int = Field(
        description="Wall-clock capture time (time.time_ns()) — stable across "
        "process restarts, unlike the journal's monotonic entry timestamp.",
    )
    run_id: str = Field(description="Operator-supplied run identifier.")
    git_sha: str | None = Field(default=None, description="Source SHA, if known.")
    overall_status: str
    total_elapsed_s: float = Field(ge=0.0)
    checks: list[CheckSnapshot] = Field(default_factory=list)

    def check_by_name(self, name: str) -> CheckSnapshot | None:
        """Return the snapshot for ``name`` or ``None`` if absent this run."""
        for check in self.checks:
            if check.name == name:
                return check
        return None


class RegressionReport(BaseModel):
    """Outcome of comparing the latest stored run against its predecessor."""

    compared: bool = Field(
        description="False when fewer than two runs exist (nothing to compare).",
    )
    regressions: list[str] = Field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        """True when at least one regression was detected."""
        return bool(self.regressions)

    def render_text(self) -> str:
        """Render a single-screen operator summary."""
        if not self.compared:
            return "Trend: insufficient history (need >=2 runs to compare)."
        if not self.regressions:
            return "Trend: OK — no regression vs previous run."
        lines = [f"Trend: {len(self.regressions)} regression(s) vs previous run:"]
        lines.extend(f"  - {r}" for r in self.regressions)
        return "\n".join(lines)


def _to_stored(
    report: PreflightReport,
    *,
    run_id: str,
    git_sha: str | None,
) -> StoredReport:
    return StoredReport(
        recorded_at_ns=time.time_ns(),
        run_id=run_id,
        git_sha=git_sha,
        overall_status=report.overall_status.value,
        total_elapsed_s=report.total_elapsed_s,
        checks=[
            CheckSnapshot(name=c.name, status=c.status.value, elapsed_s=c.elapsed_s)
            for c in report.checks
        ],
    )


async def record_report(
    journal: JournalProtocol,
    report: PreflightReport,
    *,
    run_id: str,
    git_sha: str | None = None,
) -> StoredReport:
    """Append ``report`` to ``journal`` as a single :data:`EVENT_PREFLIGHT_REPORT`.

    The journal must already be ``start()``-ed by the caller (the caller owns
    the lifecycle, mirroring how the orchestrator owns its journal). A
    ``NullJournal`` silently no-ops — recording is always safe to call.

    Args:
        journal: A started journal implementing ``JournalProtocol``.
        report: The preflight report to persist.
        run_id: Operator-supplied run identifier (e.g. a UTC stamp).
        git_sha: Optional source SHA the run was taken against.

    Returns:
        The :class:`StoredReport` that was appended (also useful for tests).
    """
    stored = _to_stored(report, run_id=run_id, git_sha=git_sha)
    await journal.append(
        JournalEntry(
            phase="validation",
            event=EVENT_PREFLIGHT_REPORT,
            payload=stored.model_dump(),
        ),
    )
    _log.info(
        "preflight_report_recorded",
        run_id=run_id,
        git_sha=git_sha,
        overall=stored.overall_status,
        checks=len(stored.checks),
    )
    return stored


async def read_report_history(journal: JournalProtocol) -> list[StoredReport]:
    """Read every stored preflight report from ``journal``, oldest first.

    Non-report journal entries (orchestrator ticks, task events) are skipped.
    Malformed report payloads are skipped with a WARN — one corrupt entry never
    aborts the trend read.

    Args:
        journal: A started journal implementing ``JournalProtocol``.

    Returns:
        Reports sorted ascending by :attr:`StoredReport.recorded_at_ns`.
    """
    history: list[StoredReport] = []
    async for entry in journal.read_all():
        if entry.event != EVENT_PREFLIGHT_REPORT:
            continue
        try:
            history.append(StoredReport.model_validate(entry.payload))
        except Exception as exc:  # pylint: disable=broad-except
            _log.warning(
                "preflight_report_parse_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
    history.sort(key=lambda r: r.recorded_at_ns)
    return history


def detect_regressions(
    history: list[StoredReport],
    *,
    slow_ratio: float = 1.5,  # hardcoded-ok: default; operator-tunable via preflight CLI
    slow_floor_s: float = 0.05,  # hardcoded-ok: default; operator-tunable via preflight CLI
) -> RegressionReport:
    """Compare the two most-recent runs and flag regressions.

    Three regression classes are surfaced:

    * **Status downgrade** — aggregate severity worsened (ok→degraded→fail).
    * **New failing check** — a check that did not FAIL last run now FAILs.
    * **Latency creep** — a check (or the total) got slower by more than
      ``slow_ratio`` *and* by more than ``slow_floor_s`` absolute. The absolute
      floor suppresses noise on sub-50 ms checks where a 1.5x jump is meaningless.

    Args:
        history: Stored reports, any order (the two newest are used).
        slow_ratio: Multiplicative slowdown threshold (e.g. ``1.5`` = +50 %).
        slow_floor_s: Absolute slowdown floor in seconds; deltas below this are
            ignored regardless of ratio.

    Returns:
        A :class:`RegressionReport`. ``compared=False`` when <2 runs exist.
    """
    if len(history) < _MIN_RUNS_TO_COMPARE:
        return RegressionReport(compared=False)
    ordered = sorted(history, key=lambda r: r.recorded_at_ns)
    prev, curr = ordered[-2], ordered[-1]
    regressions: list[str] = []

    prev_rank = _STATUS_RANK.get(prev.overall_status, 0)
    curr_rank = _STATUS_RANK.get(curr.overall_status, 0)
    if curr_rank > prev_rank:
        regressions.append(
            f"overall status downgraded {prev.overall_status} -> {curr.overall_status}",
        )

    for check in curr.checks:
        prior = prev.check_by_name(check.name)
        if check.status == "fail" and (prior is None or prior.status != "fail"):
            regressions.append(f"check '{check.name}' newly FAILing")
        if prior is not None:
            delta = check.elapsed_s - prior.elapsed_s
            if delta > slow_floor_s and check.elapsed_s > prior.elapsed_s * slow_ratio:
                regressions.append(
                    f"check '{check.name}' slowed {prior.elapsed_s:.3f}s -> "
                    f"{check.elapsed_s:.3f}s (+{delta:.3f}s)",
                )

    total_delta = curr.total_elapsed_s - prev.total_elapsed_s
    if total_delta > slow_floor_s and curr.total_elapsed_s > prev.total_elapsed_s * slow_ratio:
        regressions.append(
            f"total elapsed slowed {prev.total_elapsed_s:.3f}s -> "
            f"{curr.total_elapsed_s:.3f}s (+{total_delta:.3f}s)",
        )

    return RegressionReport(compared=True, regressions=regressions)


__all__ = [
    "EVENT_PREFLIGHT_REPORT",
    "CheckSnapshot",
    "RegressionReport",
    "StoredReport",
    "detect_regressions",
    "read_report_history",
    "record_report",
]
