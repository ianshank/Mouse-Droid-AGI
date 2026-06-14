"""Integration: validation report store wired through the factory journal.

Proves the trend store is journal-backend-agnostic — it records and reads back
through whatever ``build_journal(cfg)`` returns, exercising the full
record -> persist -> read -> detect_regressions path against the **real**
JSONL and LMDB backends (not a stub). This is the modularity contract: the
store depends only on ``JournalProtocol``, never a concrete backend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.schema import HarnessConfig, HarnessJournalConfig, Settings
from mousedroid.factory import build_journal
from mousedroid.validation.preflight import (
    PreflightCheckResult,
    PreflightReport,
    PreflightStatus,
)
from mousedroid.validation.report_store import (
    detect_regressions,
    read_report_history,
    record_report,
)


def _settings_with_journal(backend: str, path: Path) -> Settings:
    """Build Settings whose harness journal uses ``backend`` at ``path``."""
    journal_cfg = HarnessJournalConfig.model_validate({"backend": backend, "path": path})
    return Settings(harness=HarnessConfig(journal=journal_cfg))


def _report(status: PreflightStatus, elapsed_s: float, *, total: float) -> PreflightReport:
    return PreflightReport(
        checks=[PreflightCheckResult(name="camera", status=status, elapsed_s=elapsed_s)],
        total_elapsed_s=total,
    )


@pytest.mark.parametrize("backend", ["jsonl", "lmdb"])
async def test_record_read_detect_through_factory_journal(
    backend: str,
    tmp_path: Path,
) -> None:
    """Two recorded runs are read back in order and a slowdown is detected."""
    cfg = _settings_with_journal(backend, tmp_path / f"journal_{backend}")
    journal = build_journal(cfg)
    await journal.start()
    try:
        await record_report(
            journal,
            _report(PreflightStatus.OK, 0.10, total=0.5),
            run_id="run-1",
            git_sha="aaaa",
        )
        await record_report(
            journal,
            _report(PreflightStatus.OK, 0.40, total=2.0),
            run_id="run-2",
            git_sha="bbbb",
        )
        history = await read_report_history(journal)
    finally:
        await journal.stop()

    assert [r.run_id for r in history] == ["run-1", "run-2"]
    regression = detect_regressions(history)
    assert regression.compared is True
    assert regression.has_regressions is True
    assert any("camera" in r and "slowed" in r for r in regression.regressions)


async def test_null_journal_default_records_nothing(tmp_path: Path) -> None:
    """With the harness disabled (default), build_journal -> NullJournal: empty history."""
    journal = build_journal(Settings())  # harness=None by default
    await journal.start()
    try:
        await record_report(
            journal,
            _report(PreflightStatus.OK, 0.1, total=0.5),
            run_id="run-1",
        )
        history = await read_report_history(journal)
    finally:
        await journal.stop()
    assert history == []
