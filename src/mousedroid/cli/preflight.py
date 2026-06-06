"""``python -m mousedroid.cli.preflight`` — operator entry point.

Thin argparse wrapper over :func:`mousedroid.validation.preflight.run_preflight`.
Returns exit code 0 on all-pass / WARN-only, 1 on any FAIL — suitable
for ``scripts/preflight_check.sh`` replacement and operator runbooks.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from mousedroid.config.loader import load_settings
from mousedroid.logging.setup import get_logger
from mousedroid.validation.preflight import PreflightStatus, run_preflight

_log = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mousedroid.cli.preflight",
        description="Run pre-flight hardware + config checks against the loaded cfg.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Override config YAML path(s). Repeat for multiple overlays.",
    )
    parser.add_argument(
        "--checks",
        default=None,
        help=(
            "Comma-separated check names to run (subset of: camera, "
            "microphone, speaker, lidar, esp32, config). Default: all."
        ),
    )
    parser.add_argument(
        "--mock-hardware",
        action="store_true",
        help=(
            "Force ``cfg.mock_hardware=True`` (every check short-circuits "
            "to OK). Use to verify the dispatch wiring without touching "
            "real devices."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--journal-path",
        default=None,
        help=(
            "Append this run to a JSONL validation journal at PATH for "
            "trend tracking across runs. Opt-in; omit to leave no trace."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier stored with --journal-path (default: UTC stamp).",
    )
    parser.add_argument(
        "--trend",
        action="store_true",
        help=(
            "After recording, compare against the previous run in the journal "
            "and print any regressions (status downgrade, new FAIL, latency "
            "creep). Requires --journal-path. Exit 1 if a regression is found."
        ),
    )
    return parser


async def _record_and_trend(
    *,
    journal_path: str,
    run_id: str,
    report: object,
    show_trend: bool,
) -> bool:
    """Persist ``report`` to a JSONL journal; optionally print regressions.

    Returns True when a regression was detected (used to gate the exit code).
    Kept out of :func:`main` so the import + journal lifecycle stays lazy —
    operators who never pass ``--journal-path`` pay nothing.
    """
    from pathlib import Path as _Path

    from mousedroid.config.schema import HarnessJournalConfig
    from mousedroid.harness.journal.jsonl_journal import JSONLJournal
    from mousedroid.validation.report_store import (
        detect_regressions,
        read_report_history,
        record_report,
    )

    # model_validate (vs direct construction) lets the other journal tunables
    # (map_size_gb / flush_every_n / queue_max) resolve to their schema defaults
    # without restating them here — no hardcoded values, mypy-strict clean.
    journal_cfg = HarnessJournalConfig.model_validate(
        {"backend": "jsonl", "path": _Path(journal_path)},
    )
    journal = JSONLJournal(journal_cfg)
    await journal.start()
    try:
        await record_report(journal, report, run_id=run_id)  # type: ignore[arg-type]
        if not show_trend:
            return False
        history = await read_report_history(journal)
        regression = detect_regressions(history)
        sys.stdout.write(regression.render_text() + "\n")
        return regression.has_regressions
    finally:
        await journal.stop()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — returns process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ``load_settings`` signature is ``(*overlay_paths, config_dir=None)``.
    overlay_paths = [Path(p) for p in (args.config or [])]
    cfg = load_settings(*overlay_paths)
    if args.mock_hardware:
        cfg.mock_hardware = True

    if args.trend and not args.journal_path:
        parser.error("--trend requires --journal-path")

    check_names = set(args.checks.split(",")) if args.checks else None
    report = asyncio.run(run_preflight(cfg, check_names=check_names))
    output = report.model_dump_json(indent=2) if args.json else report.render_text()
    sys.stdout.write(output + "\n")

    regressed = False
    if args.journal_path:
        from datetime import datetime, timezone

        run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        regressed = asyncio.run(
            _record_and_trend(
                journal_path=args.journal_path,
                run_id=run_id,
                report=report,
                show_trend=args.trend,
            ),
        )

    # Exit 0 on OK or DEGRADED (WARN-only); 1 on FAIL or detected regression.
    if report.overall_status == PreflightStatus.FAIL or regressed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
