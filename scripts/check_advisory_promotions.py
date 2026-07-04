#!/usr/bin/env python3
"""Advisory-stage promotion-lag checker (F-020, WS-8.3).

The repo's CI pattern is: new gates land advisory (``continue-on-error:
true``) and get promoted to blocking after a documented green window. The
observed failure mode is stages that stay advisory forever because nobody
remembers the promotion. This checker closes the loop:

* scans ``.github/workflows/*.yml`` for jobs declaring
  ``continue-on-error: true``,
* cross-references ``.github/advisory_stages.yaml`` (the tracked metadata:
  when each stage landed and its promotion window),
* WARNs on **untracked** advisory jobs (no metadata = no promotion plan) and
  on **overdue** promotions (today - since > promote_after_days).

Report-only: exit 0 unless ``--strict``. ``--today YYYY-MM-DD`` makes runs
deterministic for tests and reproducible in CI logs.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import yaml

_DEFAULT_WORKFLOWS_DIR = ".github/workflows"
_DEFAULT_METADATA = ".github/advisory_stages.yaml"


def find_advisory_jobs(workflows_dir: Path) -> dict[str, str]:
    """Return {job_name: workflow_filename} for continue-on-error jobs."""
    advisory: dict[str, str] = {}
    # WARN-only contract: a bad --workflows-dir (or pre-checkout cwd) must not
    # traceback. Empty here means every tracked stage reports as stale
    # metadata downstream - the loudest safe signal.
    if not workflows_dir.is_dir():
        return advisory
    # GitHub Actions honors both extensions; missing .yaml would let a future
    # workflow's advisory job silently escape the promotion-lag guard.
    workflow_paths = sorted([*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")])
    for wf_path in workflow_paths:
        try:
            data = yaml.safe_load(wf_path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue  # actionlint owns workflow syntax; not this checker's job
        except (OSError, UnicodeDecodeError):
            continue  # unreadable/undecodable file degrades to "not scanned"
        if not isinstance(data, dict):
            continue
        jobs = data.get("jobs")
        if not isinstance(jobs, dict):
            continue
        for job_name, job in jobs.items():
            if isinstance(job, dict) and job.get("continue-on-error") is True:
                advisory[str(job_name)] = wf_path.name
    return advisory


def load_tracked_stages(metadata_path: Path) -> list[dict[str, object]]:
    """Load the tracked advisory-stage metadata (empty list when absent).

    Defensive by contract: this checker is WARN-only infrastructure, so a
    malformed file (list root, string stage entries, YAML error) degrades to
    "no tracked stages" — evaluate() then reports every advisory job as
    untracked, which is the loudest safe signal.
    """
    if not metadata_path.is_file():
        return []
    try:
        data = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return []
    if not isinstance(data, dict):
        return []
    stages = data.get("stages", [])
    if not isinstance(stages, list):
        return []
    return [stage for stage in stages if isinstance(stage, dict)]


def evaluate(
    advisory_jobs: dict[str, str],
    tracked: list[dict[str, object]],
    *,
    today: date,
) -> list[str]:
    """Return human-readable warnings (empty == healthy)."""
    warnings: list[str] = []
    tracked_by_job = {str(s.get("job")): s for s in tracked}

    for job_name, workflow in sorted(advisory_jobs.items()):
        stage = tracked_by_job.get(job_name)
        if stage is None:
            warnings.append(
                f"untracked advisory stage: job '{job_name}' ({workflow}) has "
                "continue-on-error but no entry in .github/advisory_stages.yaml "
                "- add one with since + promote_after_days"
            )
            continue
        try:
            since = date.fromisoformat(str(stage.get("since")))
            window = int(str(stage.get("promote_after_days")))
        except (ValueError, TypeError):
            # A hand-edited metadata entry must degrade to a warning, never a
            # traceback - this checker is itself advisory infrastructure.
            warnings.append(
                f"malformed metadata for job '{job_name}': 'since' must be "
                "YYYY-MM-DD and 'promote_after_days' an integer - fix the "
                "entry in .github/advisory_stages.yaml"
            )
            continue
        age = (today - since).days
        if age > window:
            warnings.append(
                f"promotion overdue: job '{job_name}' ({workflow}) has been "
                f"advisory {age} days (window {window}d since {since}) - "
                "promote to blocking or extend the window with a recorded reason"
            )

    for stage in tracked:
        job_name = str(stage.get("job"))
        if job_name not in advisory_jobs:
            warnings.append(
                f"stale metadata: '{job_name}' tracked in advisory_stages.yaml "
                "but no workflow job declares continue-on-error - promoted or "
                "removed? Clean up the entry."
            )
    return warnings


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python scripts/check_advisory_promotions.py",
        description="WARN on advisory CI stages that outlived their promotion window.",
    )
    parser.add_argument(
        "--workflows-dir",
        default=_DEFAULT_WORKFLOWS_DIR,
        help="Workflows directory (default: %(default)s).",
    )
    parser.add_argument(
        "--metadata",
        default=_DEFAULT_METADATA,
        help="Tracked advisory-stage metadata (default: %(default)s).",
    )
    parser.add_argument(
        "--today",
        default=None,
        help="Override today's date (YYYY-MM-DD) for deterministic runs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when any warning fires (default: advisory, always exit 0).",
    )
    args = parser.parse_args(argv)

    today = date.fromisoformat(args.today) if args.today else date.today()
    warnings = evaluate(
        find_advisory_jobs(Path(args.workflows_dir)),
        load_tracked_stages(Path(args.metadata)),
        today=today,
    )
    for warning in warnings:
        print(f"WARN: {warning}")
    if not warnings:
        print("advisory promotions: all stages tracked and within window")
    if args.strict and warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
