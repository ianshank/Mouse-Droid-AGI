"""``python -m mousedroid.cli.validate_pillars`` — operator entry point.

Thin argparse wrapper over :func:`mousedroid.validation.pillars.validate_all_pillars`.
Returns exit code 0 on all-pass, 1 on any FAIL — suitable for use in
``scripts/ci.sh`` and operator runbooks.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from mousedroid.config.loader import load_settings
from mousedroid.logging.setup import get_logger
from mousedroid.validation.pillars import PillarStatus, validate_all_pillars

_log = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mousedroid.cli.validate_pillars",
        description="Run all (or filtered) pillar smoke checks.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Override config YAML path(s). Repeat for multiple overlays.",
    )
    parser.add_argument(
        "--pillars",
        default=None,
        help=(
            "Comma-separated pillar names to run (subset of: safety, "
            "world_model, memory, cognitive, reward, curiosity, continual, "
            "meta, scaling, growth)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List pillars without invoking checks (CI sanity gate).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — returns process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ``load_settings`` signature is ``(*overlay_paths, config_dir=None)`` —
    # positional varargs of ``Path`` objects, NOT a keyword ``config_paths=``.
    overlay_paths = [Path(p) for p in (args.config or [])]
    cfg = load_settings(*overlay_paths)
    pillar_names = set(args.pillars.split(",")) if args.pillars else None

    report = asyncio.run(
        validate_all_pillars(
            cfg,
            pillar_names=pillar_names,
            dry_run=args.dry_run,
        ),
    )
    output = report.model_dump_json(indent=2) if args.json else report.render_text()
    sys.stdout.write(output + "\n")
    return 0 if report.overall_status == PillarStatus.OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
