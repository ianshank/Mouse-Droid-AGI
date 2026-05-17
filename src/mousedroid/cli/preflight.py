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
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — returns process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ``load_settings`` signature is ``(*overlay_paths, config_dir=None)``.
    overlay_paths = [Path(p) for p in (args.config or [])]
    cfg = load_settings(*overlay_paths)
    if args.mock_hardware:
        cfg.mock_hardware = True

    check_names = set(args.checks.split(",")) if args.checks else None
    report = asyncio.run(run_preflight(cfg, check_names=check_names))
    output = report.model_dump_json(indent=2) if args.json else report.render_text()
    sys.stdout.write(output + "\n")
    # Exit 0 on OK or DEGRADED (WARN-only); 1 on FAIL.
    return 0 if report.overall_status != PreflightStatus.FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
