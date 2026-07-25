"""``python -m mousedroid.cli.validate_pillars`` — operator entry point.

Thin argparse wrapper over :func:`mousedroid.validation.pillars.validate_all_pillars`.
Returns exit code 0 on all-pass, 1 on any FAIL — suitable for use in
``scripts/ci.sh`` and operator runbooks.

Opt-in strict mode: ``--strict-skips`` additionally exits 1 when any pillar
SKIPPED with ``skip_reason == "environment"`` (Pattern-B pytest missing from
the runtime) — an unexercised pillar must not read as a pass on a validation
platform. Config-disabled skips (memory/curiosity off in cfg) still pass;
``--strict-skips`` combined with ``--dry-run`` is rejected (dry-run skips
everything by design). The default exit-code contract (0 on OK/DEGRADED,
1 only on FAIL) is unchanged when the flag is absent.
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
    parser.add_argument(
        "--strict-skips",
        action="store_true",
        help=(
            "Exit 1 when any pillar SKIPPED with skip_reason=environment "
            "(runtime cannot exercise the check, e.g. pytest absent). "
            "Config-disabled skips still pass. Incompatible with --dry-run."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — returns process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.strict_skips and args.dry_run:
        # Dry-run marks EVERY pillar SKIPPED by design — strict mode over a
        # dry-run is contradictory, so refuse loudly (argparse exit code 2)
        # instead of silently passing or spuriously failing.
        parser.error("--strict-skips cannot be combined with --dry-run")

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
    # Exit 0 on OK or DEGRADED (WARN-only) — mirrors preflight CLI. Only
    # FAIL (or skipped-only with no OK) blocks CI.
    if report.overall_status == PillarStatus.FAIL:
        return 1
    if args.strict_skips:
        environment_skips = [r.name for r in report.results if r.skip_reason == "environment"]
        if environment_skips:
            _log.warning(
                "pillar_environment_skips_strict_fail",
                pillars=environment_skips,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
