r"""Clear a latched emergency stop (peer review D-5).

ISO 3691-4 requires that an emergency stop is reset only by deliberate human
action, and that the reset does not by itself command motion. This CLI is
that reset. It clears the record and exits; the rover arms on its next start,
so the reset and the resumption of motion are two separate acts by the
operator.

**Why a CLI and not one of the surfaces that already exist.**

* Not the telemetry REST API. It is LAN-reachable behind a bearer token, and
  a token-authenticated POST is the definition of a *remote* re-arm. It would
  also put the e-stop reset on the same ingress as ``POST /api/v1/mission``,
  the natural-language path the injection filter exists to distrust.
* Not an MCP tool. Those are the tools a language model calls. A
  ``rearm_emergency_stop`` tool would let the model that caused the stop undo
  it, straight through the actuation gate in ``mcp/tool_bridge.py``.
* Not ``scripts/preflight_check.sh``. It runs as systemd ``ExecStartPre``
  with no human attached, so anything it can clear, ``Restart=on-failure``
  clears automatically -- recreating the defect exactly.

Container deployments run it inside the service so it sees the same volume::

    docker compose -f docker-compose.jetson.yml exec mousedroid \
        python -m mousedroid.cli.rearm --operator <name> --confirm-area-clear

Exit codes: ``0`` cleared or already clear, ``1`` still latched -- whether
because ``--status`` was used or because the required flags were missing.
argparse supplies its own ``2`` for a malformed invocation.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Final

from mousedroid.config.loader import load_settings
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

# Two codes, matching ``mousedroid.cli.preflight``'s plain ``return 0`` /
# ``return 1``. A third "refused" code was considered and dropped: from a
# caller's point of view "refused because a flag was missing" and "latched,
# and I only asked for status" are the same actionable state -- the latch is
# still set -- and the difference is in the message, which is what an
# operator reads. argparse already exits 2 on a malformed invocation, so the
# usage-error case keeps its conventional code for free.
EXIT_OK: Final[int] = 0
EXIT_STILL_LATCHED: Final[int] = 1


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Separate from :func:`main` so a test can assert the interface without
    spawning a subprocess.
    """
    parser = argparse.ArgumentParser(
        prog="python -m mousedroid.cli.rearm",
        description="Clear a latched emergency stop after inspecting the rover.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Settings YAML to load.")
    parser.add_argument(
        "--operator",
        default=None,
        help="Who is clearing it. Required to clear; recorded in the audit log.",
    )
    parser.add_argument(
        "--confirm-area-clear",
        action="store_true",
        help=(
            "Affirm you have looked at the rover and its path is clear. "
            "Required to clear. No short form and no default, deliberately."
        ),
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Report the latch state and exit without changing anything.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    """Load settings, then report or clear."""
    from mousedroid.factory.safety import build_emergency_latch

    cfg = load_settings(args.config) if args.config else load_settings()
    latch = build_emergency_latch(cfg)
    if latch is None:
        sys.stdout.write(
            "safety.emergency_latch.enabled is false: no latch is in use, "
            "so there is nothing to clear.\n"
        )
        return EXIT_OK

    latched = await latch.load()
    record = latch.record
    if not latched:
        sys.stdout.write("No emergency stop is latched.\n")
        return EXIT_OK

    sys.stdout.write(
        "EMERGENCY STOP LATCHED\n"
        f"  reason:     {record.reason}\n"
        f"  causes:     {', '.join(record.causes) or '(none recorded)'}\n"
        f"  tripped at: {record.tripped_at_iso or '(unknown)'}\n"
    )
    if args.status:
        return EXIT_STILL_LATCHED

    missing = [
        flag
        for flag, present in (
            ("--operator", bool(args.operator)),
            ("--confirm-area-clear", args.confirm_area_clear),
        )
        if not present
    ]
    if missing:
        sys.stdout.write(
            "\nRefusing to clear: " + " and ".join(missing) + " required.\n"
            "Inspect the rover and its path first, then re-run with both.\n"
        )
        return EXIT_STILL_LATCHED

    await latch.rearm(operator=args.operator)
    sys.stdout.write(
        f"\nCleared by {args.operator}. Restart the service to arm the rover; "
        "clearing the latch does not by itself command motion.\n"
    )
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except Exception:
        _log.exception("rearm_failed")
        return EXIT_STILL_LATCHED


if __name__ == "__main__":
    sys.exit(main())
