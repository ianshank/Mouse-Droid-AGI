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
        python -m mousedroid.cli.rearm --config <overlay.yaml> \
        --operator <name> --confirm-area-clear

**Which overlay.** The latch lives under ``experience.path``, which the deployed overlay
sets, so the CLI must load the same overlay stack as the rover or it looks in the wrong
place -- and a latch that is not found reads as "clear". Pass ``--config`` (once per
overlay, or several after one flag): it is the documented form and it always wins. When
it is omitted the stack comes from the same environment variables the rest of the
operator tooling reads (:func:`mousedroid.validation.runtime.resolve_runtime_config`),
which is a safety net: the shipped container sets ``MOUSEDROID_CONFIG``, so a bare
invocation there still finds the production latch. With neither, the CLI loads
``default.yaml`` alone, where the latch is off; it says so on stderr and still exits ``0``
with the same stdout message as before, because it cannot know the deployed config. The
resolution is logged as ``rearm_config_resolved``.

Exit codes: ``0`` cleared or already clear, ``1`` still latched -- whether
because ``--status`` was used or because the required flags were missing.
argparse supplies its own ``2`` for a malformed invocation.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from mousedroid.config.loader import load_settings
from mousedroid.logging.setup import get_logger
from mousedroid.validation.runtime import RUNTIME_CONFIG_ENV_VARS, resolve_runtime_config

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

# Written to stderr, never stdout: the message and exit code on stdout are exactly what an
# operator or script saw before this existed. The variable list is derived from the resolver
# so the note cannot name a variable the runtime does not read, or miss one it does.
_NO_OVERLAY_NOTE: Final[str] = (
    "no config overlay was given (no --config, and none of "
    + ", ".join(RUNTIME_CONFIG_ENV_VARS)
    + " is set), so this read default.yaml alone, which is not the deployed configuration; "
    "pass --config <overlay.yaml> to inspect the latch the rover is actually holding.\n"
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Separate from :func:`main` so a test can assert the interface without
    spawning a subprocess.
    """
    parser = argparse.ArgumentParser(
        prog="python -m mousedroid.cli.rearm",
        description="Clear a latched emergency stop after inspecting the rover.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        nargs="+",
        action="extend",
        default=None,
        help=(
            "Settings YAML overlay(s) to load, in order: list several after one flag or "
            "repeat it. Always wins. When omitted, the overlay stack is the first of "
            + ", ".join(RUNTIME_CONFIG_ENV_VARS)
            + " that is set (the resolution the rest of the runtime uses), else none: "
            "default.yaml alone, where the latch is off."
        ),
    )
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


def _resolve_overlays(cli_paths: Sequence[Path] | None) -> tuple[Path, ...]:
    """Pick the overlay stack: ``--config``, else what the rest of the runtime resolves.

    This is the only place the CLI decides which config to load, and it does not read the
    environment itself -- the shared resolver does, so the CLI cannot form its own opinion
    about which variables name an overlay. A resolution that ends in ``default.yaml`` alone
    gets a one-line stderr note, because that read is almost never the deployed config and
    its "nothing to clear" is indistinguishable from a genuinely clear rover.

    Args:
        cli_paths: The parsed ``--config`` values, or ``None`` when the flag was absent.

    Returns:
        The overlay files to merge over ``default.yaml``, in order.
    """
    resolution = resolve_runtime_config(cli_paths)
    _log.info(
        "rearm_config_resolved",
        source=resolution.source,
        env_var=resolution.env_var,
        paths=[str(path) for path in resolution.paths],
    )
    if resolution.source == "default":
        sys.stderr.write(_NO_OVERLAY_NOTE)
    return resolution.paths


async def _run(args: argparse.Namespace) -> int:
    """Load settings, then report or clear."""
    from mousedroid.factory.safety import build_emergency_latch

    cfg = load_settings(*_resolve_overlays(args.config))
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
