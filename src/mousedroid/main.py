"""MouseDroid CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import Settings
from mousedroid.logging.setup import configure_logging, get_logger


def cli_entry() -> None:  # pragma: no cover
    """CLI entry point for mousedroid command."""
    parser = argparse.ArgumentParser(description="MouseDroid — MSE-6 Autonomous Navigation")
    parser.add_argument(
        "--config",
        type=Path,
        nargs="*",
        default=[],
        help="YAML config overlay files",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run health check and exit",
    )
    parser.add_argument(
        "--mock-hardware",
        action="store_true",
        help="Force mock hardware mode",
    )
    args = parser.parse_args()

    cfg = load_settings(*args.config)
    if args.mock_hardware:
        cfg = cfg.model_copy(update={"mock_hardware": True})

    configure_logging(
        cfg.logging,
        robot_id=cfg.gcp.robot_id if cfg.gcp is not None else None,
    )
    log = get_logger(__name__)
    log.info("mousedroid_starting", platform=str(cfg.platform), mock=cfg.mock_hardware)

    if args.health_check:
        asyncio.run(_health_check(cfg))
        return

    asyncio.run(_run(cfg))


async def _health_check(cfg: Settings) -> None:  # pragma: no cover
    """Run health check.

    Args:
        cfg: Settings instance.
    """
    from mousedroid.factory import build_orchestrator
    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    orch_obj = build_orchestrator(cfg)
    if not isinstance(orch_obj, MouseDroidOrchestrator):
        # Explicit check, not ``assert`` — asserts are stripped under -O
        # (PYTHONOPTIMIZE=1 is the Jetson Docker default).
        raise TypeError(f"build_orchestrator returned {type(orch_obj).__name__}")
    result = await orch_obj.health_check()
    log = get_logger(__name__)
    log.info("health_check_result", **{k: str(v) for k, v in result.items()})
    if result.get("status") == "ok":
        log.info("health_check_passed")
    else:
        log.error("health_check_failed", result=result)
        sys.exit(1)


async def _run(cfg: Settings) -> None:  # pragma: no cover
    """Run the main orchestrator loop.

    Args:
        cfg: Settings instance.
    """
    from mousedroid.factory import build_orchestrator
    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    orch_obj = build_orchestrator(cfg)
    if not isinstance(orch_obj, MouseDroidOrchestrator):
        # Explicit check, not ``assert`` — asserts are stripped under -O
        # (PYTHONOPTIMIZE=1 is the Jetson Docker default).
        raise TypeError(f"build_orchestrator returned {type(orch_obj).__name__}")
    await orch_obj.start()
    try:
        await orch_obj.run()
    finally:
        await orch_obj.stop()


if __name__ == "__main__":  # pragma: no cover
    cli_entry()
