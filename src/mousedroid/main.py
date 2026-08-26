"""MouseDroid CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import Settings
from mousedroid.logging.setup import configure_logging, get_logger

if TYPE_CHECKING:
    from mousedroid.cloud.protocol import CloudLoggingSinkProtocol


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

    from mousedroid.factory import build_cloud_logging_sink

    cloud_logging_sink = build_cloud_logging_sink(cfg)
    configure_logging(
        cfg.logging,
        cloud_logging_sink=cloud_logging_sink,
        robot_id=cfg.gcp.robot_id if cfg.gcp is not None else None,
    )
    log = get_logger(__name__)
    log.info("mousedroid_starting", platform=str(cfg.platform), mock=cfg.mock_hardware)

    if args.health_check:
        asyncio.run(_health_check(cfg, cloud_logging_sink))
        return

    asyncio.run(_run(cfg, cloud_logging_sink))


async def _health_check(
    cfg: Settings,
    cloud_logging_sink: CloudLoggingSinkProtocol | None = None,
) -> None:  # pragma: no cover
    """Run health check.

    Args:
        cfg: Settings instance.
        cloud_logging_sink: Optional Cloud Logging sink to start/close around
            the check. Start/close failures are logged and swallowed — an
            unreachable Cloud Logging backend must never block a health
            check whose entire purpose is diagnosing rover readiness.
    """
    from mousedroid.factory import build_orchestrator
    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    log = get_logger(__name__)
    if cloud_logging_sink is not None:
        try:
            await cloud_logging_sink.start()
        except Exception:
            log.warning("cloud_logging_sink_start_failed", exc_info=True)

    try:
        orch_obj = build_orchestrator(cfg)
        if not isinstance(orch_obj, MouseDroidOrchestrator):
            # Explicit check, not ``assert`` — asserts are stripped under -O
            # (PYTHONOPTIMIZE=1 is the Jetson Docker default).
            raise TypeError(f"build_orchestrator returned {type(orch_obj).__name__}")
        result = await orch_obj.health_check()
        log.info("health_check_result", **{k: str(v) for k, v in result.items()})
        if result.get("status") == "ok":
            log.info("health_check_passed")
        else:
            log.error("health_check_failed", result=result)
            sys.exit(1)
    finally:
        if cloud_logging_sink is not None:
            try:
                await cloud_logging_sink.close()
            except Exception:
                log.warning("cloud_logging_sink_close_failed", exc_info=True)


async def _run(
    cfg: Settings,
    cloud_logging_sink: CloudLoggingSinkProtocol | None = None,
) -> None:  # pragma: no cover
    """Run the main orchestrator loop.

    Args:
        cfg: Settings instance.
        cloud_logging_sink: Optional Cloud Logging sink to start/close around
            the run. Start/close failures are logged and swallowed — an
            unreachable Cloud Logging backend must never block the
            safety-critical 30 Hz loop from starting or stopping.
    """
    from mousedroid.factory import build_orchestrator
    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    log = get_logger(__name__)
    if cloud_logging_sink is not None:
        try:
            await cloud_logging_sink.start()
        except Exception:
            log.warning("cloud_logging_sink_start_failed", exc_info=True)

    try:
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
    finally:
        if cloud_logging_sink is not None:
            try:
                await cloud_logging_sink.close()
            except Exception:
                log.warning("cloud_logging_sink_close_failed", exc_info=True)


if __name__ == "__main__":  # pragma: no cover
    cli_entry()
