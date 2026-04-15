"""5-minute full-stack orchestrator endurance test.

Starts the full ``MouseDroidOrchestrator`` sense-plan-act loop and runs it
for a configurable duration, sampling health metrics at regular intervals.
Prints a structured summary on completion.

Usage::

    python3 scripts/endurance_test.py \\
        --config config/jetson_production.yaml \\
        --duration 300

Exit code 0 = test passed. Exit code 1 = OOM / crash / thermal / violation.

**SAFETY NOTE**: This script activates motors and all sensors. Only run after
receiving explicit human approval at the confirmation gate in the deployment
plan.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

import torch

from mousedroid.config.schema import Settings
from mousedroid.factory import build_health_monitor, build_orchestrator
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

_NS_PER_MS: float = 1e6
# Ring buffer capacity for tick latency samples (approx. 30 Hz x 300 s = 9000 ticks).
_LATENCY_RING_SIZE: int = 10_000


class _OrchestratorProto(Protocol):
    """Minimal structural protocol for orchestrator lifecycle methods."""

    async def start(self) -> None:
        """Start all subsystems."""
        ...

    async def tick(self) -> None:
        """Execute one sense-plan-act cycle."""
        ...

    async def stop(self) -> None:
        """Stop all subsystems gracefully."""
        ...


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="5-minute full-stack orchestrator endurance test.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="Run duration in seconds (default 300).",
    )
    parser.add_argument(
        "--health-interval",
        type=float,
        default=10.0,
        help="Seconds between health snapshots (default 10).",
    )
    return parser.parse_args(argv)


def _percentile(data: list[float], pct: float) -> float:
    """Compute percentile from a list of values without scipy.

    Args:
        data: Unsorted latency values.
        pct: Percentile in [0, 100].

    Returns:
        Percentile value, or 0.0 if data is empty.
    """
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (pct / 100.0) * (len(sorted_data) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac


async def _run(args: argparse.Namespace) -> int:
    """Async entry point.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code: 0 = passed, 1 = failed.
    """
    # Load config
    cfg = Settings(_env_file=None)  # type: ignore[call-arg]
    if args.config.exists():
        import yaml

        raw = yaml.safe_load(args.config.read_text())
        cfg = Settings.model_validate(raw)

    _log.info(
        "endurance_test_start",
        duration_s=args.duration,
        health_interval_s=args.health_interval,
        config=str(args.config),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Build orchestrator via factory (single DI wiring point)
    orch = cast(_OrchestratorProto, build_orchestrator(cfg))
    health_monitor = build_health_monitor(cfg)

    # State tracking
    tick_latencies: deque[float] = deque(maxlen=_LATENCY_RING_SIZE)
    health_snapshots: list[dict[str, object]] = []
    gpu_temp_max: float = 0.0
    total_ticks: int = 0
    failed = False

    deadline = time.monotonic() + args.duration
    next_health_check = time.monotonic() + args.health_interval

    try:
        await orch.start()

        while time.monotonic() < deadline:
            # Tick with torch.no_grad() guard around inference paths
            with torch.no_grad():
                t0 = time.perf_counter_ns()
                await orch.tick()
                t1 = time.perf_counter_ns()

            tick_latencies.append((t1 - t0) / _NS_PER_MS)
            total_ticks += 1

            # Periodic health sampling
            if time.monotonic() >= next_health_check:
                snapshot = await health_monitor.check_health()
                health_snapshots.append(snapshot)
                temp_c = cast(float, snapshot.get("gpu_temp_c", 0.0))
                if temp_c > gpu_temp_max:
                    gpu_temp_max = temp_c
                _log.info(
                    "health_snapshot",
                    elapsed_s=round(args.duration - (deadline - time.monotonic()), 1),
                    **snapshot,
                )
                next_health_check = time.monotonic() + args.health_interval

        await orch.stop()

    except MemoryError:
        _log.error("endurance_test_oom")
        failed = True
    except Exception as exc:  # pylint: disable=broad-except
        _log.error("endurance_test_crash", error=str(exc))
        failed = True

    # Memory high-water mark (resident set size in bytes -> MiB, Linux only)
    if sys.platform != "win32":
        import resource

        rusage = resource.getrusage(resource.RUSAGE_SELF)
        rss_mib = rusage.ru_maxrss / 1024  # Linux reports in KiB
    else:
        rss_mib = 0.0

    latency_list = list(tick_latencies)
    p50 = _percentile(latency_list, 50)
    p95 = _percentile(latency_list, 95)
    p99 = _percentile(latency_list, 99)
    lat_max = max(latency_list) if latency_list else 0.0

    _log.info(
        "endurance_test_summary",
        total_ticks=total_ticks,
        duration_s=round(args.duration, 1),
        tick_p50_ms=round(p50, 3),
        tick_p95_ms=round(p95, 3),
        tick_p99_ms=round(p99, 3),
        tick_max_ms=round(lat_max, 3),
        gpu_temp_max_c=round(gpu_temp_max, 1),
        memory_high_water_mib=round(rss_mib, 1),
        health_samples=len(health_snapshots),
        passed=not failed,
    )

    # Gate: temperature
    if gpu_temp_max >= cfg.health.gpu_temp_critical_c:
        _log.error(
            "endurance_gate_failed_thermal",
            gpu_temp_max_c=gpu_temp_max,
            critical_threshold_c=cfg.health.gpu_temp_critical_c,
        )
        failed = True

    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Optional argument list for testing.

    Returns:
        Exit code: 0 = passed, 1 = failed.
    """
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
