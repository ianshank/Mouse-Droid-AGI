"""Piper TTS voice synthesis latency benchmark.

Measures ``PiperTTS.synthesize()`` latency across one or more configured
personalities, asserts against a configurable p95 target, and prints a
human-readable report.

Usage::

    python3 scripts/benchmark_voice_latency.py \\
        --config config/jetson_production.yaml \\
        --personalities rocky \\
        --n-warmup 3 \\
        --n-iter 20 \\
        --p95-target-ms 500

Exit code 0 = p95 target met for all personalities.
Exit code 1 = target missed or an error occurred.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from mousedroid.config.schema import Settings
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

_NS_PER_MS: float = 1e6

# Sample utterances exercised per benchmark run (varied to avoid caching artefacts).
_BENCH_PHRASES: list[str] = [
    "Rocky ready! Systems online!",
    "Obstacle detected ahead!",
    "Path clear, proceeding forward.",
    "Low battery warning. Must charge.",
    "Navigation success! Arrived.",
]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Piper TTS synthesis latency per personality.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/jetson_production.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--personalities",
        nargs="+",
        default=["rocky"],
        metavar="NAME",
        help="Personality names to benchmark (must map to model paths in config).",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=3,
        help="Warm-up iterations per personality (discarded from timing).",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Timed iterations per personality.",
    )
    parser.add_argument(
        "--p95-target-ms",
        type=float,
        default=500.0,
        help="p95 latency target in milliseconds. Benchmark fails if any personality exceeds this.",
    )
    return parser.parse_args(argv)


def _percentile(samples: list[float], pct: float) -> float:
    """Return the *pct*-th percentile of *samples* (0-100)."""
    arr = np.array(samples, dtype=np.float64)
    return float(np.percentile(arr, pct))


async def _benchmark_personality(
    personality: str,
    cfg: Settings,
    n_warmup: int,
    n_iter: int,
) -> list[float]:
    """Return per-iteration synthesis latencies (ms) for *personality*.

    Args:
        personality: Personality name (overrides cfg.voice.personality at runtime).
        cfg: Root settings.
        n_warmup: Warm-up iterations (timings discarded).
        n_iter: Timed iterations.

    Returns:
        List of latency measurements in milliseconds.
    """
    from mousedroid.config.schema import VoiceConfig
    from mousedroid.voice.tts import PiperTTS

    voice_cfg: VoiceConfig = cfg.voice.model_copy(update={"personality": personality})
    resolved = voice_cfg.resolved_tts_model_path()
    if resolved is None:
        _log.warning(
            "voice_bench_no_model_path",
            personality=personality,
        )
        return []

    _log.info(
        "voice_bench_start",
        personality=personality,
        model_path=resolved,
        n_warmup=n_warmup,
        n_iter=n_iter,
    )

    tts = PiperTTS(voice_cfg)
    tts.start()

    phrases_cycle = [_BENCH_PHRASES[i % len(_BENCH_PHRASES)] for i in range(n_warmup + n_iter)]

    # Warm-up
    for phrase in phrases_cycle[:n_warmup]:
        with torch.no_grad():
            await tts.synthesize(phrase)

    # Timed iterations
    latencies: list[float] = []
    for phrase in phrases_cycle[n_warmup:]:
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            await tts.synthesize(phrase)
        elapsed_ms = (time.perf_counter_ns() - t0) / _NS_PER_MS
        latencies.append(elapsed_ms)

    tts.stop()
    return latencies


def _report(personality: str, latencies: list[float], p95_target_ms: float) -> bool:
    """Print a latency report and return True if p95 target is met.

    Args:
        personality: Label for the report.
        latencies: Per-iteration latencies in milliseconds.
        p95_target_ms: Maximum allowed p95 latency.

    Returns:
        True when the p95 target is met, False otherwise.
    """
    if not latencies:
        print(f"[{personality}] SKIP — no model path configured")
        return True

    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    p99 = _percentile(latencies, 99)
    min_ms = min(latencies)
    max_ms = max(latencies)
    passed = p95 <= p95_target_ms

    status = "PASS" if passed else "FAIL"
    print(
        f"[{personality}] {status}  "
        f"p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms  "
        f"min={min_ms:.1f}ms  max={max_ms:.1f}ms  "
        f"target={p95_target_ms:.0f}ms  n={len(latencies)}"
    )
    return passed


async def _run(args: argparse.Namespace) -> int:
    """Run all personality benchmarks and return exit code.

    Args:
        args: Parsed CLI arguments.

    Returns:
        0 if all targets met, 1 otherwise.
    """
    raw = __import__("yaml").safe_load(args.config.read_text())
    cfg: Settings = Settings.model_validate(raw)

    all_passed = True
    for personality in args.personalities:
        latencies = await _benchmark_personality(
            personality=personality,
            cfg=cfg,
            n_warmup=args.n_warmup,
            n_iter=args.n_iter,
        )
        passed = _report(personality, latencies, args.p95_target_ms)
        if not passed:
            all_passed = False

    return 0 if all_passed else 1


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    exit_code = asyncio.run(_run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
