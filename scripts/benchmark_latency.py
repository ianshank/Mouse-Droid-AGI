"""RSSM + MCTS inference latency benchmark.

Measures ``RSSM.imagine_step()`` and ``MCTSPlanner.plan()`` latencies on the
target device and asserts against configurable pass/fail thresholds.

Usage::

    python3 scripts/benchmark_latency.py \\
        --config config/jetson_production.yaml \\
        --checkpoint weights/rssm/final.pt

Exit code 0 = all targets met. Exit code 1 = one or more targets missed.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mousedroid.config.schema import Settings
from mousedroid.logging.setup import get_logger
from mousedroid.world_model.mcts import MCTSPlanner
from mousedroid.world_model.rssm import RSSM

_log = get_logger(__name__)

_NS_PER_MS: float = 1e6


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark RSSM imagine_step + MCTS plan latency.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("weights/rssm/final.pt"),
        help="Path to RSSM checkpoint (.pt).",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=50,
        help="Warm-up iterations (discarded from timing).",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=1000,
        help="Timed RSSM iterations.",
    )
    parser.add_argument(
        "--n-mcts-iter",
        type=int,
        default=200,
        help="Timed MCTS plan() calls.",
    )
    parser.add_argument(
        "--rssm-target-ms",
        type=float,
        default=15.0,
        help="RSSM p95 latency target in milliseconds.",
    )
    parser.add_argument(
        "--mcts-target-ms",
        type=float,
        default=50.0,
        help="MCTS p95 latency target in milliseconds.",
    )
    return parser.parse_args(argv)


@torch.no_grad()
def _benchmark_rssm(
    model: RSSM,
    n_warmup: int,
    n_iter: int,
    device: torch.device,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Run timed RSSM imagine_step loop.

    Args:
        model: Loaded RSSM model.
        n_warmup: Number of warm-up iterations to discard.
        n_iter: Number of timed iterations to collect.
        device: Target device.

    Returns:
        Array of per-call latencies in milliseconds, shape ``(n_iter,)``.
    """
    cfg = model._cfg
    h = torch.zeros(1, cfg.hidden_dim, device=device)
    z = torch.zeros(1, cfg.latent_dim, device=device)
    action = torch.zeros(1, cfg.action_dim, device=device)

    # Warm-up
    for _ in range(n_warmup):
        h, z, _ = model.imagine_step(action, h, z)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    latencies: list[float] = []
    for _ in range(n_iter):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter_ns()
        h, z, _ = model.imagine_step(action, h, z)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / _NS_PER_MS)

    return np.array(latencies, dtype=np.float64)


@torch.no_grad()
def _benchmark_mcts(
    planner: MCTSPlanner,
    model: RSSM,
    n_warmup: int,
    n_iter: int,
    n_simulations: int,
    device: torch.device,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Run timed MCTSPlanner.plan() loop.

    Args:
        planner: Configured MCTSPlanner.
        model: RSSM used as world model (for resetting h/z).
        n_warmup: Warm-up calls to discard.
        n_iter: Number of timed calls to collect.
        n_simulations: Simulations per plan() call.
        device: Target device.

    Returns:
        Array of per-call latencies in milliseconds, shape ``(n_iter,)``.
    """
    cfg = model._cfg
    h = torch.zeros(1, cfg.hidden_dim, device=device)
    z = torch.zeros(1, cfg.latent_dim, device=device)

    for _ in range(n_warmup):
        planner.plan(h, z, n_simulations=n_simulations)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    latencies: list[float] = []
    for _ in range(n_iter):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter_ns()
        planner.plan(h, z, n_simulations=n_simulations)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / _NS_PER_MS)

    return np.array(latencies, dtype=np.float64)


def _report(
    label: str, latencies: np.ndarray[Any, np.dtype[np.float64]], target_p95_ms: float
) -> bool:
    """Log percentile summary and return True if target met.

    Args:
        label: Component name for log context.
        latencies: Per-call latency array in milliseconds.
        target_p95_ms: Pass/fail threshold for p95.

    Returns:
        True if p95 <= target_p95_ms, else False.
    """
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    passed = p95 <= target_p95_ms

    _log.info(
        "latency_report",
        component=label,
        p50_ms=round(p50, 3),
        p95_ms=round(p95, 3),
        p99_ms=round(p99, 3),
        target_p95_ms=target_p95_ms,
        passed=passed,
        n_samples=len(latencies),
    )
    return passed


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Optional argument list for testing.

    Returns:
        Exit code: 0 = all targets met, 1 = target(s) missed.
    """
    args = _parse_args(argv)

    # Load config — Settings reads YAML via MOUSEDROID_CONFIG or env vars;
    # here we resolve the path and pass it through _env_file to keep DI clean.
    cfg = Settings(_env_file=None)  # type: ignore[call-arg]
    if args.config.exists():
        import yaml

        raw = yaml.safe_load(args.config.read_text())
        cfg = Settings.model_validate(raw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log.info("benchmark_start", device=str(device), config=str(args.config))

    # Build RSSM
    model = RSSM(cfg.model)
    model.to(device)
    model.eval()

    if args.checkpoint.exists():
        state = torch.load(str(args.checkpoint), map_location=device, weights_only=True)
        # Checkpoints may be stored as {"model_state_dict": …} or bare state dicts.
        state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict)
        _log.info("checkpoint_loaded", path=str(args.checkpoint))
    else:
        _log.warning("checkpoint_not_found_using_random_weights", path=str(args.checkpoint))

    # Build MCTS planner — uses RSSM as world model
    planner = MCTSPlanner(cfg.mcts, model, action_dim=cfg.model.action_dim)

    # Run RSSM benchmark
    _log.info("rssm_benchmark_starting", n_warmup=args.n_warmup, n_iter=args.n_iter)
    rssm_latencies = _benchmark_rssm(model, args.n_warmup, args.n_iter, device)
    rssm_ok = _report("rssm_imagine_step", rssm_latencies, args.rssm_target_ms)

    # Run MCTS benchmark
    _log.info("mcts_benchmark_starting", n_warmup=10, n_iter=args.n_mcts_iter)
    mcts_latencies = _benchmark_mcts(
        planner,
        model,
        n_warmup=10,
        n_iter=args.n_mcts_iter,
        n_simulations=cfg.mcts.n_simulations_max,
        device=device,
    )
    mcts_ok = _report("mcts_plan", mcts_latencies, args.mcts_target_ms)

    all_ok = rssm_ok and mcts_ok
    _log.info("benchmark_complete", passed=all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
