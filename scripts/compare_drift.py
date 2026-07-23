#!/usr/bin/env python3
"""Baseline-vs-corrupted drift comparison harness (F-023, ADR-015).

Trains two identically-initialised RSSMs — standard objective vs
corrupted-history augmentation — and reports deterministic open-loop drift on
a held-out batch (range headline channel; documented negative results are an
acceptable outcome — the script reports, it does not gate unless
``--gate-max-regression`` is passed).

Deliberately FLAG-INDEPENDENT of ``training.drift.enabled`` (that flag gates
the production-pretraining integration only). Default episodes are seeded
synthetic random-walk dynamics (no MuJoCo dependency — CI-safe); pass
``--mujoco`` to use the real ``SimEpisodeGenerator`` (requires the ``[arm]``
extra).

The optional ``--memory on|both`` ablation measures the F-023
``BoundedContextMemory`` at the ``measure_drift`` posterior-warmup seam on a
single-episode batch. NOTE: it operates on the plain RSSM latent
(``hidden_dim``), distinct from the deployed DualStream combined latent.

Usage:
    python scripts/compare_drift.py --synthetic --episodes 8 --seq-len 48 \
        --steps 60 --seed 42 --out reports/drift_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import DriftTrainingConfig, ModelConfig, Settings
from mousedroid.constants import SENSOR_SLOT_MAP
from mousedroid.training.drift_metrics import DriftReport, measure_drift
from mousedroid.training.drift_reduction import (
    DriftComparisonResult,
    train_pair_and_compare,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Settings YAML")
    parser.add_argument("--episodes", type=int, default=8, help="Episodes per batch (B)")
    parser.add_argument("--seq-len", type=int, default=48, help="Sequence length (T)")
    parser.add_argument("--steps", type=int, default=60, help="Optimisation steps per arm")
    parser.add_argument("--train-batches", type=int, default=4, help="Distinct train batches")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Override drift seed")
    parser.add_argument("--corruption-prob", type=float, default=None)
    parser.add_argument("--max-prefix-frac", type=float, default=None)
    parser.add_argument("--recovery-weight", type=float, default=None)
    parser.add_argument("--context-steps", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument(
        "--no-residual-head", action="store_true", help="Skip the evaluation-only head"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=True,
        help="Seeded synthetic episodes (default; CI-safe)",
    )
    parser.add_argument(
        "--mujoco",
        action="store_true",
        help="Use SimEpisodeGenerator episodes instead (requires [arm] extra)",
    )
    parser.add_argument(
        "--memory",
        choices=("off", "on", "both"),
        default="off",
        help="Optional bounded-context memory ablation at the warmup seam",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Target device for both arms (auto = cuda when available, else cpu)",
    )
    parser.add_argument("--out", default="reports/drift_comparison.json")
    parser.add_argument(
        "--gate-max-regression",
        type=float,
        default=None,
        help=(
            "Exit 1 if augmented headline mean MSE exceeds baseline * (1 + G). "
            "Unset (default): report-only, exit 0."
        ),
    )
    return parser.parse_args(argv)


def _drift_cfg(cfg: Settings, args: argparse.Namespace) -> DriftTrainingConfig:
    base = cfg.training.drift or DriftTrainingConfig()
    overrides: dict[str, object] = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.corruption_prob is not None:
        overrides["corruption_prob"] = args.corruption_prob
    if args.max_prefix_frac is not None:
        overrides["max_prefix_frac"] = args.max_prefix_frac
    if args.recovery_weight is not None:
        overrides["recovery_weight"] = args.recovery_weight
    if args.context_steps is not None:
        overrides["eval_context_steps"] = args.context_steps
    if args.horizon is not None:
        overrides["eval_horizon"] = args.horizon
    if args.no_residual_head:
        overrides["residual_head"] = False
    return base.model_copy(update=overrides)


def _model_cfg(cfg: Settings) -> ModelConfig:
    """Vision-OFF model config (the build_rssm_trainable convention)."""
    return ModelConfig.model_validate(
        {**cfg.model.model_dump(), "vision_dim": 0, "vision_proj_dim": 0}
    )


def _synthetic_batch(
    mcfg: ModelConfig, *, episodes: int, seq_len: int, generator: torch.Generator
) -> dict[str, Tensor]:
    """Seeded random-walk episode batch with action-coupled learnable dynamics."""
    b, t = episodes, seq_len
    action = torch.zeros(b, t, mcfg.action_dim)
    motor = torch.zeros(b, t, mcfg.motor_state_dim)
    ultra = torch.zeros(b, t, max(1, mcfg.ultrasonic_dim))
    a = torch.zeros(b, mcfg.action_dim)
    m = torch.zeros(b, mcfg.motor_state_dim)
    m[:, -1] = 1.0  # battery starts full
    r = torch.rand(b, ultra.shape[-1], generator=generator) * 2.0 + 1.0
    for step in range(t):
        a = 0.7 * a + 0.3 * torch.randn(b, mcfg.action_dim, generator=generator)
        vel_dims = min(mcfg.action_dim, mcfg.motor_state_dim - 1)
        m = m.clone()
        m[:, :vel_dims] = (
            0.85 * m[:, :vel_dims]
            + 0.15 * a[:, :vel_dims]
            + 0.02 * torch.randn(b, vel_dims, generator=generator)
        )
        m[:, -1] = (m[:, -1] - 0.0005).clamp(min=0.0)  # slow battery drain
        # Range is coupled to forward velocity — the environment-shaped signal.
        r = (r - 0.05 * m[:, :1] + 0.01 * torch.randn(b, r.shape[-1], generator=generator)).clamp(
            0.1, 4.0
        )
        action[:, step] = a
        motor[:, step] = m
        ultra[:, step] = r
    n_slots = len(SENSOR_SLOT_MAP)
    valid = torch.zeros(b, t, n_slots)
    valid[..., SENSOR_SLOT_MAP["motor"]] = 1.0
    valid[..., SENSOR_SLOT_MAP["ultrasonic"]] = 1.0
    batch = {"motor": motor, "ultrasonic": ultra, "valid_mask": valid, "action": action}
    if mcfg.lidar_dim > 0:
        # Zero-filled placeholder so a lidar-enabled model config still runs;
        # measure_drift deliberately EXCLUDES zero-filled channels from scoring.
        batch["lidar"] = torch.zeros(b, t, mcfg.lidar_dim)
    return batch


def _mujoco_batches(
    cfg: Settings, *, episodes: int, seq_len: int, n_batches: int
) -> list[dict[str, Tensor]]:
    """Opt-in real-sim episodes via SimEpisodeGenerator (needs mujoco)."""
    from mousedroid.factory import build_rover_env
    from mousedroid.training.rover_obs_adapter import RoverObsAdapter
    from mousedroid.training.sim_episode_generator import SimEpisodeGenerator

    tcfg = cfg.training
    env = build_rover_env(cfg)
    try:
        adapter = RoverObsAdapter(battery_v=cfg.rover.sim.battery_v)
        generator = SimEpisodeGenerator(
            env,
            adapter,
            n_episodes=episodes,
            seq_len=seq_len,
            seed=tcfg.rssm_data_seed,
            explore_action_rad_s=tcfg.rssm_explore_action_rad_s,
            explore_smoothing=tcfg.rssm_explore_smoothing,
        )
        batches: list[dict[str, Tensor]] = []
        for _ in range(n_batches):
            ep = generator.generate()
            batches.append(
                {
                    "motor": ep.motor,
                    "ultrasonic": ep.ultrasonic,
                    "lidar": ep.lidar,
                    "valid_mask": ep.valid_mask,
                    "action": ep.action,
                }
            )
        return batches
    finally:
        env.close()


def _memory_ablation(
    cfg: Settings,
    mcfg: ModelConfig,
    drift_cfg: DriftTrainingConfig,
    single_episode_batch: dict[str, Tensor],
    device: torch.device | None,
) -> dict[str, object]:
    """Memory-off vs memory-on drift on a FRESH seeded model pair (B=1).

    The F-023 memory is applied at the ``measure_drift`` posterior-warmup seam
    only — mirroring the deployment observe seam.
    """
    from mousedroid.config.schema import WorldModelMemoryConfig
    from mousedroid.training.drift_reduction import _seeded_model_pair
    from mousedroid.world_model.bounded_context import BoundedContextMemory

    memory_cfg = cfg.world_model_memory or WorldModelMemoryConfig.model_validate({"enabled": True})
    if memory_cfg.sink_warmup_ticks >= drift_cfg.eval_context_steps:
        print(
            f"WARNING: sink_warmup_ticks ({memory_cfg.sink_warmup_ticks}) >= "
            f"context_steps ({drift_cfg.eval_context_steps}) — the sink is never "
            "captured during warmup, so this ablation measures the ring/EMA only."
        )
    model, decoders = _seeded_model_pair(mcfg, drift_cfg.seed, device)
    context = BoundedContextMemory(memory_cfg, h_dim=mcfg.hidden_dim, z_dim=mcfg.latent_dim)
    off = measure_drift(
        model,
        single_episode_batch,
        decoders,
        context_steps=drift_cfg.eval_context_steps,
        horizon=drift_cfg.eval_horizon,
        seed=drift_cfg.seed,
    )
    on = measure_drift(
        model,
        single_episode_batch,
        decoders,
        context_steps=drift_cfg.eval_context_steps,
        horizon=drift_cfg.eval_horizon,
        seed=drift_cfg.seed,
        latent_context=context,
    )
    return {
        "note": "RSSM-latent ablation (not the deployed DualStream combined latent)",
        "memory_off": _report_dict(off),
        "memory_on": _report_dict(on),
    }


def _report_dict(report: DriftReport) -> dict[str, object]:
    return {
        "headline_channel": report.headline_channel,
        "means": {ch: report.mean(ch) for ch in report.channels()},
        "finals": {ch: report.final(ch) for ch in report.channels()},
        "per_step_mse": {ch: list(curve) for ch, curve in report.per_step_mse.items()},
    }


def _print_table(result: DriftComparisonResult) -> None:
    channel = result.baseline.headline_channel
    print("\n| metric | baseline | augmented |")
    print("|---|---|---|")
    for ch in result.baseline.channels():
        aug = result.augmented.mean(ch) if ch in result.augmented.channels() else float("nan")
        star = " (headline)" if ch == channel else ""
        print(f"| mean {ch}{star} | {result.baseline.mean(ch):.6f} | {aug:.6f} |")
    print(
        f"| final {channel} | {result.baseline.final(channel):.6f} "
        f"| {result.augmented.final(channel):.6f} |"
    )
    print(
        f"\nheadline improvement (baseline - augmented, positive = better): "
        f"{result.headline_improvement:.6f} over {result.steps} steps "
        f"({result.corrupted_batches} corrupted batches)\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_settings(Path(args.config))
    drift_cfg = _drift_cfg(cfg, args)
    mcfg = _model_cfg(cfg)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(drift_cfg.seed)
    if args.mujoco:
        train_batches = _mujoco_batches(
            cfg, episodes=args.episodes, seq_len=args.seq_len, n_batches=args.train_batches
        )
        held_out = _mujoco_batches(cfg, episodes=args.episodes, seq_len=args.seq_len, n_batches=1)[
            0
        ]
    else:
        train_batches = [
            _synthetic_batch(mcfg, episodes=args.episodes, seq_len=args.seq_len, generator=gen)
            for _ in range(args.train_batches)
        ]
        # Held-out draws AFTER the train batches from the same stream — disjoint
        # by construction (fresh random-walk trajectories).
        held_out = _synthetic_batch(
            mcfg, episodes=args.episodes, seq_len=args.seq_len, generator=gen
        )

    device = None if args.device == "auto" else torch.device(args.device)
    result = train_pair_and_compare(
        mcfg,
        drift_cfg,
        train_batches,
        held_out,
        steps=args.steps,
        learning_rate=args.lr,
        device=device,
    )
    _print_table(result)

    report: dict[str, object] = {
        "change": "mouse-droid-alayaworld-memory-distill (F-023)",
        "drift_cfg": drift_cfg.model_dump(),
        "episodes": args.episodes,
        "seq_len": args.seq_len,
        "steps": result.steps,
        "corrupted_batches": result.corrupted_batches,
        "baseline": _report_dict(result.baseline),
        "augmented": _report_dict(result.augmented),
        "headline_improvement": result.headline_improvement,
        "source": "mujoco" if args.mujoco else "synthetic",
    }
    if args.memory in ("on", "both"):
        single = _synthetic_batch(mcfg, episodes=1, seq_len=args.seq_len, generator=gen)
        report["memory_ablation"] = _memory_ablation(cfg, mcfg, drift_cfg, single, device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"report written to {out_path}")

    if args.gate_max_regression is not None:
        channel = result.baseline.headline_channel
        ceiling = result.baseline.mean(channel) * (1.0 + args.gate_max_regression)
        if result.augmented.mean(channel) > ceiling:
            print(
                f"GATE FAILED: augmented mean {channel} MSE "
                f"{result.augmented.mean(channel):.6f} > ceiling {ceiling:.6f}"
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
