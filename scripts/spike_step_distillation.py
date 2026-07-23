#!/usr/bin/env python3
"""SPIKE — k-step imagination distillation feasibility (F-023). NON-PRODUCTION.

Time-boxed, non-binding evaluation of whether k composed ``imagine_step``
calls can be distilled into ONE forward pass of a compact "jump" student —
the RSSM analog of AlayaWorld's step-count distillation (video-diffusion
steps do not transfer; the honest mapping here is the MCTS planner's
sequential imagination).

Honesty framing (baked into the go/no-go template this feeds,
``docs/analysis/alayaworld-distillation-spike.md``):

- **Teacher is the deterministic prior-MEAN k-step composition** (no
  sampling) predicting ``(h_k, z_k, gamma-discounted k-step return)`` with
  ``gamma`` from ``MCTSConfig`` — matching ``MCTSPlanner._rollout``'s actual
  discounted accumulator. A stochastic teacher has an irreducible MSE floor
  and does not agree with itself across seeds.
- **Latency compares the deployed primitive** (k sequential STOCHASTIC
  ``imagine_step`` calls) against one student forward — the primitive-level
  speedup. The CONSUMER-level ceiling is separate: MCTS ``plan()`` makes
  roughly 500-650 ``imagine_step`` calls at defaults, of which the depth-5
  rollouts are only ~40% (tree expansion needs intermediate states), capping
  end-to-end planner gain at ~1.25-1.6x regardless of primitive speedup.
- **Agreement** = argmax over a fixed candidate-action grid
  (``MCTSConfig.n_action_candidates``) of the predicted k-step discounted
  return, teacher-mean vs student.

Numbers produced in-container are CPU-relative; the Jetson measurement is an
operator run (``docs/runbooks/jetson-alayaworld-spike.md``).

Usage:
    python scripts/spike_step_distillation.py --k 2,4,8 --distill-steps 200 \
        --trials 200 --seed 42 --out reports/spike_step_distillation.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousedroid.common.torch_device import resolve_device
from mousedroid.config.loader import load_settings
from mousedroid.config.schema import ModelConfig, Settings
from mousedroid.growth.distillation import KnowledgeDistiller
from mousedroid.validation.latency_stats import summarize
from mousedroid.world_model.rssm import RSSM

# Consumer-ceiling constants for the report (verified against MCTSPlanner at
# defaults: 9-candidate expand + 50 sims x depth-5 rollouts + re-expansions).
_PLAN_CALLS_ESTIMATE = "500-650"
_ROLLOUT_SHARE = "~40%"
_CONSUMER_CEILING = "~1.25-1.6x"


class KStepTeacherAdapter(nn.Module):
    """Paramless deterministic prior-MEAN k-step teacher.

    The RSSM is deliberately NOT registered as a child module
    (``object.__setattr__``) so the distiller's teacher-freeze loop is a
    no-op and never mutates the shared model — the ``VLATeacherModule``
    paramless-adapter pattern.
    """

    def __init__(self, model: RSSM, *, k: int, gamma: float) -> None:
        super().__init__()
        object.__setattr__(self, "_model", model)
        self._k = k
        self._gamma = gamma
        self._h_dim = model.cfg.hidden_dim
        self._z_dim = model.cfg.latent_dim
        self._a_dim = model.cfg.action_dim

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Compose k prior-MEAN steps: x = [h | z | a_1..a_k] -> [h_k | z_k | return]."""
        model: RSSM = self._model
        h = x[:, : self._h_dim]
        z = x[:, self._h_dim : self._h_dim + self._z_dim]
        actions = x[:, self._h_dim + self._z_dim :]
        ret = torch.zeros(x.shape[0], 1, device=x.device)
        for i in range(self._k):
            action = actions[:, i * self._a_dim : (i + 1) * self._a_dim]
            h = model.gru(torch.cat([z, action], dim=-1), h)
            prior_mean, _prior_logvar = model.prior(h).chunk(2, dim=-1)
            z = prior_mean  # deterministic: the mean, never a sample
            reward = model.reward_head(torch.cat([h, z], dim=-1))
            ret = ret + (self._gamma**i) * reward
        return torch.cat([h, z, ret], dim=-1)


class KStepJumpStudent(nn.Module):
    """Compact one-forward jump student: [h | z | a_1..a_k] -> [h_k | z_k | return]."""

    def __init__(self, cfg: ModelConfig, *, k: int, hidden: int) -> None:
        super().__init__()
        in_dim = cfg.hidden_dim + cfg.latent_dim + k * cfg.action_dim
        out_dim = cfg.hidden_dim + cfg.latent_dim + 1
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.net(x)
        return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--k", default="2,4,8", help="Comma-separated jump lengths")
    parser.add_argument("--distill-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-states", type=int, default=512, help="On-distribution states")
    parser.add_argument("--trials", type=int, default=200, help="Latency timing trials")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--student-hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint", default=None, help="Optional trained RSSM checkpoint (migrated)"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Target device (auto = cuda when available, else cpu)",
    )
    parser.add_argument("--out", default="reports/spike_step_distillation.json")
    return parser.parse_args(argv)


def _sync(device: torch.device) -> None:
    """Synchronise CUDA before reading the wall clock (honest GPU latency)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_cfg(cfg: Settings) -> ModelConfig:
    return ModelConfig.model_validate(
        {**cfg.model.model_dump(), "vision_dim": 0, "vision_proj_dim": 0}
    )


def _build_model(cfg: Settings, mcfg: ModelConfig, checkpoint: str | None) -> RSSM:
    if checkpoint is not None:
        from mousedroid.world_model.checkpoint_migration import load_rssm_with_migration

        return load_rssm_with_migration(Path(checkpoint), mcfg, torch.device("cpu"))
    return RSSM(mcfg)


@torch.no_grad()
def _roll_states(
    model: RSSM, *, n: int, gen: torch.Generator, device: torch.device
) -> tuple[Tensor, Tensor]:
    """On-distribution (h, z) pairs: prior rollout from zeros under random actions.

    Random draws come from the CPU ``gen`` (device-independent determinism)
    and are moved to ``device`` before use.
    """
    cfg = model.cfg
    h = torch.zeros(n, cfg.hidden_dim, device=device)
    z = torch.zeros(n, cfg.latent_dim, device=device)
    warmup = 10  # hardcoded-ok: spike-only mixing steps to leave the zero state
    for _ in range(warmup):
        action = torch.tanh(torch.randn(n, cfg.action_dim, generator=gen)).to(device)
        h = model.gru(torch.cat([z, action], dim=-1), h)
        prior_mean, prior_logvar = model.prior(h).chunk(2, dim=-1)
        eps = torch.randn(prior_mean.shape, generator=gen).to(device)
        z = prior_mean + torch.exp(0.5 * prior_logvar) * eps
    return h, z


def _inputs(h: Tensor, z: Tensor, *, k: int, a_dim: int, gen: torch.Generator) -> Tensor:
    actions = torch.tanh(torch.randn(h.shape[0], k * a_dim, generator=gen)).to(h.device)
    return torch.cat([h, z, actions], dim=-1)


def _action_grid_agreement(
    teacher: KStepTeacherAdapter,
    student: KStepJumpStudent,
    h: Tensor,
    z: Tensor,
    *,
    k: int,
    a_dim: int,
    n_candidates: int,
    gen: torch.Generator,
) -> float:
    """Argmax-over-candidate-grid agreement on predicted discounted return."""
    matches = 0
    n = h.shape[0]
    ret_idx = -1  # return is the last output column
    for i in range(n):
        cand_actions = torch.tanh(torch.randn(n_candidates, k * a_dim, generator=gen)).to(h.device)
        hi = h[i : i + 1].expand(n_candidates, -1)
        zi = z[i : i + 1].expand(n_candidates, -1)
        x = torch.cat([hi, zi, cand_actions], dim=-1)
        with torch.no_grad():
            teacher_ret = teacher(x)[:, ret_idx]
            student_ret = student(x)[:, ret_idx]
        if int(teacher_ret.argmax()) == int(student_ret.argmax()):
            matches += 1
    return matches / max(1, n)


def _time_primitive(
    model: RSSM, *, k: int, trials: int, gen: torch.Generator, device: torch.device
) -> list[float]:
    """Wall time of k sequential (stochastic) imagine_step calls — the deployed leg."""
    cfg = model.cfg
    samples_ms: list[float] = []
    h = torch.zeros(1, cfg.hidden_dim, device=device)
    z = torch.zeros(1, cfg.latent_dim, device=device)
    action = torch.tanh(torch.randn(1, cfg.action_dim, generator=gen)).to(device)
    for _ in range(trials):
        _sync(device)
        start = time.perf_counter()
        hh, zz = h, z
        for _ in range(k):
            hh, zz, _reward = model.imagine_step(action, hh, zz)
        _sync(device)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return samples_ms


def _time_student(
    student: KStepJumpStudent, x: Tensor, *, trials: int, device: torch.device
) -> list[float]:
    samples_ms: list[float] = []
    single = x[:1]
    with torch.no_grad():
        for _ in range(trials):
            _sync(device)
            start = time.perf_counter()
            student(single)
            _sync(device)
            samples_ms.append((time.perf_counter() - start) * 1000.0)
    return samples_ms


def _run_for_k(
    model: RSSM,
    cfg: Settings,
    mcfg: ModelConfig,
    args: argparse.Namespace,
    k: int,
    device: torch.device,
) -> dict[str, object]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed + k)
    torch.manual_seed(args.seed + k)
    gamma = cfg.mcts.gamma
    teacher = KStepTeacherAdapter(model, k=k, gamma=gamma)
    student = KStepJumpStudent(mcfg, k=k, hidden=args.student_hidden).to(device)
    distiller = KnowledgeDistiller(
        teacher, student, temperature=1.0, alpha=1.0, lr=args.lr, objective="regression"
    )

    h, z = _roll_states(model, n=args.n_states, gen=gen, device=device)
    losses: list[float] = []
    for _ in range(args.distill_steps):
        idx = torch.randint(0, args.n_states, (args.batch_size,), generator=gen)
        x = _inputs(h[idx], z[idx], k=k, a_dim=mcfg.action_dim, gen=gen)
        losses.append(float(distiller.distill_step(x).detach()))

    h_eval, z_eval = _roll_states(model, n=64, gen=gen, device=device)
    x_eval = _inputs(h_eval, z_eval, k=k, a_dim=mcfg.action_dim, gen=gen)
    student.eval()
    with torch.no_grad():
        teacher_out = teacher(x_eval)
        student_out = student(x_eval)
    hz_dim = mcfg.hidden_dim + mcfg.latent_dim
    eval_mse = {
        "hz": float(((teacher_out[:, :hz_dim] - student_out[:, :hz_dim]) ** 2).mean()),
        "return": float(((teacher_out[:, -1] - student_out[:, -1]) ** 2).mean()),
    }
    agreement = _action_grid_agreement(
        teacher,
        student,
        h_eval,
        z_eval,
        k=k,
        a_dim=mcfg.action_dim,
        n_candidates=cfg.mcts.n_action_candidates,
        gen=gen,
    )
    primitive = summarize(_time_primitive(model, k=k, trials=args.trials, gen=gen, device=device))
    student_lat = summarize(_time_student(student, x_eval, trials=args.trials, device=device))
    speedup_p50 = primitive.p50_ms / max(student_lat.p50_ms, 1e-9)
    speedup_p95 = primitive.p95_ms / max(student_lat.p95_ms, 1e-9)
    return {
        "k": k,
        "distill_loss_first": losses[0],
        "distill_loss_last": losses[-1],
        "eval_mse": eval_mse,
        "action_agreement": agreement,
        "primitive_latency_ms": {
            "p50": primitive.p50_ms,
            "p95": primitive.p95_ms,
            "p99": primitive.p99_ms,
        },
        "student_latency_ms": {
            "p50": student_lat.p50_ms,
            "p95": student_lat.p95_ms,
            "p99": student_lat.p99_ms,
        },
        "primitive_speedup_p50": speedup_p50,
        "primitive_speedup_p95": speedup_p95,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_settings(Path(args.config))
    mcfg = _model_cfg(cfg)
    device = resolve_device(args.device)
    model = _build_model(cfg, mcfg, args.checkpoint).to(device)
    model.eval()

    ks = [int(v) for v in str(args.k).split(",") if v.strip()]
    results = [_run_for_k(model, cfg, mcfg, args, k, device) for k in ks]

    print("\n| k | agreement | primitive p95 (ms) | student p95 (ms) | speedup p95 |")
    print("|---|---|---|---|---|")
    for row in results:
        prim = row["primitive_latency_ms"]["p95"]  # type: ignore[index]
        stud = row["student_latency_ms"]["p95"]  # type: ignore[index]
        print(
            f"| {row['k']} | {row['action_agreement']:.3f} | {prim:.3f} "
            f"| {stud:.3f} | {row['primitive_speedup_p95']:.2f}x |"
        )
    print(
        f"\nCONSUMER CEILING: MCTS plan() makes ~{_PLAN_CALLS_ESTIMATE} imagine_step "
        f"calls; rollouts are {_ROLLOUT_SHARE} of them, so end-to-end planner gain "
        f"caps at {_CONSUMER_CEILING} regardless of the primitive speedup above.\n"
        "These numbers are CPU-relative; the Jetson run is the operator gate "
        "(docs/runbooks/jetson-alayaworld-spike.md)."
    )

    report = {
        "spike": "alayaworld-step-distillation (F-023, non-binding)",
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "distill_steps": args.distill_steps,
        "student_hidden": args.student_hidden,
        "results": results,
        "consumer_ceiling": {
            "plan_imagine_calls_estimate": _PLAN_CALLS_ESTIMATE,
            "rollout_share": _ROLLOUT_SHARE,
            "end_to_end_ceiling": _CONSUMER_CEILING,
        },
        "device": str(device),
        "environment": (f"container-{device.type} (Jetson measurement pending operator run)"),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
