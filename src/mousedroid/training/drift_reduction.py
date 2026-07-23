"""Paired baseline-vs-corrupted drift-training comparison (F-023, ADR-015).

:func:`train_pair_and_compare` trains two RSSMs from IDENTICAL seeded
initialisations — one with the standard ``train_sequence`` objective, one with
the corrupted-history ``train_sequence_corrupted`` objective — and scores both
with the deterministic :func:`~mousedroid.training.drift_metrics.measure_drift`
harness on a held-out batch. A documented negative result (augmented worse
than baseline) is an acceptable outcome; the harness reports, it does not
gate.

Parity discipline: both arms re-seed the global RNG identically before every
optimisation step, so with ``corruption_prob=0`` the augmented arm reproduces
the baseline arm EXACTLY (pinned by
``tests/unit/training/test_drift_reduction.py``) — differences can only come
from the corruption objective, never from RNG stream divergence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from mousedroid.config.schema import DriftTrainingConfig, ModelConfig
from mousedroid.logging.setup import get_logger
from mousedroid.training.drift_metrics import DriftReport, measure_drift
from mousedroid.world_model.rssm import RSSM, DriftCorrectionHead, RawModalityDecoders

_log = get_logger(__name__)


@dataclass(frozen=True)
class DriftComparisonResult:
    """Outcome of one paired baseline-vs-augmented comparison.

    Attributes:
        baseline: Drift report for the standard-objective model.
        augmented: Drift report for the corrupted-history model.
        baseline_final_train_loss: Last training-step loss of the baseline arm.
        augmented_final_train_loss: Last training-step loss of the augmented arm.
        steps: Optimisation steps each arm ran.
        seed: Seed the pair was pinned to.
        corrupted_batches: How many augmented-arm steps used the corrupted
            objective (the rest fell through to ``train_sequence``).
    """

    baseline: DriftReport
    augmented: DriftReport
    baseline_final_train_loss: float
    augmented_final_train_loss: float
    steps: int
    seed: int
    corrupted_batches: int

    @property
    def headline_improvement(self) -> float:
        """Baseline-minus-augmented mean headline MSE (positive = augmented better)."""
        channel = self.baseline.headline_channel
        return self.baseline.mean(channel) - self.augmented.mean(channel)


def _seeded_model_pair(model_cfg: ModelConfig, seed: int) -> tuple[RSSM, RawModalityDecoders]:
    """Build an (RSSM, decoders) pair from a pinned seed."""
    torch.manual_seed(seed)
    model = RSSM(model_cfg)
    torch.manual_seed(seed)
    decoders = RawModalityDecoders(model_cfg)
    return model, decoders


def train_pair_and_compare(
    model_cfg: ModelConfig,
    drift_cfg: DriftTrainingConfig,
    train_batches: list[dict[str, Tensor]],
    held_out_batch: dict[str, Tensor],
    *,
    steps: int,
    learning_rate: float,
) -> DriftComparisonResult:
    """Train baseline + augmented arms and measure both on held-out data.

    Args:
        model_cfg: Model dims for both arms (identical seeded inits).
        drift_cfg: Corruption knobs (``corruption_prob``, ``max_prefix_frac``,
            ``recovery_weight``, ``residual_head``, eval window, ``seed``).
            ``drift_cfg.enabled`` is deliberately NOT consulted — this offline
            harness is flag-independent by design (the flag gates the
            production-pretraining integration only).
        train_batches: ``(B, T, ...)`` training batches, cycled over ``steps``.
        held_out_batch: DISJOINT held-out batch for :func:`measure_drift`
            (the caller owns disjointness — the
            ``factory._build_held_out_sequence_batch`` pattern).
        steps: Optimisation steps per arm.
        learning_rate: Adam learning rate for both arms.

    Returns:
        A :class:`DriftComparisonResult` (negative results included, never
        hidden).

    Raises:
        ValueError: If ``train_batches`` is empty or ``steps`` is not positive.
    """
    if not train_batches:
        msg = "train_batches must be non-empty"
        raise ValueError(msg)
    if steps <= 0:
        msg = f"steps must be positive; got {steps}"
        raise ValueError(msg)
    seed = drift_cfg.seed

    baseline, baseline_decoders = _seeded_model_pair(model_cfg, seed)
    augmented, augmented_decoders = _seeded_model_pair(model_cfg, seed)
    head: DriftCorrectionHead | None = None
    if drift_cfg.residual_head:
        torch.manual_seed(seed)
        head = DriftCorrectionHead(model_cfg)

    opt_baseline = torch.optim.Adam(
        list(baseline.parameters()) + list(baseline_decoders.parameters()),
        lr=learning_rate,
    )
    augmented_params = list(augmented.parameters()) + list(augmented_decoders.parameters())
    if head is not None:
        augmented_params += list(head.parameters())
    opt_augmented = torch.optim.Adam(augmented_params, lr=learning_rate)

    flip_gen = torch.Generator(device="cpu")
    flip_gen.manual_seed(seed)
    prefix_gen = torch.Generator(device="cpu")
    prefix_gen.manual_seed(seed + 1)

    baseline_loss = 0.0
    augmented_loss = 0.0
    corrupted_batches = 0
    for i in range(steps):
        batch = train_batches[i % len(train_batches)]
        step_seed = seed + 1000 + i

        torch.manual_seed(step_seed)
        opt_baseline.zero_grad()
        out_b = baseline.train_sequence(batch, baseline_decoders)
        out_b["loss"].backward()  # type: ignore[no-untyped-call]  # torch stub gap
        opt_baseline.step()
        baseline_loss = float(out_b["loss"].detach())

        corrupt = bool(torch.rand(1, generator=flip_gen).item() < drift_cfg.corruption_prob)
        torch.manual_seed(step_seed)
        opt_augmented.zero_grad()
        if corrupt:
            corrupted_batches += 1
            out_a = augmented.train_sequence_corrupted(
                batch,
                augmented_decoders,
                max_prefix_frac=drift_cfg.max_prefix_frac,
                recovery_weight=drift_cfg.recovery_weight,
                residual_head=head,
                generator=prefix_gen,
            )
            total = out_a["loss"] + out_a["residual_loss"]
        else:
            out_a = augmented.train_sequence(batch, augmented_decoders)
            total = out_a["loss"]
        total.backward()  # type: ignore[no-untyped-call]  # torch stub gap
        opt_augmented.step()
        augmented_loss = float(out_a["loss"].detach())

    baseline_report = measure_drift(
        baseline,
        held_out_batch,
        baseline_decoders,
        context_steps=drift_cfg.eval_context_steps,
        horizon=drift_cfg.eval_horizon,
        seed=seed,
    )
    augmented_report = measure_drift(
        augmented,
        held_out_batch,
        augmented_decoders,
        context_steps=drift_cfg.eval_context_steps,
        horizon=drift_cfg.eval_horizon,
        seed=seed,
        residual_head=head,
    )
    result = DriftComparisonResult(
        baseline=baseline_report,
        augmented=augmented_report,
        baseline_final_train_loss=baseline_loss,
        augmented_final_train_loss=augmented_loss,
        steps=steps,
        seed=seed,
        corrupted_batches=corrupted_batches,
    )
    _log.info(
        "drift_pair_compared",
        headline=baseline_report.headline_channel,
        baseline_mean=round(baseline_report.mean(baseline_report.headline_channel), 6),
        augmented_mean=round(augmented_report.mean(augmented_report.headline_channel), 6),
        improvement=round(result.headline_improvement, 6),
        corrupted_batches=corrupted_batches,
        steps=steps,
    )
    return result


__all__ = ["DriftComparisonResult", "train_pair_and_compare"]
