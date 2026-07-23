"""Deterministic open-loop drift measurement for the RSSM (F-023, ADR-015).

:func:`measure_drift` scores how far the RSSM's OPEN-LOOP prior imagination
drifts from ground truth over a fixed horizon: posterior warmup on real
observations for ``context_steps``, then ``horizon`` prior-only steps driven by
ground-truth actions, reporting per-step reconstruction MSE **per modality**
plus latent divergence versus the posterior-inferred trajectory.

Metric-honesty contracts (adversarial-review findings, pinned by
``tests/unit/training/test_drift_metrics.py``):

- **Range (ultrasonic) is the headline channel** — it is the only
  environment-coupled signal in real replay records. Decoded-motor MSE is
  reported but secondary: with ground-truth actions fed to the rollout, motor
  reconstruction largely copies the action through the GRU. This robot has NO
  pose channel (``motor_state = [vx, vy, omega, battery]``), so the change
  request's "pose error" is substituted by these channels — declared in the
  spec delta.
- **Zero-filled channels are excluded** — replay batches zero-fill lidar and
  vision, and an MSE that rewards decoding zeros would flatter the model.
  Only motor + range (when enabled) are scored against raw targets.
- **``valid_mask`` is threaded** into every per-step loss exactly as the
  encoder consumes it (per-modality slots via ``SENSOR_SLOT_MAP``).
- **Determinism** — the RNG discipline is copied from
  ``learning/on_device/scoring.py::score_dynamics``: global CPU (+ CUDA when
  available) RNG captured, ``manual_seed`` immediately before the scored
  forward, ``eval()`` + ``no_grad``, everything restored afterwards. Same
  seed + batch + weights ⇒ byte-identical report.

This module lives in ``training/`` (NOT ``validation/`` — that package is
deliberately torch-free by charter).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mousedroid.constants import SENSOR_SLOT_MAP
from mousedroid.logging.setup import get_logger
from mousedroid.world_model.protocol import LatentContextProtocol, WorldModelProtocol

if TYPE_CHECKING:
    from mousedroid.world_model.rssm import RSSM, DriftCorrectionHead, RawModalityDecoders

_log = get_logger(__name__)


@dataclass(frozen=True)
class DriftReport:
    """Per-channel open-loop drift curves for one measurement.

    Attributes:
        context_steps: Posterior warmup steps on ground truth.
        horizon: Open-loop prior rollout steps scored.
        seed: RNG seed the measurement was pinned to.
        headline_channel: ``"range"`` when the range channel was scored,
            else ``"motor"`` — the channel drift comparisons gate on.
        per_step_mse: Channel name → per-step MSE curve (length ``horizon``).
            Channels: ``motor``, ``range`` (when enabled), ``latent_h``,
            ``latent_z``, and ``motor_corrected`` (when a residual head was
            supplied).
    """

    context_steps: int
    horizon: int
    seed: int
    headline_channel: str
    per_step_mse: Mapping[str, tuple[float, ...]]

    def channels(self) -> tuple[str, ...]:
        """Return the scored channel names."""
        return tuple(self.per_step_mse)

    def mean(self, channel: str) -> float:
        """Mean MSE over the horizon for ``channel``."""
        curve = self.per_step_mse[channel]
        return sum(curve) / len(curve)

    def final(self, channel: str) -> float:
        """Final-step MSE for ``channel`` (cumulative-drift proxy)."""
        return self.per_step_mse[channel][-1]


def _masked_mse(pred: Tensor, target: Tensor, mask_col: Tensor) -> float:
    """Per-sample-masked MSE — samples with mask 0 contribute nothing."""
    per_sample = ((pred - target) ** 2).mean(dim=-1)
    denom = mask_col.sum().clamp(min=1.0)
    return float((per_sample * mask_col).sum() / denom)


def _posterior_warmup_step(
    model: RSSM,
    batch: Mapping[str, Tensor],
    step: int,
    h: Tensor,
    z: Tensor,
) -> tuple[Tensor, Tensor]:
    """One posterior step on ground-truth observations (no loss)."""
    motor = batch["motor"]
    mask = batch["valid_mask"]
    actions = batch["action"]
    ultra = batch["ultrasonic"][:, step] if model.encoder.ultrasonic_enabled else None
    lidar = batch["lidar"][:, step] if model.encoder.lidar_enabled else None
    vision = batch["vision"][:, step] if model.encoder.vision_enabled else None
    obs_embed = model.encoder(vision, ultra, motor[:, step], mask[:, step], lidar=lidar)
    h = model.gru(torch.cat([z, actions[:, step]], dim=-1), h)
    post_params = model.posterior(torch.cat([h, obs_embed], dim=-1))
    z, _, _ = model._sample_gaussian(post_params)
    return h, z


def measure_drift(
    world_model: WorldModelProtocol,
    batch: Mapping[str, Tensor],
    decoders: RawModalityDecoders,
    *,
    context_steps: int,
    horizon: int,
    seed: int,
    residual_head: DriftCorrectionHead | None = None,
    latent_context: LatentContextProtocol | None = None,
) -> DriftReport:
    """Measure open-loop drift of the RSSM prior against ground truth.

    Args:
        world_model: The concrete :class:`RSSM` to score (the only engine with
            the required training internals; anything else raises
            :class:`TypeError` — the ``scoring.py`` capability-narrow pattern).
        batch: A held-out ``(B, T, ...)`` sequence batch with
            ``T >= context_steps + horizon``.
        decoders: External raw-modality reconstruction heads (shared between
            the models being compared, like ``score_dynamics``).
        context_steps: Posterior warmup steps on ground truth.
        horizon: Open-loop prior rollout steps to score.
        seed: RNG seed (determinism contract above).
        residual_head: Optional evaluation-only
            :class:`~mousedroid.world_model.rssm.DriftCorrectionHead`. When
            supplied, an additional ``motor_corrected`` channel reports drift
            with the predicted residual applied to the decoded motor — this is
            where the trained-but-not-deployed head is consumed.
        latent_context: Optional F-023 memory, applied ONLY during the
            posterior warmup (mirrors the deployment observe seam exactly; the
            open-loop rollout is untouched). Requires ``B == 1`` — the memory
            operates on single carried states. Note: this measures the RSSM
            latent, distinct from the deployed DualStream combined latent.

    Returns:
        A :class:`DriftReport` with per-step per-channel MSE curves.

    Raises:
        TypeError: If ``world_model`` is not a concrete ``RSSM``.
        ValueError: If the batch is shorter than ``context_steps + horizon``,
            or ``latent_context`` is supplied with a batch of size > 1.
    """
    from mousedroid.world_model.rssm import RSSM

    if not isinstance(world_model, RSSM):
        msg = (
            "measure_drift requires a concrete RSSM (the only engine exposing "
            f"the training internals); got {type(world_model).__name__}"
        )
        raise TypeError(msg)
    motor = batch["motor"]
    b, t, _ = motor.shape
    if t < context_steps + horizon:
        msg = f"batch length {t} < context_steps + horizon = {context_steps + horizon}"
        raise ValueError(msg)
    if latent_context is not None and b != 1:
        msg = f"latent_context ablation requires batch size 1; got {b}"
        raise ValueError(msg)

    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = world_model.training
    decoders_was_training = decoders.training
    try:
        torch.manual_seed(seed)
        world_model.eval()
        decoders.eval()
        if residual_head is not None:
            residual_head.eval()
        with torch.no_grad():
            report = _rollout_and_score(
                world_model,
                batch,
                decoders,
                context_steps=context_steps,
                horizon=horizon,
                seed=seed,
                residual_head=residual_head,
                latent_context=latent_context,
            )
    finally:
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        if was_training:
            world_model.train()
        if decoders_was_training:
            decoders.train()

    _log.info(
        "drift_measured",
        headline=report.headline_channel,
        headline_mean=round(report.mean(report.headline_channel), 6),
        headline_final=round(report.final(report.headline_channel), 6),
        context_steps=context_steps,
        horizon=horizon,
        seed=seed,
    )
    return report


def _rollout_and_score(
    model: RSSM,
    batch: Mapping[str, Tensor],
    decoders: RawModalityDecoders,
    *,
    context_steps: int,
    horizon: int,
    seed: int,
    residual_head: DriftCorrectionHead | None,
    latent_context: LatentContextProtocol | None,
) -> DriftReport:
    """Warmup + parallel (open-loop, posterior) rollouts + per-step scoring."""
    motor = batch["motor"]
    mask = batch["valid_mask"]
    actions = batch["action"]
    b = motor.shape[0]
    device = motor.device
    cfg = model.cfg
    range_enabled = model.encoder.ultrasonic_enabled and decoders.range_enabled

    h = torch.zeros(b, cfg.hidden_dim, device=device)
    z = torch.zeros(b, cfg.latent_dim, device=device)
    for step in range(context_steps):
        h, z = _posterior_warmup_step(model, batch, step, h, z)
        if latent_context is not None:
            latent_context.observe(h, z)
            h, z = latent_context.contextualize(h, z)

    h_roll, z_roll = h, z
    h_post, z_post = h, z
    motor_slot = SENSOR_SLOT_MAP["motor"]
    ultra_slot = SENSOR_SLOT_MAP["ultrasonic"]
    curves: dict[str, list[float]] = {"motor": [], "latent_h": [], "latent_z": []}
    if range_enabled:
        curves["range"] = []
    if residual_head is not None:
        curves["motor_corrected"] = []

    for j in range(horizon):
        step = context_steps + j
        action = actions[:, step]
        # Open-loop leg: prior imagination only.
        h_roll = model.gru(torch.cat([z_roll, action], dim=-1), h_roll)
        z_roll, _, _ = model._sample_gaussian(model.prior(h_roll))
        hz_roll = torch.cat([h_roll, z_roll], dim=-1)
        # Posterior twin: the observation-anchored reference trajectory.
        h_post, z_post = _posterior_warmup_step(model, batch, step, h_post, z_post)

        motor_mask = mask[:, step, motor_slot]
        decoded_motor = decoders.decode_motor(hz_roll)
        curves["motor"].append(_masked_mse(decoded_motor, motor[:, step], motor_mask))
        if range_enabled:
            ultra_mask = mask[:, step, ultra_slot]
            decoded_range = decoders.decode_range(hz_roll)
            curves["range"].append(
                _masked_mse(decoded_range, batch["ultrasonic"][:, step], ultra_mask)
            )
        if residual_head is not None:
            corrected = decoded_motor + residual_head(hz_roll)
            curves["motor_corrected"].append(_masked_mse(corrected, motor[:, step], motor_mask))
        curves["latent_h"].append(float(((h_roll - h_post) ** 2).mean()))
        curves["latent_z"].append(float(((z_roll - z_post) ** 2).mean()))

    headline = "range" if range_enabled else "motor"
    return DriftReport(
        context_steps=context_steps,
        horizon=horizon,
        seed=seed,
        headline_channel=headline,
        per_step_mse={name: tuple(curve) for name, curve in curves.items()},
    )


__all__ = ["DriftReport", "measure_drift"]
