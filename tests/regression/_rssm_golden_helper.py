"""Deterministic helper for the Phase 2 golden RSSM regression test.

This helper mirrors the per-step training loop in
``training/train_rssm.py::train_rssm`` but strips the file I/O, the
``DataLoader``, and the AMP path so the curve is bit-stable on CPU.

Design constraints:

* CPU-only — no CUDA dependence in CI.
* Tiny dims — the test must finish in well under a second.
* Single optimizer (Adam) with a fixed learning rate; no scheduler.
* Synthetic batches generated from a seeded ``torch.Generator`` so the
  loss curve is reproducible.
* No reliance on the YAML configs or on the LMDB experience logger.

The loss formula is the same as ``train_rssm``:

    loss_t = MSE(decoded_obs, encoded_obs) + kl_beta * KL(post || prior)

averaged over the time dimension within each step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from mousedroid.config.schema import ModelConfig
from mousedroid.constants import SENSOR_SLOT_MAP
from mousedroid.world_model.rssm import RSSM

# Tiny, fixed dims chosen so the test runs in <1 s on CPU and the loss
# curve is numerically well-conditioned. These are *not* production
# dimensions and intentionally bypass the YAML schema defaults.
GOLDEN_VISION_DIM: int = 8
GOLDEN_MOTOR_STATE_DIM: int = 4
GOLDEN_HIDDEN_DIM: int = 16
GOLDEN_LATENT_DIM: int = 8
GOLDEN_ACTION_DIM: int = 3
GOLDEN_OBS_DIM: int = 16
GOLDEN_VISION_PROJ_DIM: int = 8
GOLDEN_MOTOR_PROJ_DIM: int = 8
GOLDEN_BATCH: int = 4
GOLDEN_SEQ_LEN: int = 4
GOLDEN_LR: float = 1e-3
GOLDEN_KL_BETA: float = 1.0
# Ultrasonic is the cheapest required distance modality (model schema
# requires either ultrasonic or lidar to be enabled).
GOLDEN_ULTRASONIC_DIM: int = 1
GOLDEN_ULTRASONIC_PROJ_DIM: int = 4


@dataclass(frozen=True)
class GoldenRSSMConfig:
    """Knobs the golden harness exposes — all have backwards-compatible defaults."""

    num_steps: int = 10
    seed: int = 0
    learning_rate: float = GOLDEN_LR
    kl_beta: float = GOLDEN_KL_BETA
    batch_size: int = GOLDEN_BATCH
    seq_len: int = GOLDEN_SEQ_LEN


def build_golden_model_config() -> ModelConfig:
    """Return the tiny ``ModelConfig`` used by the golden harness.

    Built via :meth:`ModelConfig.model_validate` so unspecified fields fall
    back to their schema defaults — keeps the harness backwards-compatible
    when new ``ModelConfig`` fields are added in future PRs.
    """
    return ModelConfig.model_validate(
        {
            "vision_dim": GOLDEN_VISION_DIM,
            "ultrasonic_dim": GOLDEN_ULTRASONIC_DIM,
            "motor_state_dim": GOLDEN_MOTOR_STATE_DIM,
            "hidden_dim": GOLDEN_HIDDEN_DIM,
            "latent_dim": GOLDEN_LATENT_DIM,
            "action_dim": GOLDEN_ACTION_DIM,
            "obs_dim": GOLDEN_OBS_DIM,
            "vision_proj_dim": GOLDEN_VISION_PROJ_DIM,
            "ultrasonic_proj_dim": GOLDEN_ULTRASONIC_PROJ_DIM,
            "motor_proj_dim": GOLDEN_MOTOR_PROJ_DIM,
        }
    )


def _make_synthetic_batch(
    mcfg: ModelConfig,
    cfg: GoldenRSSMConfig,
    generator: torch.Generator,
) -> dict[str, Tensor]:
    """Sample one deterministic batch from ``generator``."""
    n_slots = len(SENSOR_SLOT_MAP)
    valid = torch.zeros(cfg.batch_size, cfg.seq_len, n_slots, dtype=torch.float32)
    valid[..., SENSOR_SLOT_MAP["vision"]] = 1.0
    valid[..., SENSOR_SLOT_MAP["motor"]] = 1.0
    valid[..., SENSOR_SLOT_MAP["ultrasonic"]] = 1.0
    return {
        "vision": torch.randn(cfg.batch_size, cfg.seq_len, mcfg.vision_dim, generator=generator),
        "ultrasonic": torch.randn(
            cfg.batch_size, cfg.seq_len, mcfg.ultrasonic_dim, generator=generator
        ),
        "motor_state": torch.randn(
            cfg.batch_size, cfg.seq_len, mcfg.motor_state_dim, generator=generator
        ),
        "valid_mask": valid,
        "actions": torch.randn(cfg.batch_size, cfg.seq_len, mcfg.action_dim, generator=generator),
    }


def _train_step(
    rssm: RSSM,
    optimizer: torch.optim.Optimizer,
    mcfg: ModelConfig,
    batch: dict[str, Tensor],
    cfg: GoldenRSSMConfig,
) -> tuple[float, float, float]:
    """Run one optimizer step and return ``(recon, kl, total)``."""
    mse = nn.MSELoss()
    batch_size = batch["vision"].shape[0]
    seq_len = batch["vision"].shape[1]

    h = torch.zeros(batch_size, mcfg.hidden_dim)
    z = torch.zeros(batch_size, mcfg.latent_dim)

    optimizer.zero_grad()
    total_recon = torch.tensor(0.0)
    total_kl = torch.tensor(0.0)

    for t in range(seq_len):
        obs_embed = rssm.encoder(
            batch["vision"][:, t],
            batch["ultrasonic"][:, t],
            batch["motor_state"][:, t],
            batch["valid_mask"][:, t],
        )
        prev_action = batch["actions"][:, max(0, t - 1)]
        gru_input = torch.cat([z, prev_action], dim=-1)
        h = rssm.gru(gru_input, h)

        post_params = rssm.posterior(torch.cat([h, obs_embed], dim=-1))
        z, post_mean, post_logvar = rssm._sample_gaussian(post_params)

        prior_params = rssm.prior(h)
        _, prior_mean, prior_logvar = rssm._sample_gaussian(prior_params)

        obs_recon = rssm.decode(h, z)
        total_recon = total_recon + mse(obs_recon, obs_embed)
        total_kl = total_kl + rssm._kl_divergence(post_mean, post_logvar, prior_mean, prior_logvar)

    total_recon = total_recon / seq_len
    total_kl = total_kl / seq_len
    loss = total_recon + cfg.kl_beta * total_kl
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    return float(total_recon.item()), float(total_kl.item()), float(loss.item())


def compute_rssm_loss_curve(
    cfg: GoldenRSSMConfig | None = None,
) -> list[dict[str, float]]:
    """Run the deterministic golden RSSM training loop.

    Returns:
        A list of length ``cfg.num_steps``; each entry is a dict with keys
        ``recon``, ``kl``, ``total``, all ``float``.

    The loop:

    1. Pins ``torch.manual_seed(cfg.seed)`` and builds a CPU
       ``torch.Generator`` seeded with the same value, so every parameter
       init *and* every synthetic batch is reproducible.
    2. Builds an :class:`RSSM` from :func:`build_golden_model_config`.
    3. Runs ``cfg.num_steps`` optimizer steps; each step uses a fresh
       synthetic batch sampled from the seeded generator.
    """
    cfg = cfg or GoldenRSSMConfig()

    torch.manual_seed(cfg.seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(cfg.seed)

    mcfg = build_golden_model_config()
    rssm = RSSM(mcfg)
    optimizer = torch.optim.Adam(rssm.parameters(), lr=cfg.learning_rate)

    curve: list[dict[str, float]] = []
    for _ in range(cfg.num_steps):
        batch = _make_synthetic_batch(mcfg, cfg, generator)
        recon, kl, total = _train_step(rssm, optimizer, mcfg, batch, cfg)
        curve.append({"recon": recon, "kl": kl, "total": total})

    return curve
