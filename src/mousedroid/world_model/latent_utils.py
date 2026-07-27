"""Shared latent-space utilities for RSSM variants.

Consolidates the Gaussian sampling and KL divergence logic used by
both :class:`~mousedroid.world_model.rssm.RSSM` and
:class:`~mousedroid.world_model.dual_stream_rssm.DualStreamRSSM`.
"""

from __future__ import annotations

import torch
from torch import Tensor


def sample_gaussian(params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Split params into mean/logvar and sample via reparameterization.

    Args:
        params: Concatenated ``[mean, logvar]``, shape ``(batch, latent*2)``.

    Returns:
        Tuple of ``(sample, mean, logvar)``.
    """
    mean, logvar = params.chunk(2, dim=-1)
    std = torch.exp(logvar * 0.5)
    eps = torch.randn_like(std)
    sample = mean + std * eps
    return sample, mean, logvar


def kl_divergence(
    post_mean: Tensor,
    post_logvar: Tensor,
    prior_mean: Tensor,
    prior_logvar: Tensor,
) -> Tensor:
    """Analytic KL(posterior || prior) for diagonal Gaussians.

    Args:
        post_mean: Posterior mean.
        post_logvar: Posterior log-variance.
        prior_mean: Prior mean.
        prior_logvar: Prior log-variance.

    Returns:
        Scalar KL divergence averaged over the batch.
    """
    kl = 0.5 * (
        prior_logvar
        - post_logvar
        + (post_logvar.exp() + (post_mean - prior_mean).pow(2)) / prior_logvar.exp()
        - 1.0
    )
    return kl.sum(dim=-1).mean()


_DEFAULT_LOGVAR_CLAMP = 10.0


def balanced_free_bits_kl(
    post_mean: Tensor,
    post_logvar: Tensor,
    prior_mean: Tensor,
    prior_logvar: Tensor,
    *,
    alpha: float,
    free_nats: float,
    logvar_clamp: float = _DEFAULT_LOGVAR_CLAMP,
) -> Tensor:
    """KL-balanced, free-bits, fp32-stable KL(posterior || prior).

    Implements Dreamer-v2/v3-style KL balancing — ``alpha`` weights the
    prior-update term (posterior detached) against the posterior-update term
    (prior detached) — followed by a free-bits floor at ``free_nats`` nats.
    Computed in float32 with logvars clamped to ``[-logvar_clamp, logvar_clamp]``
    so an fp16 AMP context cannot overflow ``exp(logvar)`` into NaN.

    Args:
        post_mean: Posterior mean, shape ``(batch, latent_dim)``.
        post_logvar: Posterior log-variance, same shape.
        prior_mean: Prior mean, same shape.
        prior_logvar: Prior log-variance, same shape.
        alpha: Balancing weight in ``[0, 1]`` (Dreamer default ~0.8).
        free_nats: Per-batch free-bits floor (nats). ``0`` disables the floor.
        logvar_clamp: Symmetric ``|logvar|`` clamp before ``exp`` (config-driven
            via ``ModelConfig.logvar_clamp``; default preserves prior behaviour).

    Returns:
        Scalar mean KL (after balancing + free-bits), as a float32 tensor.
    """

    def _kl(pm: Tensor, plv: Tensor, qm: Tensor, qlv: Tensor) -> Tensor:
        pm, plv = pm.float(), plv.float().clamp(-logvar_clamp, logvar_clamp)
        qm, qlv = qm.float(), qlv.float().clamp(-logvar_clamp, logvar_clamp)
        return 0.5 * (
            qlv - plv + (plv.exp() + (pm - qm) ** 2) / qlv.exp() - 1.0
        )  # hardcoded-ok  # type: ignore[no-any-return,unused-ignore]

    kl_lhs = _kl(post_mean.detach(), post_logvar.detach(), prior_mean, prior_logvar)
    kl_rhs = _kl(post_mean, post_logvar, prior_mean.detach(), prior_logvar.detach())
    kl = alpha * kl_lhs + (1.0 - alpha) * kl_rhs
    # Free-bits floor is PER-SAMPLE (sum over latent dims), applied BEFORE the
    # batch mean. Clamping the batch average instead would let a single
    # high-KL sample lift the mean above ``free_nats`` and mask collapse
    # (KL -> 0) in the other samples.
    kl = kl.sum(dim=-1)
    if free_nats > 0.0:
        kl = torch.clamp(kl, min=free_nats)
    return kl.mean()
