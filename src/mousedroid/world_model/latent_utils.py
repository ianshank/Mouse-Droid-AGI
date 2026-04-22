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
