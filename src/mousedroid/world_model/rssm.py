"""Recurrent State-Space Model (RSSM) for latent world modelling."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.config.schema import ModelConfig
from mousedroid.logging.setup import get_logger
from mousedroid.sensing.protocol import ObservationProtocol
from mousedroid.world_model.encoder import MultimodalEncoder

_log = get_logger(__name__)


class RSSM(nn.Module):
    """Recurrent State-Space Model with posterior/prior latent dynamics.

    Implements the core world-model loop: *observe* embeds real observations
    and computes posterior latent states; *imagine* rolls forward using only
    the learned prior.

    Args:
        cfg: Model configuration with all dimension parameters.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self._cfg = cfg

        # Sub-modules
        self.encoder = MultimodalEncoder(cfg)
        self.gru = nn.GRUCell(cfg.latent_dim + cfg.action_dim, cfg.hidden_dim)

        # Posterior: h + obs_embed -> z parameters
        self.posterior = nn.Linear(cfg.hidden_dim + cfg.obs_dim, cfg.latent_dim * 2)

        # Prior: h -> z parameters
        self.prior = nn.Linear(cfg.hidden_dim, cfg.latent_dim * 2)

        # Reward head: h + z -> scalar
        self.reward_head = nn.Linear(cfg.hidden_dim + cfg.latent_dim, 1)

        # Observation decoder: h + z -> reconstructed obs embedding
        self.observation_decoder = nn.Linear(
            cfg.hidden_dim + cfg.latent_dim,
            cfg.obs_dim,
        )

        _log.info(
            "rssm_init",
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            action_dim=cfg.action_dim,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_gaussian(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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

    @staticmethod
    def _kl_divergence(
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

    def decode(self, h: Tensor, z: Tensor) -> Tensor:
        """Decode hidden + latent state into reconstructed observation embedding.

        Args:
            h: Hidden state, shape ``(batch, hidden_dim)``.
            z: Latent sample, shape ``(batch, latent_dim)``.

        Returns:
            Reconstructed observation embedding, shape ``(batch, obs_dim)``.
        """
        return self.observation_decoder(torch.cat([h, z], dim=-1))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe_step(
        self,
        observation: ObservationProtocol,
        prev_action: Tensor,
        h: Tensor,
        z: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, float]:
        """Process one real observation step.

        Args:
            observation: Sensor bundle implementing ``ObservationProtocol``.
            prev_action: Previous action, shape ``(1, action_dim)``.
            h: Previous hidden state, shape ``(1, hidden_dim)``.
            z: Previous latent sample, shape ``(1, latent_dim)``.

        Returns:
            ``(new_h, new_z, reconstructed_obs, surprise)``
        """
        device = h.device

        # Convert observation arrays to tensors.
        vision = torch.as_tensor(
            observation.vision_features,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        ultrasonic = torch.as_tensor(
            [observation.distance_m],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        motor = torch.as_tensor(
            observation.motor_state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        mask = torch.as_tensor(
            observation.valid_mask,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        # Encode
        obs_embed = self.encoder(vision, ultrasonic, motor, mask)

        # GRU step
        gru_input = torch.cat([z, prev_action], dim=-1)
        new_h: Tensor = self.gru(gru_input, h)

        # Posterior
        post_params = self.posterior(torch.cat([new_h, obs_embed], dim=-1))
        new_z, post_mean, post_logvar = self._sample_gaussian(post_params)

        # Prior (for KL surprise)
        prior_params = self.prior(new_h)
        _, prior_mean, prior_logvar = self._sample_gaussian(prior_params)

        surprise = self._kl_divergence(post_mean, post_logvar, prior_mean, prior_logvar)

        return new_h, new_z, obs_embed, float(surprise.item())

    @torch.no_grad()
    def imagine_step(
        self,
        action: Tensor,
        h: Tensor,
        z: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Imagine one step forward using learned prior only.

        Args:
            action: Action to imagine, shape ``(1, action_dim)``.
            h: Hidden state, shape ``(1, hidden_dim)``.
            z: Latent sample, shape ``(1, latent_dim)``.

        Returns:
            ``(new_h, new_z, predicted_reward)``
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)
        gru_input = torch.cat([z, action], dim=-1)
        new_h: Tensor = self.gru(gru_input, h)

        prior_params = self.prior(new_h)
        new_z, _, _ = self._sample_gaussian(prior_params)

        predicted_reward: Tensor = self.reward_head(torch.cat([new_h, new_z], dim=-1))
        return new_h, new_z, predicted_reward
