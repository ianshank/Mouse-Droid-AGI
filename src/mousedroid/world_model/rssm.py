"""Recurrent State-Space Model (RSSM) for latent world modelling."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.config.schema import ModelConfig
from mousedroid.logging.setup import get_logger
from mousedroid.sensing.protocol import ObservationProtocol
from mousedroid.world_model.encoder import MultimodalEncoder
from mousedroid.world_model.latent_utils import (
    balanced_free_bits_kl,
    kl_divergence,
    sample_gaussian,
)

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

    @property
    def cfg(self) -> ModelConfig:
        """Return the model configuration (read-only)."""
        return self._cfg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_gaussian(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Split params into mean/logvar and sample via reparameterization."""
        return sample_gaussian(params)

    @staticmethod
    def _kl_divergence(
        post_mean: Tensor,
        post_logvar: Tensor,
        prior_mean: Tensor,
        prior_logvar: Tensor,
    ) -> Tensor:
        """Analytic KL(posterior || prior) for diagonal Gaussians."""
        return kl_divergence(post_mean, post_logvar, prior_mean, prior_logvar)

    def decode(self, h: Tensor, z: Tensor) -> Tensor:
        """Decode hidden + latent state into reconstructed observation embedding.

        Args:
            h: Hidden state, shape ``(batch, hidden_dim)``.
            z: Latent sample, shape ``(batch, latent_dim)``.

        Returns:
            Reconstructed observation embedding, shape ``(batch, obs_dim)``.
        """
        decoded = self.observation_decoder(torch.cat([h, z], dim=-1))
        return cast(Tensor, decoded)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
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
        ultrasonic: Tensor | None = None
        if self.encoder.ultrasonic_enabled:
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

        # Extract audio if the encoder supports it.
        audio: Tensor | None = None
        if self.encoder.audio_enabled:
            audio_data = observation.audio_chunk
            if len(audio_data) > 0:
                audio = torch.as_tensor(
                    audio_data,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

        # Extract LiDAR features if the encoder supports it.
        lidar: Tensor | None = None
        if self.encoder.lidar_enabled:
            lidar_data = observation.lidar_features
            if lidar_data is not None and len(lidar_data) > 0:
                lidar = torch.as_tensor(
                    lidar_data,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

        # Encode
        obs_embed = self.encoder(vision, ultrasonic, motor, mask, audio=audio, lidar=lidar)

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

    def train_sequence(
        self, batch: dict[str, Tensor], decoders: RawModalityDecoders
    ) -> dict[str, Tensor]:
        """Gradient-enabled sequence rollout for dynamics pretraining.

        Reconstructs the RAW per-modality sim observations (motor/range/lidar) —
        fixed targets, so the objective cannot collapse the way an ``obs_embed``
        self-reconstruction would. KL uses the balanced free-bits helper in
        float32. Vision is expected OFF (encoder built with ``vision_dim=0``); the
        ``batch`` therefore carries no vision tensor. This path is deliberately
        NOT decorated ``@torch.no_grad`` — the deployment inference methods above
        keep that decorator (CLAUDE.md invariant #7).

        The reconstruction heads live on the external ``decoders`` module (NOT on
        ``self``) so the deployment RSSM's ``state_dict`` + seeded init stay
        byte-identical; the pretrainer owns ``decoders`` and trains them jointly.

        Args:
            batch: Dict of ``(B, T, ...)`` tensors with keys ``motor``,
                ``valid_mask``, ``action`` (always) and ``ultrasonic`` / ``lidar``
                when those modalities are enabled.
            decoders: Raw-modality reconstruction heads (built from this model's
                config), owned + optimized by the pretrainer.

        Returns:
            Dict with scalar tensors ``loss``, ``recon``, ``kl`` and a detached
            ``posterior_std`` collapse probe.
        """
        motor = batch["motor"]
        actions = batch["action"]
        mask = batch["valid_mask"]
        b, t, _ = motor.shape
        device = motor.device
        h = torch.zeros(b, self._cfg.hidden_dim, device=device)
        z = torch.zeros(b, self._cfg.latent_dim, device=device)

        recon = torch.zeros((), device=device)
        kl_total = torch.zeros((), device=device)
        post_stds: list[Tensor] = []

        range_enabled = self.encoder.ultrasonic_enabled
        lidar_enabled = self.encoder.lidar_enabled
        for step in range(t):
            ultra = batch["ultrasonic"][:, step] if range_enabled else None
            lidar = batch["lidar"][:, step] if lidar_enabled else None
            obs_embed = self.encoder(None, ultra, motor[:, step], mask[:, step], lidar=lidar)

            gru_in = torch.cat([z, actions[:, step]], dim=-1)
            h = self.gru(gru_in, h)

            post_params = self.posterior(torch.cat([h, obs_embed], dim=-1))
            z, post_mean, post_logvar = self._sample_gaussian(post_params)
            prior_params = self.prior(h)
            _, prior_mean, prior_logvar = self._sample_gaussian(prior_params)

            with torch.autocast(device_type=device.type, enabled=False):
                kl_total = kl_total + balanced_free_bits_kl(
                    post_mean,
                    post_logvar,
                    prior_mean,
                    prior_logvar,
                    alpha=self._cfg.kl_balance_alpha,
                    free_nats=self._cfg.kl_free_nats,
                )

            hz = torch.cat([h, z], dim=-1)
            recon = recon + nn.functional.mse_loss(decoders.decode_motor(hz), motor[:, step])
            if range_enabled and ultra is not None:
                recon = recon + nn.functional.mse_loss(decoders.decode_range(hz), ultra)
            if lidar_enabled and lidar is not None:
                recon = recon + nn.functional.mse_loss(decoders.decode_lidar(hz), lidar)
            post_stds.append((0.5 * post_logvar).exp().mean().detach())

        recon = recon / t
        kl = kl_total / t
        loss = recon + self._cfg.kl_beta * kl
        return {
            "loss": loss,
            "recon": recon.detach(),
            "kl": kl.detach(),
            "posterior_std": torch.stack(post_stds).mean(),
        }


class RawModalityDecoders(nn.Module):
    """Raw per-modality reconstruction heads for RSSM dynamics pretraining.

    Deliberately kept OUT of :class:`RSSM` so the deployment model's
    ``state_dict`` and seeded weight init stay byte-identical (adding these
    ``Linear`` heads to ``RSSM.__init__`` would break checkpoint loading and
    shift the seeded golden curves). :class:`RSSMPretrainer` owns an instance,
    includes it in the optimizer, and passes it to :meth:`RSSM.train_sequence`.

    Args:
        cfg: The model config (decoder dims mirror the encoder's modalities).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        feat = cfg.hidden_dim + cfg.latent_dim
        self.decode_motor = nn.Linear(feat, cfg.motor_state_dim)
        self.range_enabled = cfg.ultrasonic_dim > 0
        if self.range_enabled:
            self.decode_range = nn.Linear(feat, 1)
        self.lidar_enabled = cfg.lidar_dim > 0
        if self.lidar_enabled:
            self.decode_lidar = nn.Linear(feat, cfg.lidar_dim)
