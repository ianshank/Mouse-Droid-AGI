"""Dual-Stream Hybrid CfC/GRU Recurrent State-Space Model.

Augments the standard GRU-based RSSM with a CfC (Closed-form
Continuous-depth) liquid neural network stream, providing adaptive
time constants for fast-dynamics sensing and interpretable ODE-based
safety traces.

The combined hidden state ``h = concat(h_gru, h_cfc)`` feeds into
the posterior, prior, reward, and decoder heads.  When
``cfc_hidden_dim > 0`` in the config, this module should be used
instead of the classic :class:`~mousedroid.world_model.rssm.RSSM`.

Conforms to :class:`~mousedroid.world_model.protocol.WorldModelProtocol`
and :class:`~mousedroid.world_model.protocol.SafetyTraceProtocol`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.config.schema import ModelConfig
from mousedroid.logging.setup import get_logger
from mousedroid.sensing.protocol import ObservationProtocol
from mousedroid.world_model.cfc_cell import CfCWrapper
from mousedroid.world_model.encoder import MultimodalEncoder
from mousedroid.world_model.stream_fusion import StreamFusion

_log = get_logger(__name__)


class DualStreamRSSM(nn.Module):
    """Dual-stream RSSM with GRU (slow dynamics) + CfC (fast dynamics).

    Architecture::

        Input (z_prev, action) --+--> GRUCell  --> h_slow --+
                                 |                          +-- concat --> h_combined
                                 +--> CfCCell  --> h_fast --+
                                                            |
                                       posterior / prior / reward / decoder heads

    Args:
        cfg: Model configuration with all dimension parameters.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.cfc_hidden_dim <= 0:
            raise ValueError(
                f"DualStreamRSSM requires cfc_hidden_dim > 0, got {cfg.cfc_hidden_dim}. "
                "Use RSSM instead for pure-GRU mode."
            )
        self._cfg = cfg
        recurrent_input_dim = cfg.latent_dim + cfg.action_dim
        combined_dim = cfg.hidden_dim + cfg.cfc_hidden_dim

        # Shared encoder
        self.encoder = MultimodalEncoder(cfg)

        # Dual recurrent streams
        self.gru = nn.GRUCell(recurrent_input_dim, cfg.hidden_dim)
        self.cfc = CfCWrapper(recurrent_input_dim, cfg)

        # Stream fusion
        self.fusion = StreamFusion(cfg.hidden_dim, cfg.cfc_hidden_dim)

        # Posterior: h_combined + obs_embed -> z parameters
        self.posterior = nn.Linear(combined_dim + cfg.obs_dim, cfg.latent_dim * 2)

        # Prior: h_combined -> z parameters
        self.prior = nn.Linear(combined_dim, cfg.latent_dim * 2)

        # Reward head: h_combined + z -> scalar
        self.reward_head = nn.Linear(combined_dim + cfg.latent_dim, 1)

        # Observation decoder: h_combined + z -> reconstructed obs embedding
        self.observation_decoder = nn.Linear(
            combined_dim + cfg.latent_dim,
            cfg.obs_dim,
        )

        _log.info(
            "dual_stream_rssm_init",
            gru_hidden_dim=cfg.hidden_dim,
            cfc_hidden_dim=cfg.cfc_hidden_dim,
            combined_dim=combined_dim,
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
        """Decode combined hidden + latent state into reconstructed obs embedding.

        Args:
            h: Combined hidden state, shape ``(batch, combined_dim)``.
            z: Latent sample, shape ``(batch, latent_dim)``.

        Returns:
            Reconstructed observation embedding, shape ``(batch, obs_dim)``.
        """
        decoded = self.observation_decoder(torch.cat([h, z], dim=-1))
        return cast(Tensor, decoded)

    # ------------------------------------------------------------------
    # Stream parameter iterators (for dual-optimizer training)
    # ------------------------------------------------------------------

    def gru_parameters(self) -> Iterator[nn.Parameter]:
        """Yield parameters belonging to the GRU stream.

        Includes the GRU cell and all shared heads (posterior, prior,
        reward, decoder, encoder).  The GRU optimizer is the primary
        optimizer and owns the shared parameters.

        Yields:
            GRU-stream and shared-head parameters.
        """
        yield from self.encoder.parameters()
        yield from self.gru.parameters()
        yield from self.posterior.parameters()
        yield from self.prior.parameters()
        yield from self.reward_head.parameters()
        yield from self.observation_decoder.parameters()

    def cfc_parameters(self) -> Iterator[nn.Parameter]:
        """Yield parameters belonging to the CfC stream only.

        The CfC optimizer only updates CfC-specific parameters to
        avoid doubling the effective learning rate on shared heads.

        Yields:
            CfC cell parameters.
        """
        yield from self.cfc.parameters()

    # ------------------------------------------------------------------
    # Safety trace extraction
    # ------------------------------------------------------------------

    def get_safety_trace(self, h: Tensor) -> Tensor:
        """Extract CfC hidden state for safety monitor inspection.

        Satisfies :class:`~mousedroid.world_model.protocol.SafetyTraceProtocol`.

        Args:
            h: Combined hidden state, shape ``(batch, combined_dim)``.

        Returns:
            CfC portion of hidden state, shape ``(batch, cfc_hidden_dim)``.
        """
        return self.fusion.extract_cfc_state(h)

    # ------------------------------------------------------------------
    # Public API (WorldModelProtocol)
    # ------------------------------------------------------------------

    def observe_step(
        self,
        observation: ObservationProtocol,
        prev_action: Tensor,
        h: Tensor,
        z: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, float]:
        """Process one real observation step with dual-stream dynamics.

        Args:
            observation: Sensor bundle implementing ``ObservationProtocol``.
            prev_action: Previous action, shape ``(1, action_dim)``.
            h: Previous combined hidden state, shape ``(1, combined_dim)``.
            z: Previous latent sample, shape ``(1, latent_dim)``.

        Returns:
            ``(new_h, new_z, obs_embed, surprise)``
        """
        device = h.device

        # --- Convert observation arrays to tensors ---
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

        # Extract audio if the encoder supports it.
        audio: Tensor | None = None
        if self.encoder._audio_enabled:
            audio_data = observation.audio_chunk
            if len(audio_data) > 0:
                audio = torch.as_tensor(
                    audio_data,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

        # Extract LiDAR features if the encoder supports it.
        lidar: Tensor | None = None
        if self.encoder._lidar_enabled:
            lidar_data = observation.lidar_features
            if lidar_data is not None and len(lidar_data) > 0:
                lidar = torch.as_tensor(
                    lidar_data,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

        # Encode
        obs_embed = self.encoder(vision, ultrasonic, motor, mask, audio=audio, lidar=lidar)

        # --- Split combined hidden state ---
        h_slow = self.fusion.extract_gru_state(h)
        h_fast = self.fusion.extract_cfc_state(h)

        # --- Dual-stream recurrent step ---
        recurrent_input = torch.cat([z, prev_action], dim=-1)
        new_h_slow: Tensor = self.gru(recurrent_input, h_slow)
        new_h_fast: Tensor = self.cfc(recurrent_input, h_fast)

        # --- Fuse streams ---
        new_h = self.fusion.fuse(new_h_slow, new_h_fast)

        # --- Posterior ---
        post_params = self.posterior(torch.cat([new_h, obs_embed], dim=-1))
        new_z, post_mean, post_logvar = self._sample_gaussian(post_params)

        # --- Prior (for KL surprise) ---
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
            h: Combined hidden state, shape ``(1, combined_dim)``.
            z: Latent sample, shape ``(1, latent_dim)``.

        Returns:
            ``(new_h, new_z, predicted_reward)``
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)

        # Split combined hidden state
        h_slow = self.fusion.extract_gru_state(h)
        h_fast = self.fusion.extract_cfc_state(h)

        # Dual-stream step
        recurrent_input = torch.cat([z, action], dim=-1)
        new_h_slow: Tensor = self.gru(recurrent_input, h_slow)
        new_h_fast: Tensor = self.cfc(recurrent_input, h_fast)

        # Fuse
        new_h = self.fusion.fuse(new_h_slow, new_h_fast)

        # Prior
        prior_params = self.prior(new_h)
        new_z, _, _ = self._sample_gaussian(prior_params)

        # Reward prediction
        predicted_reward: Tensor = self.reward_head(torch.cat([new_h, new_z], dim=-1))
        return new_h, new_z, predicted_reward
