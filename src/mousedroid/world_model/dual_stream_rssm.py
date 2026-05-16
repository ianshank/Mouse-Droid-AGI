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
from mousedroid.world_model.latent_utils import kl_divergence, sample_gaussian
from mousedroid.world_model.observation_packer import pack_observation
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

    @torch.no_grad()
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
            ``(new_h, new_z, obs_embed, surprise)`` where ``surprise`` is
            a Python ``float`` (backwards-compatible contract). Internally
            this method delegates to :meth:`observe_step_traceable` so the
            PyTorch and ONNX runtime paths share a single implementation —
            see ``world_model.observation_packer`` for the shared
            ``ObservationProtocol`` → ``Tensor`` conversion.
        """
        device = h.device
        packed = pack_observation(observation, self._cfg, device=device)
        new_h, new_z, obs_embed, surprise_tensor = self.observe_step_traceable(
            vision=packed.vision,
            motor=packed.motor,
            valid_mask=packed.valid_mask,
            ultrasonic=packed.ultrasonic,
            audio=packed.audio,
            lidar=packed.lidar,
            prev_action=prev_action,
            h=h,
            z=z,
        )
        return new_h, new_z, obs_embed, float(surprise_tensor.item())

    @torch.no_grad()
    def observe_step_traceable(
        self,
        *,
        vision: Tensor,
        motor: Tensor,
        valid_mask: Tensor,
        prev_action: Tensor,
        h: Tensor,
        z: Tensor,
        ultrasonic: Tensor | None = None,
        audio: Tensor | None = None,
        lidar: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Tensor-only variant of :meth:`observe_step` — ONNX-traceable.

        This is the single point of truth for one dual-stream observation
        update. :meth:`observe_step` calls it (after running the packer)
        for the PyTorch path; ``scripts/export_dual_stream_rssm_onnx.py``
        traces it for the ONNX path. Returning surprise as a ``Tensor``
        (not ``float``) is what makes the method consumable by
        ``torch.onnx.export``.

        Args:
            vision: Vision features, shape ``(batch, cfg.vision_dim)``.
            motor: Motor state, shape ``(batch, cfg.motor_state_dim)``.
            valid_mask: Per-modality validity scores,
                shape ``(batch, n_modalities)``.
            ultrasonic: Optional ultrasonic reading,
                shape ``(batch, cfg.ultrasonic_dim)``; ``None`` when
                ``cfg.ultrasonic_dim == 0``.
            audio: Optional audio samples, shape ``(batch, cfg.audio_dim)``;
                ``None`` when ``cfg.audio_dim == 0``.
            lidar: Optional LiDAR features, shape ``(batch, cfg.lidar_dim)``;
                ``None`` when ``cfg.lidar_dim == 0``.
            prev_action: Previous action, shape ``(batch, cfg.action_dim)``.
            h: Previous combined hidden state, shape ``(batch, combined_dim)``.
            z: Previous latent sample, shape ``(batch, cfg.latent_dim)``.

        Returns:
            ``(new_h, new_z, obs_embed, surprise)`` — all ``Tensor``.
        """
        # Encode (the encoder already branches on enabled-modality flags).
        obs_embed = self.encoder(
            vision,
            ultrasonic,
            motor,
            valid_mask,
            audio=audio,
            lidar=lidar,
        )

        # Split combined hidden state.
        h_slow = self.fusion.extract_gru_state(h)
        h_fast = self.fusion.extract_cfc_state(h)

        # Dual-stream recurrent step.
        recurrent_input = torch.cat([z, prev_action], dim=-1)
        new_h_slow: Tensor = self.gru(recurrent_input, h_slow)
        new_h_fast: Tensor = self.cfc(recurrent_input, h_fast)

        # Fuse streams.
        new_h = self.fusion.fuse(new_h_slow, new_h_fast)

        # Posterior.
        post_params = self.posterior(torch.cat([new_h, obs_embed], dim=-1))
        new_z, post_mean, post_logvar = self._sample_gaussian(post_params)

        # Prior (for KL surprise).
        prior_params = self.prior(new_h)
        _, prior_mean, prior_logvar = self._sample_gaussian(prior_params)

        surprise = self._kl_divergence(post_mean, post_logvar, prior_mean, prior_logvar)
        return new_h, new_z, obs_embed, surprise

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
