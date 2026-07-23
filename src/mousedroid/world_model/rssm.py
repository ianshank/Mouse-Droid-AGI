"""Recurrent State-Space Model (RSSM) for latent world modelling."""

from __future__ import annotations

from collections.abc import Mapping
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

        # Convert observation arrays to tensors. Vision is gated on the encoder so
        # a vision-OFF RSSM does not require camera features on the observation
        # (mirrors MultimodalEncoder.forward accepting vision=None).
        vision: Tensor | None = None
        if self.encoder.vision_enabled:
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

    @torch.no_grad()
    def posterior_step(
        self, batch: Mapping[str, Tensor], step: int, h: Tensor, z: Tensor
    ) -> tuple[Tensor, Tensor]:
        """One posterior update on ground-truth observations — no loss, no grad.

        The observation-anchored counterpart of :meth:`imagine_step`, exposed so
        evaluation harnesses (e.g. drift measurement) do NOT re-implement the
        encoder modality-slicing + GRU/posterior sequence against private
        internals. Encoder modality gating mirrors :meth:`train_sequence`'s
        per-step core.

        Args:
            batch: ``(B, T, ...)`` batch with ``motor`` / ``valid_mask`` /
                ``action`` (always) and ``ultrasonic`` / ``lidar`` / ``vision``
                when enabled.
            step: Time index into the batch.
            h: Hidden state entering this step.
            z: Latent sample entering this step.

        Returns:
            ``(new_h, new_z)`` after the posterior update.
        """
        motor = batch["motor"]
        ultra = batch["ultrasonic"][:, step] if self.encoder.ultrasonic_enabled else None
        lidar = batch["lidar"][:, step] if self.encoder.lidar_enabled else None
        vision = batch["vision"][:, step] if self.encoder.vision_enabled else None
        obs_embed = self.encoder(
            vision, ultra, motor[:, step], batch["valid_mask"][:, step], lidar=lidar
        )
        new_h: Tensor = self.gru(torch.cat([z, batch["action"][:, step]], dim=-1), h)
        new_z, _, _ = self._sample_gaussian(self.posterior(torch.cat([new_h, obs_embed], dim=-1)))
        return new_h, new_z

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
        b, t, _ = motor.shape
        device = motor.device
        h = torch.zeros(b, self._cfg.hidden_dim, device=device)
        z = torch.zeros(b, self._cfg.latent_dim, device=device)

        recon = torch.zeros((), device=device)
        kl_total = torch.zeros((), device=device)
        post_stds: list[Tensor] = []

        for step in range(t):
            h, z, recon, kl_total, post_std, _hz = self._posterior_recon_step(
                batch, decoders, step, h, z, recon, kl_total
            )
            post_stds.append(post_std)

        recon = recon / t
        kl = kl_total / t
        loss = recon + self._cfg.kl_beta * kl
        return {
            "loss": loss,
            "recon": recon.detach(),
            "kl": kl.detach(),
            "posterior_std": torch.stack(post_stds).mean(),
        }

    def _posterior_recon_step(
        self,
        batch: dict[str, Tensor],
        decoders: RawModalityDecoders,
        step: int,
        h: Tensor,
        z: Tensor,
        recon: Tensor,
        kl_total: Tensor,
        *,
        recon_weight: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """One grad-enabled posterior training step (shared per-step core).

        Extracted from :meth:`train_sequence` and reused by
        :meth:`train_sequence_corrupted` so the two paths can never drift
        apart numerically. The RUNNING accumulators (``recon``, ``kl_total``)
        are threaded through (not summed locally) so the floating-point
        addition order — and therefore the loss value — is byte-identical to
        the pre-refactor inline loop. RNG draw order per step is posterior
        sample then prior sample (both from the global RNG).

        Args:
            batch: The ``(B, T, ...)`` training batch (see ``train_sequence``).
            decoders: External raw-modality reconstruction heads.
            step: Time index into the batch.
            h: Hidden state entering this step.
            z: Latent sample entering this step.
            recon: Running reconstruction-loss accumulator.
            kl_total: Running KL accumulator.
            recon_weight: Multiplier on this step's reconstruction terms.
                ``1.0`` (default) adds the terms unscaled — provably inert, so
                ``train_sequence`` is byte-identical to its pre-refactor form.

        Returns:
            ``(h, z, recon, kl_total, post_std_step, hz)``.
        """
        motor = batch["motor"]
        actions = batch["action"]
        mask = batch["valid_mask"]
        device = motor.device
        range_enabled = self.encoder.ultrasonic_enabled
        lidar_enabled = self.encoder.lidar_enabled
        vision_enabled = self.encoder.vision_enabled

        ultra = batch["ultrasonic"][:, step] if range_enabled else None
        lidar = batch["lidar"][:, step] if lidar_enabled else None
        vision = batch["vision"][:, step] if vision_enabled else None
        obs_embed = self.encoder(vision, ultra, motor[:, step], mask[:, step], lidar=lidar)

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
                logvar_clamp=self._cfg.logvar_clamp,
            )

        def _add(current: Tensor, term: Tensor) -> Tensor:
            # ``weight == 1.0`` adds the raw term (no multiply op) so the
            # default path reproduces the pre-refactor graph exactly.
            if recon_weight == 1.0:
                return current + term
            return current + recon_weight * term

        hz = torch.cat([h, z], dim=-1)
        recon = _add(recon, nn.functional.mse_loss(decoders.decode_motor(hz), motor[:, step]))
        if range_enabled and ultra is not None:
            recon = _add(recon, nn.functional.mse_loss(decoders.decode_range(hz), ultra))
        if lidar_enabled and lidar is not None:
            recon = _add(recon, nn.functional.mse_loss(decoders.decode_lidar(hz), lidar))
        if decoders.vision_enabled and vision is not None:
            # The vision target is the L2-normalised MeanPool feature vector;
            # MSE here is an auxiliary alignment signal (not the primary
            # objective). Collapse is guarded globally by the posterior_std
            # probe + the other raw-modality recon terms, so a plain MSE is
            # sufficient — a cosine term is an available future refinement.
            recon = _add(recon, nn.functional.mse_loss(decoders.decode_vision(hz), vision))
        # fp32 + clamp (like the KL) so an AMP fp16 logvar can't overflow the
        # collapse probe into inf/nan and make the logged std unreliable.
        _clamped_lv = post_logvar.float().clamp(-self._cfg.logvar_clamp, self._cfg.logvar_clamp)
        post_std = (0.5 * _clamped_lv).exp().mean().detach()
        return h, z, recon, kl_total, post_std, hz

    def _open_loop_prefix(
        self, batch: dict[str, Tensor], k: int, h: Tensor, z: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Roll ``k`` open-loop prior steps under ``no_grad`` (self-corruption).

        The model's OWN prior imagination — no posterior correction — is the
        corrupted history (F-023, adapted from AlayaWorld's corrupted-history
        training). States are detached; no loss is computed on the prefix.

        Args:
            batch: The training batch (only ``action`` is consumed).
            k: Prefix length (``0`` skips the loop entirely — no RNG draws).
            h: Zero-initialised hidden state.
            z: Zero-initialised latent sample.

        Returns:
            ``(h, z)`` after ``k`` prior steps, detached.
        """
        actions = batch["action"]
        with torch.no_grad():
            for step in range(k):
                gru_in = torch.cat([z, actions[:, step]], dim=-1)
                h = self.gru(gru_in, h)
                prior_params = self.prior(h)
                z, _, _ = self._sample_gaussian(prior_params)
        return h.detach(), z.detach()

    def train_sequence_corrupted(
        self,
        batch: dict[str, Tensor],
        decoders: RawModalityDecoders,
        *,
        max_prefix_frac: float,
        recovery_weight: float = 1.0,
        residual_head: DriftCorrectionHead | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        """Corrupted-history recovery training (F-023, ADR-015).

        A random-length prefix ``k ~ U[0, floor(max_prefix_frac * T)]``
        (capped at ``T - 1`` so at least one step trains) is rolled OPEN-LOOP
        under the model's own prior (``no_grad``, detached — the
        self-generated corrupted history); the suffix runs the standard
        posterior recon+KL path via the shared per-step helper, training the
        model to recover toward ground truth.

        k=0 contract: with ``residual_head=None``, a forced ``k = 0`` is
        allclose-identical to :meth:`train_sequence` — the prefix draw comes
        from a PRIVATE generator (``generator=None`` constructs a fresh
        ``torch.Generator``; the GLOBAL RNG is never consumed for it, so the
        suffix's per-step draw order matches exactly), ``recovery_weight`` is
        applied only when ``k > 0``, and normalisation is by ``T - k``
        (``= T`` at ``k = 0``).

        Args:
            batch: Dict of ``(B, T, ...)`` tensors (see ``train_sequence``).
            decoders: External raw-modality reconstruction heads.
            max_prefix_frac: Upper bound on the prefix as a fraction of ``T``.
            recovery_weight: Multiplier on the recovery suffix's recon terms
                when ``k > 0`` (``1.0`` = uniform weighting).
            residual_head: Optional evaluation-only
                :class:`DriftCorrectionHead`. When supplied, its residual loss
                is returned under the SEPARATE ``residual_loss`` key (never
                folded into ``loss``); the head's input and target are both
                detached, so no gradient reaches the RSSM or the decoders.
            generator: RNG for the prefix-length draw. ``None`` constructs a
                private ``torch.Generator`` — the draw must NEVER consume the
                global RNG or the k=0 equality contract breaks.

        Returns:
            Dict with ``loss``, ``recon``, ``kl``, ``posterior_std`` (as
            ``train_sequence``) plus ``prefix_len`` (0-dim int tensor) and
            ``residual_loss`` (grad-attached when a head is supplied; zero
            otherwise).

        Raises:
            ValueError: If ``max_prefix_frac`` is outside ``(0, 1]``.
        """
        if not 0.0 < max_prefix_frac <= 1.0:
            msg = f"max_prefix_frac must be in (0, 1]; got {max_prefix_frac}"
            raise ValueError(msg)
        motor = batch["motor"]
        b, t, _ = motor.shape
        device = motor.device
        if generator is not None:
            gen = generator
        else:
            # Seed the private generator from the CURRENT global seed rather
            # than leaving it unseeded — this keeps the prefix-length draw
            # reproducible across runs (tied to the caller's ``manual_seed``)
            # WITHOUT consuming the global RNG stream (which would break the
            # k=0 equality contract with ``train_sequence``).
            gen = torch.Generator(device="cpu")
            gen.manual_seed(torch.initial_seed())
        max_k = min(int(max_prefix_frac * t), t - 1)
        k = int(torch.randint(0, max_k + 1, (1,), generator=gen).item())

        h = torch.zeros(b, self._cfg.hidden_dim, device=device)
        z = torch.zeros(b, self._cfg.latent_dim, device=device)
        h, z = self._open_loop_prefix(batch, k, h, z)

        recon = torch.zeros((), device=device)
        kl_total = torch.zeros((), device=device)
        residual_loss = torch.zeros((), device=device)
        post_stds: list[Tensor] = []
        weight = recovery_weight if k > 0 else 1.0

        for step in range(k, t):
            h, z, recon, kl_total, post_std, hz = self._posterior_recon_step(
                batch, decoders, step, h, z, recon, kl_total, recon_weight=weight
            )
            post_stds.append(post_std)
            if residual_head is not None:
                hz_detached = hz.detach()
                residual_target = motor[:, step] - decoders.decode_motor(hz_detached).detach()
                residual_loss = residual_loss + nn.functional.mse_loss(
                    residual_head(hz_detached), residual_target
                )

        n_train = t - k
        recon = recon / n_train
        kl = kl_total / n_train
        loss = recon + self._cfg.kl_beta * kl
        return {
            "loss": loss,
            "recon": recon.detach(),
            "kl": kl.detach(),
            "posterior_std": torch.stack(post_stds).mean(),
            "prefix_len": torch.tensor(k, device=device),
            "residual_loss": residual_loss / n_train,
        }


class DriftCorrectionHead(nn.Module):
    """Evaluation-only residual-correction head (F-023, ADR-015).

    Predicts the correction residual ``delta_motor = motor_gt - decode_motor(hz)``
    from a DETACHED ``hz`` — by construction, zero gradient reaches the RSSM
    or the decoders; the head trains alone. It is **trained-but-not-deployed**:
    :func:`~mousedroid.training.drift_metrics.measure_drift` consumes it to
    report drift with and without the residual correction, but no rover
    inference path ever reads it, and (like :class:`RawModalityDecoders`) it
    adds NO parameters to the RSSM ``state_dict``.

    Args:
        cfg: The model config (dims mirror the RSSM latent + motor state).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.head = nn.Linear(cfg.hidden_dim + cfg.latent_dim, cfg.motor_state_dim)

    def forward(self, hz: Tensor) -> Tensor:
        """Predict the motor-channel correction residual.

        Args:
            hz: Detached ``cat(h, z)``, shape ``(batch, hidden_dim + latent_dim)``.

        Returns:
            Predicted ``delta_motor``, shape ``(batch, motor_state_dim)``.
        """
        return cast(Tensor, self.head(hz))


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
        self.vision_enabled = cfg.vision_dim > 0
        if self.vision_enabled:
            self.decode_vision = nn.Linear(feat, cfg.vision_dim)
