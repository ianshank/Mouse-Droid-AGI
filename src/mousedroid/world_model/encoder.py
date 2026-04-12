"""Multimodal encoder — fuses vision, ultrasonic, motor, and optional audio."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.config.schema import ModelConfig
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class MultimodalEncoder(nn.Module):
    """Project and fuse heterogeneous sensor streams into a single embedding.

    Each modality is linearly projected then gated by the corresponding slice
    of *valid_mask* before concatenation and fusion.  Audio is optional and
    controlled by ``cfg.audio_dim`` — when zero the encoder behaves identically
    to the original 3-modality version for full backwards compatibility.

    Args:
        cfg: Model configuration with all dimension parameters.
    """

    # Indices into the valid_mask: vision, ultrasonic, motor, audio.
    _VISION_IDX = 0
    _ULTRASONIC_IDX = 1
    _MOTOR_IDX = 2
    _AUDIO_IDX = 3

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.vision_proj = nn.Linear(cfg.vision_dim, cfg.vision_proj_dim)
        self.ultrasonic_proj = nn.Linear(cfg.ultrasonic_dim, cfg.ultrasonic_proj_dim)
        self.motor_proj = nn.Linear(cfg.motor_state_dim, cfg.motor_proj_dim)

        self._audio_enabled = cfg.audio_dim > 0 and cfg.audio_proj_dim > 0
        fused_dim = cfg.vision_proj_dim + cfg.ultrasonic_proj_dim + cfg.motor_proj_dim
        if self._audio_enabled:
            self.audio_proj = nn.Linear(cfg.audio_dim, cfg.audio_proj_dim)
            fused_dim += cfg.audio_proj_dim

        self.fusion = nn.Linear(fused_dim, cfg.obs_dim)
        self.act = nn.ReLU()

        _log.info(
            "encoder_init",
            vision_proj=cfg.vision_proj_dim,
            ultrasonic_proj=cfg.ultrasonic_proj_dim,
            motor_proj=cfg.motor_proj_dim,
            audio_proj=cfg.audio_proj_dim if self._audio_enabled else 0,
            audio_enabled=self._audio_enabled,
            fused_dim=fused_dim,
            obs_dim=cfg.obs_dim,
        )

    def forward(
        self,
        vision: Tensor,
        ultrasonic: Tensor,
        motor_state: Tensor,
        valid_mask: Tensor,
        audio: Tensor | None = None,
    ) -> Tensor:
        """Encode multimodal observation into a single embedding.

        Args:
            vision: Vision features, shape ``(batch, vision_dim)``.
            ultrasonic: Ultrasonic reading, shape ``(batch, ultrasonic_dim)``.
            motor_state: Motor state, shape ``(batch, motor_state_dim)``.
            valid_mask: Per-modality validity, shape ``(batch, n_modalities)``.
                Supports both 3-element (legacy) and 4-element masks.
            audio: Optional audio features, shape ``(batch, audio_dim)``.
                Ignored when audio is disabled (``audio_dim=0``).

        Returns:
            Fused observation embedding, shape ``(batch, obs_dim)``.
        """
        v = self.act(self.vision_proj(vision))
        u = self.act(self.ultrasonic_proj(ultrasonic))
        m = self.act(self.motor_proj(motor_state))

        # Gate each projection by its validity score.
        v = v * valid_mask[:, self._VISION_IDX : self._VISION_IDX + 1]
        u = u * valid_mask[:, self._ULTRASONIC_IDX : self._ULTRASONIC_IDX + 1]
        m = m * valid_mask[:, self._MOTOR_IDX : self._MOTOR_IDX + 1]

        parts: list[Tensor] = [v, u, m]

        if self._audio_enabled:
            if audio is not None:
                a = self.act(self.audio_proj(audio))
            else:
                # Audio enabled but no data provided — use zeros.
                batch_size = vision.shape[0]
                a = torch.zeros(
                    batch_size,
                    self.audio_proj.out_features,
                    device=vision.device,
                    dtype=vision.dtype,
                )
            # Gate by audio validity if mask has enough elements.
            if valid_mask.shape[-1] > self._AUDIO_IDX:
                a = a * valid_mask[:, self._AUDIO_IDX : self._AUDIO_IDX + 1]
            parts.append(a)

        fused: Tensor = self.fusion(torch.cat(parts, dim=-1))
        return fused
