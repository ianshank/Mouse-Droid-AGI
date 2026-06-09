"""Multimodal encoder — fuses vision, ultrasonic, motor, and optional audio."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.config.schema import ModelConfig
from mousedroid.constants import SENSOR_SLOT_MAP
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class MultimodalEncoder(nn.Module):
    """Project and fuse heterogeneous sensor streams into a single embedding.

    Each modality is linearly projected then gated by the corresponding slice
    of *valid_mask* before concatenation and fusion.  Vision, audio and LiDAR
    are optional and controlled by ``cfg.vision_dim`` / ``cfg.audio_dim`` /
    ``cfg.lidar_dim`` — when zero the branch is dropped entirely.  The default
    ``vision_dim=256`` keeps the deployed model byte-identical to pre-feature
    builds (invariant #9).

    Args:
        cfg: Model configuration with all dimension parameters.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()

        self._vision_enabled = cfg.vision_dim > 0 and cfg.vision_proj_dim > 0
        self._ultrasonic_enabled = cfg.ultrasonic_dim > 0 and cfg.ultrasonic_proj_dim > 0
        self._audio_enabled = cfg.audio_dim > 0 and cfg.audio_proj_dim > 0
        self._lidar_enabled = cfg.lidar_dim > 0 and cfg.lidar_proj_dim > 0

        # Construction order preserves the original (vision -> motor -> ...) so the
        # seeded weight init is byte-identical for the default (vision-on) config.
        fused_dim = cfg.motor_proj_dim
        if self._vision_enabled:
            self.vision_proj = nn.Linear(cfg.vision_dim, cfg.vision_proj_dim)
            fused_dim += cfg.vision_proj_dim
        self.motor_proj = nn.Linear(cfg.motor_state_dim, cfg.motor_proj_dim)

        if self._ultrasonic_enabled:
            self.ultrasonic_proj = nn.Linear(cfg.ultrasonic_dim, cfg.ultrasonic_proj_dim)
            fused_dim += cfg.ultrasonic_proj_dim
        if self._audio_enabled:
            self.audio_proj = nn.Linear(cfg.audio_dim, cfg.audio_proj_dim)
            fused_dim += cfg.audio_proj_dim
        if self._lidar_enabled:
            self.lidar_proj = nn.Linear(cfg.lidar_dim, cfg.lidar_proj_dim)
            fused_dim += cfg.lidar_proj_dim

        self.fusion = nn.Linear(fused_dim, cfg.obs_dim)
        self.act = nn.ReLU()

        _log.info(
            "encoder_init",
            vision_proj=cfg.vision_proj_dim if self._vision_enabled else 0,
            vision_enabled=self._vision_enabled,
            ultrasonic_proj=cfg.ultrasonic_proj_dim if self._ultrasonic_enabled else 0,
            ultrasonic_enabled=self._ultrasonic_enabled,
            motor_proj=cfg.motor_proj_dim,
            audio_proj=cfg.audio_proj_dim if self._audio_enabled else 0,
            audio_enabled=self._audio_enabled,
            lidar_proj=cfg.lidar_proj_dim if self._lidar_enabled else 0,
            lidar_enabled=self._lidar_enabled,
            fused_dim=fused_dim,
            obs_dim=cfg.obs_dim,
        )

    @property
    def vision_enabled(self) -> bool:
        """Whether vision modality is active."""
        return self._vision_enabled

    @property
    def audio_enabled(self) -> bool:
        """Whether audio modality is active."""
        return self._audio_enabled

    @property
    def ultrasonic_enabled(self) -> bool:
        """Whether ultrasonic modality is active."""
        return self._ultrasonic_enabled

    @property
    def lidar_enabled(self) -> bool:
        """Whether LiDAR modality is active."""
        return self._lidar_enabled

    @staticmethod
    def _gate_projection(projected: Tensor, valid_mask: Tensor, modality_name: str) -> Tensor:
        """Gate a projected modality by its valid-mask slot."""
        slot_index = SENSOR_SLOT_MAP[modality_name]
        if valid_mask.shape[-1] <= slot_index:
            return torch.zeros_like(projected)
        return projected * valid_mask[:, slot_index : slot_index + 1]

    def forward(
        self,
        vision: Tensor | None,
        ultrasonic: Tensor | None,
        motor_state: Tensor,
        valid_mask: Tensor,
        audio: Tensor | None = None,
        lidar: Tensor | None = None,
    ) -> Tensor:
        """Encode multimodal observation into a single embedding.

        Args:
            vision: Vision features, shape ``(batch, vision_dim)``, or ``None``
                when vision is disabled (``vision_dim=0``).
            ultrasonic: Ultrasonic reading, shape ``(batch, ultrasonic_dim)``.
            motor_state: Motor state, shape ``(batch, motor_state_dim)``.
            valid_mask: Per-modality validity, shape ``(batch, n_modalities)``.
                Supports 3-element (legacy), 4-element, and 5-element masks.
            audio: Optional audio features, shape ``(batch, audio_dim)``.
                Ignored when audio is disabled (``audio_dim=0``).
            lidar: Optional LiDAR features, shape ``(batch, lidar_dim)``.
                Ignored when LiDAR is disabled (``lidar_dim=0``).

        Returns:
            Fused observation embedding, shape ``(batch, obs_dim)``.

        Raises:
            ValueError: If vision is enabled but ``vision`` tensor is ``None``.
        """
        # ``motor_state`` is always present; use it as the reference tensor for
        # zero-filling so vision-disabled paths don't need a camera tensor.
        ref = motor_state

        # Concat order preserves the original [vision, ultrasonic, motor, audio,
        # lidar] so a checkpoint's fusion weights stay valid (byte-identical
        # default). Motor is always present; the rest are gated by config.
        parts: list[Tensor] = []

        if self._vision_enabled:
            if vision is None:
                msg = "vision tensor must be provided when vision_dim > 0"
                raise ValueError(msg)
            v = self.act(self.vision_proj(vision))
            parts.append(self._gate_projection(v, valid_mask, "vision"))

        if self._ultrasonic_enabled:
            if ultrasonic is not None:
                u = self.act(self.ultrasonic_proj(ultrasonic))
            else:
                u = torch.zeros(
                    ref.shape[0],
                    self.ultrasonic_proj.out_features,
                    device=ref.device,
                    dtype=ref.dtype,
                )
            parts.append(self._gate_projection(u, valid_mask, "ultrasonic"))

        m = self.act(self.motor_proj(motor_state))
        parts.append(self._gate_projection(m, valid_mask, "motor"))

        if self._audio_enabled:
            if audio is not None:
                a = self.act(self.audio_proj(audio))
            else:
                # Audio enabled but no data provided — use zeros.
                a = torch.zeros(
                    ref.shape[0],
                    self.audio_proj.out_features,
                    device=ref.device,
                    dtype=ref.dtype,
                )
            parts.append(self._gate_projection(a, valid_mask, "audio"))

        if self._lidar_enabled:
            if lidar is not None:
                el = self.act(self.lidar_proj(lidar))
            else:
                # LiDAR enabled but no data provided — use zeros.
                el = torch.zeros(
                    ref.shape[0],
                    self.lidar_proj.out_features,
                    device=ref.device,
                    dtype=ref.dtype,
                )
            parts.append(self._gate_projection(el, valid_mask, "lidar"))

        fused: Tensor = self.fusion(torch.cat(parts, dim=-1))
        return fused
