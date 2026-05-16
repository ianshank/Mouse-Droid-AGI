"""Unit tests for ``world_model.observation_packer``.

The packer is the single source of truth that converts an
``ObservationProtocol`` (the duck-typed sensor bundle the orchestrator
hands off) into the dense ``Tensor`` set the dual-stream RSSM consumes
in both PyTorch and ONNX runtime modes. Production ``observe_step()``
calls it, and so does ``DualStreamRSSMOnnx.observe_step()``. This test
suite locks in:

1. Shape contract per modality (vision, ultrasonic, motor, audio, lidar, mask)
2. Disabled-modality handling (``cfg.audio_dim == 0``, etc.)
3. Empty-buffer handling (``audio_chunk`` of length 0)
4. ``lidar_features=None`` handling
5. Device/dtype invariants (float32, target device)
6. No silent ``ultrasonic`` injection — when ``ultrasonic_dim=0``
   the packer must NOT return an ultrasonic tensor (rover baseline)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from mousedroid.config.schema import ModelConfig
from mousedroid.world_model.observation_packer import (
    pack_observation,
)


@dataclass
class _StubObservation:
    """Minimal ObservationProtocol-compatible stub for tests."""

    timestamp: float = 0.0
    vision_features: NDArray[np.float32] | None = None
    distance_m: float = 1.5
    motor_state: NDArray[np.float32] | None = None
    audio_chunk: NDArray[np.float32] | None = None
    valid_mask: NDArray[np.float32] | None = None
    n_modalities: int = 5
    lidar_features: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        if self.vision_features is None:
            self.vision_features = np.zeros(16, dtype=np.float32)
        if self.motor_state is None:
            self.motor_state = np.zeros(4, dtype=np.float32)
        if self.audio_chunk is None:
            self.audio_chunk = np.zeros(0, dtype=np.float32)
        if self.valid_mask is None:
            self.valid_mask = np.ones(5, dtype=np.float32)


def _base_cfg(
    *,
    vision_dim: int = 16,
    ultrasonic_dim: int = 1,
    motor_state_dim: int = 4,
    audio_dim: int = 0,
    lidar_dim: int = 0,
) -> ModelConfig:
    """Minimal ModelConfig for packer tests. Ultrasonic on by default.

    The Pydantic ``ModelConfig`` validator requires that ``<modality>_dim``
    and ``<modality>_proj_dim`` are both zero or both positive — mirror
    that invariant here so tests construct legal configs.
    """
    ultrasonic_proj = 4 if ultrasonic_dim > 0 else 0
    audio_proj = 4 if audio_dim > 0 else 0
    lidar_proj = 4 if lidar_dim > 0 else 0
    return ModelConfig(
        vision_dim=vision_dim,
        ultrasonic_dim=ultrasonic_dim,
        ultrasonic_proj_dim=ultrasonic_proj,
        motor_state_dim=motor_state_dim,
        audio_dim=audio_dim,
        audio_proj_dim=audio_proj,
        lidar_dim=lidar_dim,
        lidar_proj_dim=lidar_proj,
    )


class TestPackObservationShapes:
    """Shape contract per modality."""

    def test_vision_shape_is_unsqueezed_to_batch_1(self) -> None:
        cfg = _base_cfg(vision_dim=16)
        obs = _StubObservation()
        packed = pack_observation(obs, cfg, device=torch.device("cpu"))
        assert packed.vision.shape == (1, 16)

    def test_motor_shape_is_unsqueezed_to_batch_1(self) -> None:
        cfg = _base_cfg(motor_state_dim=4)
        obs = _StubObservation()
        packed = pack_observation(obs, cfg, device=torch.device("cpu"))
        assert packed.motor.shape == (1, 4)

    def test_ultrasonic_shape_when_enabled(self) -> None:
        cfg = _base_cfg(ultrasonic_dim=1)
        obs = _StubObservation(distance_m=2.5)
        packed = pack_observation(obs, cfg, device=torch.device("cpu"))
        assert packed.ultrasonic is not None
        assert packed.ultrasonic.shape == (1, 1)
        assert packed.ultrasonic.item() == pytest.approx(2.5, abs=1e-6)

    def test_valid_mask_shape_preserves_n_modalities(self) -> None:
        cfg = _base_cfg()
        obs = _StubObservation(valid_mask=np.array([1.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32))
        packed = pack_observation(obs, cfg, device=torch.device("cpu"))
        assert packed.valid_mask.shape == (1, 5)


class TestPackObservationDisabledModalities:
    """Optional modalities (``ultrasonic_dim=0``, ``audio_dim=0``, ``lidar_dim=0``)."""

    def test_ultrasonic_disabled_returns_none(self) -> None:
        # Pydantic requires lidar_dim > 0 if ultrasonic_dim == 0 (at least one
        # distance modality must be active).
        cfg = _base_cfg(ultrasonic_dim=0, lidar_dim=12)
        obs = _StubObservation(distance_m=2.5)  # value is ignored
        packed = pack_observation(obs, cfg, device=torch.device("cpu"))
        assert packed.ultrasonic is None

    def test_audio_disabled_returns_none(self) -> None:
        cfg = _base_cfg(audio_dim=0)
        obs = _StubObservation(audio_chunk=np.zeros(0, dtype=np.float32))
        packed = pack_observation(obs, cfg, device=torch.device("cpu"))
        assert packed.audio is None

    def test_lidar_disabled_returns_none(self) -> None:
        cfg = _base_cfg(lidar_dim=0)
        obs = _StubObservation(lidar_features=None)
        packed = pack_observation(obs, cfg, device=torch.device("cpu"))
        assert packed.lidar is None


class TestPackObservationAudioBuffer:
    """Audio-specific buffer handling."""

    def test_audio_enabled_with_buffer(self) -> None:
        cfg = _base_cfg(audio_dim=8)
        chunk = np.arange(8, dtype=np.float32)
        obs = _StubObservation(audio_chunk=chunk)
        packed = pack_observation(obs, cfg, device=torch.device("cpu"))
        assert packed.audio is not None
        assert packed.audio.shape == (1, 8)
        assert torch.allclose(
            packed.audio.flatten(),
            torch.from_numpy(chunk),
            atol=1e-6,
        )

    def test_audio_enabled_with_empty_buffer_returns_zero_tensor(self) -> None:
        cfg = _base_cfg(audio_dim=8)
        obs = _StubObservation(audio_chunk=np.zeros(0, dtype=np.float32))
        packed = pack_observation(obs, cfg, device=torch.device("cpu"))
        assert packed.audio is not None
        assert packed.audio.shape == (1, 8)
        assert torch.all(packed.audio == 0.0)


class TestPackObservationLidar:
    """LiDAR-specific handling."""

    def test_lidar_enabled_with_features(self) -> None:
        cfg = _base_cfg(lidar_dim=12)
        feats = np.arange(12, dtype=np.float32)
        obs = _StubObservation(lidar_features=feats)
        packed = pack_observation(obs, cfg, device=torch.device("cpu"))
        assert packed.lidar is not None
        assert packed.lidar.shape == (1, 12)

    def test_lidar_enabled_with_none_returns_zero_tensor(self) -> None:
        cfg = _base_cfg(lidar_dim=12)
        obs = _StubObservation(lidar_features=None)
        packed = pack_observation(obs, cfg, device=torch.device("cpu"))
        assert packed.lidar is not None
        assert packed.lidar.shape == (1, 12)
        assert torch.all(packed.lidar == 0.0)


class TestPackObservationDeviceDtype:
    """Device and dtype invariants — float32 + caller-supplied device."""

    def test_all_tensors_are_float32(self) -> None:
        cfg = _base_cfg(audio_dim=8, lidar_dim=12)
        obs = _StubObservation(
            audio_chunk=np.ones(8, dtype=np.float32),
            lidar_features=np.ones(12, dtype=np.float32),
        )
        packed = pack_observation(obs, cfg, device=torch.device("cpu"))
        assert packed.vision.dtype == torch.float32
        assert packed.motor.dtype == torch.float32
        assert packed.valid_mask.dtype == torch.float32
        assert packed.ultrasonic is not None
        assert packed.ultrasonic.dtype == torch.float32
        assert packed.audio is not None
        assert packed.audio.dtype == torch.float32
        assert packed.lidar is not None
        assert packed.lidar.dtype == torch.float32

    def test_target_device_is_respected(self) -> None:
        cfg = _base_cfg()
        obs = _StubObservation()
        device = torch.device("cpu")
        packed = pack_observation(obs, cfg, device=device)
        assert packed.vision.device == device
        assert packed.motor.device == device
        assert packed.valid_mask.device == device


class TestPackObservationRoverBaseline:
    """Regression net: the rover baseline production config must not
    silently re-enable ultrasonic when both ultrasonic and lidar are
    available — operator opts in/out explicitly via cfg dimensions.

    This test does NOT prove rover-only deployment; it just confirms the
    packer respects ``cfg.ultrasonic_dim`` as the single source of truth.
    """

    def test_ultrasonic_disabled_emits_no_ultrasonic_tensor(self) -> None:
        cfg = _base_cfg(ultrasonic_dim=0, lidar_dim=12)
        obs = _StubObservation(distance_m=42.0)  # bogus value
        packed = pack_observation(obs, cfg, device=torch.device("cpu"))
        assert packed.ultrasonic is None


class TestPackedObservationDataclass:
    """``PackedObservation`` is a typed container — verify the shape."""

    def test_packed_is_a_dataclass(self) -> None:
        cfg = _base_cfg()
        packed = pack_observation(_StubObservation(), cfg, device=torch.device("cpu"))
        # All required fields present
        assert hasattr(packed, "vision")
        assert hasattr(packed, "motor")
        assert hasattr(packed, "valid_mask")
        assert hasattr(packed, "ultrasonic")
        assert hasattr(packed, "audio")
        assert hasattr(packed, "lidar")


class TestValidMaskWidthNormalization:
    """``valid_mask`` is right-padded/truncated to a stable width.

    Without normalisation, cross-deployment masks (4-wide without LiDAR,
    5-wide with LiDAR) crash the ONNX engine — ORT requires exact input
    shapes at runtime.
    """

    def test_four_wide_mask_is_padded_to_five(self) -> None:
        from mousedroid.constants import N_SENSOR_MODALITIES_WITH_LIDAR

        cfg = _base_cfg()
        # Older sensing produces 4-wide masks (vision/ultrasonic/motor/audio).
        narrow_mask = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        obs = _StubObservation(valid_mask=narrow_mask)
        packed = pack_observation(obs, cfg, device=torch.device("cpu"))
        # Width is exactly the canonical N_SENSOR_MODALITIES_WITH_LIDAR.
        assert packed.valid_mask.shape == (1, N_SENSOR_MODALITIES_WITH_LIDAR)
        # First 4 slots preserved; padding slot is 0 (invalid).
        assert torch.all(packed.valid_mask[..., :4] == 1.0)
        assert packed.valid_mask[..., 4].item() == 0.0

    def test_five_wide_mask_passes_through_unchanged(self) -> None:
        from mousedroid.constants import N_SENSOR_MODALITIES_WITH_LIDAR

        cfg = _base_cfg()
        mask = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        obs = _StubObservation(valid_mask=mask)
        packed = pack_observation(obs, cfg, device=torch.device("cpu"))
        assert packed.valid_mask.shape == (1, N_SENSOR_MODALITIES_WITH_LIDAR)
        assert torch.allclose(packed.valid_mask.flatten(), torch.from_numpy(mask), atol=1e-6)

    def test_wider_mask_is_truncated(self) -> None:
        """6-wide mask (defensive — no current sensor emits this) is truncated."""
        from mousedroid.constants import N_SENSOR_MODALITIES_WITH_LIDAR

        cfg = _base_cfg()
        mask = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.5], dtype=np.float32)
        obs = _StubObservation(valid_mask=mask, n_modalities=6)
        packed = pack_observation(obs, cfg, device=torch.device("cpu"))
        assert packed.valid_mask.shape == (1, N_SENSOR_MODALITIES_WITH_LIDAR)
        # The trailing 0.5 is dropped — only the first 5 slots survive.
        assert torch.all(packed.valid_mask[..., :5] == 1.0)


class TestMissingDataMaskZeroing:
    """Enabled-but-missing modality data zeroes the slot in valid_mask.

    Without this, the encoder's ``_gate_projection`` would multiply the
    Linear projection of zero (= ``relu(bias)``, generally non-zero) by
    a non-zero mask slot, contributing junk activations. Zeroing the
    slot makes the missing-data case match the encoder's native "no data"
    path (zeros in projected space, bypassing bias).
    """

    def test_audio_missing_zeros_audio_slot(self) -> None:
        from mousedroid.constants import SENSOR_SLOT_MAP

        cfg = _base_cfg(audio_dim=8)
        # All slots initially valid (1.0), but audio data is empty.
        obs = _StubObservation(
            audio_chunk=np.zeros(0, dtype=np.float32),
            valid_mask=np.ones(5, dtype=np.float32),
        )
        packed = pack_observation(obs, cfg, device=torch.device("cpu"))
        audio_slot = SENSOR_SLOT_MAP["audio"]
        assert packed.valid_mask[..., audio_slot].item() == 0.0
        # Other slots untouched.
        for slot_name in ("vision", "ultrasonic", "motor"):
            slot = SENSOR_SLOT_MAP[slot_name]
            assert packed.valid_mask[..., slot].item() == 1.0

    def test_lidar_missing_zeros_lidar_slot(self) -> None:
        from mousedroid.constants import SENSOR_SLOT_MAP

        cfg = _base_cfg(lidar_dim=12)
        obs = _StubObservation(
            lidar_features=None,
            valid_mask=np.ones(5, dtype=np.float32),
        )
        packed = pack_observation(obs, cfg, device=torch.device("cpu"))
        lidar_slot = SENSOR_SLOT_MAP["lidar"]
        assert packed.valid_mask[..., lidar_slot].item() == 0.0

    def test_audio_present_preserves_audio_slot(self) -> None:
        """When audio data is provided, the audio slot stays at the source value."""
        from mousedroid.constants import SENSOR_SLOT_MAP

        cfg = _base_cfg(audio_dim=8)
        obs = _StubObservation(
            audio_chunk=np.ones(8, dtype=np.float32),
            valid_mask=np.ones(5, dtype=np.float32),
        )
        packed = pack_observation(obs, cfg, device=torch.device("cpu"))
        audio_slot = SENSOR_SLOT_MAP["audio"]
        assert packed.valid_mask[..., audio_slot].item() == 1.0
