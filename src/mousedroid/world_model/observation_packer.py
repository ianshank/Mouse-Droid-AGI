"""Observation → tensor packing — single source of truth.

Both the PyTorch :class:`~mousedroid.world_model.dual_stream_rssm.DualStreamRSSM`
and the ONNX runtime :class:`~mousedroid.world_model.dual_stream_rssm_onnx.DualStreamRSSMOnnx`
must convert an ``ObservationProtocol`` into the dense ``Tensor`` set the
encoder consumes. Without a shared packer the two paths drift over time
(different dtype handling, different empty-buffer fallbacks, different
disabled-modality semantics) and the cross-engine equivalence guarantee
in ``cfg.world_model.engine`` evaporates.

This module owns that conversion exactly once.

Tensor shapes returned by :func:`pack_observation`:

* ``vision``       — ``(1, cfg.vision_dim)``     ``float32``
* ``motor``        — ``(1, cfg.motor_state_dim)``  ``float32``
* ``valid_mask``   — ``(1, n_modalities)``         ``float32``
* ``ultrasonic``   — ``(1, cfg.ultrasonic_dim)``   ``float32`` *or* ``None``
                     when ``cfg.ultrasonic_dim == 0``
* ``audio``        — ``(1, cfg.audio_dim)``        ``float32`` *or* ``None``
                     when ``cfg.audio_dim == 0``
* ``lidar``        — ``(1, cfg.lidar_dim)``        ``float32`` *or* ``None``
                     when ``cfg.lidar_dim == 0``

Disabled-modality contract: when ``cfg.<modality>_dim == 0`` the corresponding
field is ``None`` (no zero-tensor fallback). The caller — typically the
encoder forward — already branches on whether each modality is enabled and
must not depend on receiving a synthetic zero tensor.

Missing-data contract: when a modality is *enabled* in ``cfg`` but the
observation arrives with no data (``audio_chunk`` of length 0, or
``lidar_features=None``) the packer returns a zero tensor of the correct
shape. This matches the existing fallback baked into
:class:`MultimodalEncoder.forward` so the runtime path produces identical
output to the PyTorch reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from numpy.typing import NDArray
from torch import Tensor

from mousedroid.config.schema import ModelConfig
from mousedroid.sensing.protocol import ObservationProtocol

# No module-level logger here on purpose: the packer is a pure stateless
# converter, every code path is covered by upstream / downstream logs in
# observe_step and DualStreamRSSMOnnx.observe_step. Adding a redundant
# logger would emit ~30Hz times packer-call duplicates of those events.


@dataclass(frozen=True)
class PackedObservation:
    """Typed container for the packer output — one tensor per modality.

    ``ultrasonic``, ``audio``, and ``lidar`` are ``None`` when the
    corresponding ``cfg.<modality>_dim == 0`` (modality disabled).

    Attributes:
        vision: Vision features, shape ``(1, cfg.vision_dim)``.
        motor: Motor state, shape ``(1, cfg.motor_state_dim)``.
        valid_mask: Per-modality validity scores,
            shape ``(1, observation.n_modalities)``.
        ultrasonic: Optional ultrasonic reading, shape ``(1, cfg.ultrasonic_dim)``.
        audio: Optional audio samples, shape ``(1, cfg.audio_dim)``.
        lidar: Optional LiDAR feature vector, shape ``(1, cfg.lidar_dim)``.
    """

    vision: Tensor
    motor: Tensor
    valid_mask: Tensor
    ultrasonic: Tensor | None
    audio: Tensor | None
    lidar: Tensor | None


def _as_tensor(
    array: NDArray[Any],
    *,
    device: torch.device,
) -> Tensor:
    """Convert numpy array to a ``(1, *array.shape)`` float32 Tensor."""
    return torch.as_tensor(array, dtype=torch.float32, device=device).unsqueeze(0)


def pack_observation(
    observation: ObservationProtocol,
    cfg: ModelConfig,
    *,
    device: torch.device,
) -> PackedObservation:
    """Convert an ``ObservationProtocol`` into the tensor set the RSSM consumes.

    Args:
        observation: Sensor bundle implementing :class:`ObservationProtocol`.
        cfg: Model configuration — single source of truth for which
            modalities are enabled (``ultrasonic_dim``, ``audio_dim``,
            ``lidar_dim``).
        device: Target device for all returned tensors. The orchestrator
            passes ``cfg.world_model.device`` here; the ONNX runtime passes
            ``cpu`` because ``onnxruntime`` consumes numpy arrays.

    Returns:
        A :class:`PackedObservation` with one ``(1, dim)`` Tensor per
        enabled modality and ``None`` for disabled ones.
    """
    vision = _as_tensor(observation.vision_features, device=device)
    motor = _as_tensor(observation.motor_state, device=device)
    valid_mask = _as_tensor(observation.valid_mask, device=device)

    ultrasonic: Tensor | None
    if cfg.ultrasonic_dim > 0:
        # The ObservationProtocol exposes distance_m as a scalar float; the
        # encoder expects shape (batch, ultrasonic_dim=1).
        ultrasonic = torch.as_tensor(
            [observation.distance_m], dtype=torch.float32, device=device
        ).unsqueeze(0)
    else:
        ultrasonic = None

    audio: Tensor | None
    if cfg.audio_dim > 0:
        audio_data = observation.audio_chunk
        if len(audio_data) > 0:
            audio = _as_tensor(audio_data, device=device)
        else:
            # Audio modality enabled but no data this tick — encoder expects
            # a zero tensor of the configured shape.
            audio = torch.zeros((1, cfg.audio_dim), dtype=torch.float32, device=device)
    else:
        audio = None

    lidar: Tensor | None
    if cfg.lidar_dim > 0:
        lidar_data = observation.lidar_features
        if lidar_data is not None and len(lidar_data) > 0:
            lidar = _as_tensor(lidar_data, device=device)
        else:
            lidar = torch.zeros((1, cfg.lidar_dim), dtype=torch.float32, device=device)
    else:
        lidar = None

    return PackedObservation(
        vision=vision,
        motor=motor,
        valid_mask=valid_mask,
        ultrasonic=ultrasonic,
        audio=audio,
        lidar=lidar,
    )
