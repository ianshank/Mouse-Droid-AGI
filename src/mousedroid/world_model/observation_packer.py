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

* ``vision``       — ``(1, cfg.vision_dim)``                       ``float32``
* ``motor``        — ``(1, cfg.motor_state_dim)``                  ``float32``
* ``valid_mask``   — ``(1, N_SENSOR_MODALITIES_WITH_LIDAR)`` = 5   ``float32``
* ``ultrasonic``   — ``(1, cfg.ultrasonic_dim)`` *or* ``None`` when ``cfg.ultrasonic_dim == 0``
* ``audio``        — ``(1, cfg.audio_dim)``     *or* ``None`` when ``cfg.audio_dim == 0``
* ``lidar``        — ``(1, cfg.lidar_dim)``     *or* ``None`` when ``cfg.lidar_dim == 0``

The ``valid_mask`` width is normalised to
:data:`mousedroid.constants.N_SENSOR_MODALITIES_WITH_LIDAR` regardless of how
many slots the source observation populates. Observations that arrive narrower
(e.g. 4-wide masks from non-LiDAR deployments) are right-padded with zeros so
disabled slots count as invalid. This pins the ONNX-side ``valid_mask`` shape
to a stable value across configurations — ORT requires exact input shapes at
runtime and a varying mask width would crash the ONNX engine.

Disabled-modality contract: when ``cfg.<modality>_dim == 0`` the corresponding
modality tensor is ``None`` (no zero-tensor fallback). The caller — typically
the encoder forward — already branches on whether each modality is enabled
and must not depend on receiving a synthetic zero tensor.

Missing-data contract: when a modality is *enabled* in ``cfg`` but the
observation arrives with no data (``audio_chunk`` of length 0, or
``lidar_features=None``):
  * The modality tensor is a zero-filled tensor of the configured shape, AND
  * The corresponding slot in ``valid_mask`` is forced to 0.

Both halves are needed. The encoder gates each projected modality by its
``valid_mask`` slot via ``_gate_projection``; a zero input fed through the
``Linear`` projection produces ``relu(bias)`` (generally non-zero), which
would diverge from :class:`MultimodalEncoder`'s native "data missing"
path (zeros in projected space, bypassing bias). Zeroing the mask slot makes
``_gate_projection`` multiply the contribution by 0 in both paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from numpy.typing import NDArray
from torch import Tensor

from mousedroid.config.schema import ModelConfig
from mousedroid.constants import N_SENSOR_MODALITIES_WITH_LIDAR, SENSOR_SLOT_MAP
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


def _normalised_valid_mask(
    raw: NDArray[Any],
    *,
    device: torch.device,
) -> Tensor:
    """Right-pad/truncate ``raw`` to a fixed-width valid_mask of shape ``(1, N)``.

    ``N = N_SENSOR_MODALITIES_WITH_LIDAR`` regardless of how many slots the
    incoming observation actually populated. Older sensor bundles produce
    4-wide masks (vision/ultrasonic/motor/audio) on non-LiDAR deployments;
    the ONNX-side ``valid_mask`` shape must be deployment-independent so
    ORT runtime input validation doesn't fail on cross-deployment scrapes.

    Padding semantics: any slot beyond the source mask defaults to 0
    (invalid). That matches the encoder's gating: a 0 slot means the
    modality contributes 0 to the fused embedding, exactly as if the
    sensor were physically absent.
    """
    tensor = _as_tensor(raw, device=device)
    width = tensor.shape[-1]
    if width == N_SENSOR_MODALITIES_WITH_LIDAR:
        return tensor
    if width > N_SENSOR_MODALITIES_WITH_LIDAR:
        return tensor[..., :N_SENSOR_MODALITIES_WITH_LIDAR]
    padding = torch.zeros(
        (1, N_SENSOR_MODALITIES_WITH_LIDAR - width),
        dtype=torch.float32,
        device=device,
    )
    return torch.cat([tensor, padding], dim=-1)


def _zero_valid_mask_slot(mask: Tensor, modality: str) -> Tensor:
    """Return a copy of ``mask`` with the ``modality``'s slot zeroed.

    Used when an enabled modality has no data this tick — see the
    missing-data contract in the module docstring. The encoder gates
    each modality's contribution by its mask slot, so zeroing the slot
    makes the missing-data case match the encoder's native "no data"
    path (which produces zeros in projected space rather than
    ``relu(bias)``).
    """
    slot = SENSOR_SLOT_MAP[modality]
    if mask.shape[-1] <= slot:
        # Slot beyond the mask width — already implicitly invalid.
        return mask
    mask = mask.clone()
    mask[..., slot] = 0.0
    return mask


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
    # Width-normalise: cross-deployment observation masks are 4-wide (no
    # LiDAR) or 5-wide (with LiDAR); ONNX needs a stable shape.
    valid_mask = _normalised_valid_mask(observation.valid_mask, device=device)

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
            # Audio modality enabled but no data this tick — return a zero
            # tensor AND zero the audio slot in valid_mask so the encoder's
            # gating multiplies the contribution by 0. Without the mask
            # update, the Linear projection would emit ``relu(bias)`` which
            # is generally non-zero — diverging from the encoder's native
            # "data missing" path that uses zeros in projected space.
            audio = torch.zeros((1, cfg.audio_dim), dtype=torch.float32, device=device)
            valid_mask = _zero_valid_mask_slot(valid_mask, "audio")
    else:
        audio = None

    lidar: Tensor | None
    if cfg.lidar_dim > 0:
        lidar_data = observation.lidar_features
        if lidar_data is not None and len(lidar_data) > 0:
            lidar = _as_tensor(lidar_data, device=device)
        else:
            # See audio branch above for the rationale on zeroing the mask.
            lidar = torch.zeros((1, cfg.lidar_dim), dtype=torch.float32, device=device)
            valid_mask = _zero_valid_mask_slot(valid_mask, "lidar")
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
