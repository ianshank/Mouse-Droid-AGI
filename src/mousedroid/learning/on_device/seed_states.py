"""Replay-record observation adapter + validity-mask synthesis (Phase 6).

These helpers adapt flat replay records into the world-model observation surface
and are REUSED by the WS-E2 sequence-batch builder
(:func:`mousedroid.learning.on_device.rssm_refiner.build_sequence_batch`) that
feeds the recon-loss regression gate.

Two pieces:

* :func:`record_to_observation` — adapt a flat
  :class:`~mousedroid.experience.record.MouseDroidExperienceRecord` (which carries
  only vision / distance / motor / action / reward / surprise) into an object
  satisfying the FULL :class:`~mousedroid.sensing.protocol.ObservationProtocol`.
  Audio is an empty array, lidar is ``None`` (the encoder gates both safely), and
  the ``valid_mask`` is SYNTHESIZED from the live encoder's enabled flags (see
  :func:`build_valid_mask`).
* :func:`build_valid_mask` — synthesize the per-modality validity mask from the
  LIVE encoder's enabled flags, length 4 (no lidar) or 5 (with lidar).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.constants import SENSOR_SLOT_MAP

if TYPE_CHECKING:
    from mousedroid.experience.record import MouseDroidExperienceRecord
    from mousedroid.world_model.encoder import MultimodalEncoder


@dataclass(frozen=True)
class _RecordObservation:
    """Lightweight :class:`ObservationProtocol` view over an experience record.

    Adapts the flat replay record into the full observation surface the encoder
    expects. The record stores no audio / lidar / mask, so audio is an empty
    array, lidar is ``None`` (the encoder zero-fills both when their modality is
    enabled), and the mask is synthesized by :func:`build_valid_mask`.

    Attributes mirror :class:`~mousedroid.sensing.protocol.ObservationProtocol`
    exactly so this object passes a ``runtime_checkable`` ``isinstance`` check.
    """

    timestamp: float
    vision_features: NDArray[np.float32]
    distance_m: float
    motor_state: NDArray[np.float32]
    audio_chunk: NDArray[np.float32]
    lidar_features: NDArray[np.float32] | None
    valid_mask: NDArray[np.float32]

    @property
    def n_modalities(self) -> int:
        """Number of sensor modalities tracked by ``valid_mask``."""
        return len(self.valid_mask)


def build_valid_mask(
    record: MouseDroidExperienceRecord,
    encoder: MultimodalEncoder,
) -> NDArray[np.float32]:
    """Synthesize a ``valid_mask`` from the LIVE encoder's enabled modalities.

    The length + slot order are derived from the encoder's ``*_enabled`` flags
    (NOT a literal): slots ``[vision, ultrasonic, motor, audio, (lidar)]`` per
    :data:`~mousedroid.constants.SENSOR_SLOT_MAP`. The mask is length 4 when the
    encoder has no lidar branch and length 5 when it does — matching the
    4-or-5 contract the encoder gates safely (a 4-slot mask never indexes the
    lidar slot).

    Slot values:

    * **motor** — always ``1.0`` (motor state is always recorded);
    * **ultrasonic** / **audio** — ``1.0`` when the encoder enables that branch
      (the record always carries a distance scalar; audio is zero-filled by the
      encoder so the gate treats the branch as present-but-padded);
    * **vision** — ``1.0`` only when vision is encoder-enabled AND the record's
      ``vision_features`` is non-empty (an empty vector ⇒ ``0.0`` so the encoder
      gates the vision projection to zero);
    * **lidar** — ``0.0`` (replay records carry no lidar features).

    Args:
        record: The source experience record.
        encoder: The live world-model multimodal encoder.

    Returns:
        A ``float32`` mask of length 4 (no lidar) or 5 (with lidar).
    """
    last_slot = SENSOR_SLOT_MAP["lidar"] if encoder.lidar_enabled else SENSOR_SLOT_MAP["audio"]
    mask = np.zeros(last_slot + 1, dtype=np.float32)

    # Motor is always present.
    mask[SENSOR_SLOT_MAP["motor"]] = 1.0
    if encoder.vision_enabled and record.vision_features.size > 0:
        mask[SENSOR_SLOT_MAP["vision"]] = 1.0
    if encoder.ultrasonic_enabled:
        mask[SENSOR_SLOT_MAP["ultrasonic"]] = 1.0
    if encoder.audio_enabled:
        mask[SENSOR_SLOT_MAP["audio"]] = 1.0
    # Lidar slot (index 4) stays 0.0 — replay records never carry lidar features.
    return mask


def record_to_observation(
    record: MouseDroidExperienceRecord,
    *,
    valid_mask: NDArray[np.float32],
) -> _RecordObservation:
    """Adapt a replay record into a full :class:`ObservationProtocol`.

    Exposes ``vision_features`` / ``distance_m`` / ``motor_state`` / ``timestamp``
    from the record; ``audio_chunk`` is an empty array and ``lidar_features`` is
    ``None`` (the encoder gates both). The caller supplies the synthesized
    ``valid_mask`` (see :func:`build_valid_mask`) so the mask's length + order
    track the live encoder rather than a literal.

    Args:
        record: The source experience record.
        valid_mask: The synthesized per-modality validity mask.

    Returns:
        An :class:`ObservationProtocol`-conforming view over the record.
    """
    return _RecordObservation(
        timestamp=record.timestamp,
        vision_features=record.vision_features,
        distance_m=record.distance_m,
        motor_state=record.motor_state,
        audio_chunk=np.zeros(0, dtype=np.float32),
        lidar_features=None,
        valid_mask=valid_mask,
    )


__all__ = ["build_valid_mask", "record_to_observation"]
