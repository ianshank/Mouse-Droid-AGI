"""Replay-encoded seed states for the on-device regression gate (Phase 6 WS-E1).

The WS4 regression gate scores a candidate against a baseline from a FIXED set of
world-model ``(h, z)`` seed states. #134 produced those by ``manual_seed``-
sampling latents directly — reproducible but ungrounded in any real trajectory.
WS-E1 adds an opt-in, replay-GROUNDED source: roll the live world model's
``observe_step`` from a zero ``(h, z)`` across a held-out slice of recorded
experience, collecting the posterior ``(h, z)`` at each step as a seed state.

The new path is wired behind ``cfg.on_device_learning.seed_state_source``: the
default ``"sampled"`` keeps the #134 behaviour byte-identical; ``"replay_encoded"``
calls :func:`encode_seed_states`.

Two pieces:

* :func:`record_to_observation` — adapt a flat
  :class:`~mousedroid.experience.record.MouseDroidExperienceRecord` (which carries
  only vision / distance / motor / action / reward / surprise) into an object
  satisfying the FULL :class:`~mousedroid.sensing.protocol.ObservationProtocol`
  that ``observe_step`` consumes. Audio is an empty array, lidar is ``None`` (the
  encoder gates both safely), and the ``valid_mask`` is SYNTHESIZED from the live
  encoder's enabled flags (see :func:`build_valid_mask`).
* :func:`encode_seed_states` — the deterministic, ``no_grad`` + ``eval`` rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray

from mousedroid.constants import SENSOR_SLOT_MAP
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.experience.record import MouseDroidExperienceRecord
    from mousedroid.learning.on_device.scoring import SeedState
    from mousedroid.world_model.encoder import MultimodalEncoder
    from mousedroid.world_model.rssm import RSSM

_log = get_logger(__name__)


@dataclass(frozen=True)
class _RecordObservation:
    """Lightweight :class:`ObservationProtocol` view over an experience record.

    Adapts the flat replay record into the full observation surface
    ``observe_step`` expects. The record stores no audio / lidar / mask, so audio
    is an empty array, lidar is ``None`` (the encoder zero-fills both when their
    modality is enabled), and the mask is synthesized by :func:`build_valid_mask`.

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


def encode_seed_states(
    world_model: RSSM,
    records: list[MouseDroidExperienceRecord],
    n_seed: int,
    *,
    device: torch.device,
) -> list[SeedState]:
    """Encode replay-grounded ``(h, z)`` seed states for the regression gate.

    Rolls ``observe_step`` from a zero ``(h, z)`` across ``records`` — the
    ``prev_action`` fed to step ``t`` is ``records[t-1].action`` (zeros at the
    first step) — and collects the posterior ``(h, z)`` after each step as a seed
    state. Returns up to ``n_seed`` states.

    Determinism: ``observe_step`` is ``@torch.no_grad``-decorated and the global
    RNG state is captured + restored, so the same model weights + same records
    ALWAYS yield byte-identical seed states. The model is forced into ``eval()``
    for the rollout and its prior train-mode is restored afterwards. All tensors
    are placed on ``device``; the base model parameters are never mutated.

    An empty ``records`` slice returns an empty list and emits a structured
    ``on_device_seed_states_empty_replay`` warning so the caller falls back to the
    sampled path.

    Args:
        world_model: The live :class:`RSSM` whose encoder + posterior produce the
            latent states.
        records: The held-out replay slice to encode (chronological order).
        n_seed: Maximum number of seed states to return.
        device: The device on which to place the rolled ``(h, z)`` tensors.

    Returns:
        A list of up to ``n_seed`` ``(h, z)`` seed-state pairs (each ``(1, dim)``);
        empty when ``records`` is empty.
    """
    if not records:
        _log.warning("on_device_seed_states_empty_replay", n_seed=n_seed)
        return []

    cfg = world_model.cfg
    rng_state = torch.get_rng_state()
    was_training = world_model.training

    seed_states: list[SeedState] = []
    try:
        world_model.eval()
        with torch.no_grad():
            h = torch.zeros(1, cfg.hidden_dim, device=device)
            z = torch.zeros(1, cfg.latent_dim, device=device)
            prev_action = torch.zeros(1, cfg.action_dim, device=device)
            for record in records:
                mask = build_valid_mask(record, world_model.encoder)
                observation = record_to_observation(record, valid_mask=mask)
                h, z, _recon, _surprise = world_model.observe_step(observation, prev_action, h, z)
                seed_states.append((h, z))
                if len(seed_states) >= n_seed:
                    break
                # The action that PRODUCED the next observation is this record's
                # action (the next observe_step uses it as prev_action).
                prev_action = torch.as_tensor(
                    record.action, dtype=torch.float32, device=device
                ).unsqueeze(0)
    finally:
        torch.set_rng_state(rng_state)
        if was_training:
            world_model.train()

    _log.info(
        "on_device_seed_states_encoded",
        n_records=len(records),
        n_seed_states=len(seed_states),
        device=str(device),
    )
    return seed_states


__all__ = ["build_valid_mask", "encode_seed_states", "record_to_observation"]
