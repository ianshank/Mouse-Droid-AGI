"""Unit tests for the replay-record observation adapter + mask synthesis.

These helpers are REUSED by the WS-E2 sequence-batch builder that feeds the
recon-loss regression gate. Pins:

* :func:`record_to_observation` produces an object satisfying the FULL
  :class:`~mousedroid.sensing.protocol.ObservationProtocol` (all 8 members) from
  a :class:`~mousedroid.experience.record.MouseDroidExperienceRecord` — vision /
  distance / motor copied from the record, audio an empty array, lidar ``None``,
  timestamp passed through;
* the synthesized ``valid_mask`` length + order is derived from the LIVE
  ``world_model.encoder`` enabled flags (NOT a literal): length ==
  the cfg modality count (4 with lidar off, 5 with lidar on), order
  ``[vision, ultrasonic, motor, audio, (lidar)]``, with the vision slot ``0``
  when ``vision_features`` is empty (or vision is encoder-disabled).
"""

from __future__ import annotations

import numpy as np
import torch

from mousedroid.config.schema import ModelConfig
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.learning.on_device.seed_states import record_to_observation
from mousedroid.sensing.protocol import ObservationProtocol
from mousedroid.world_model.rssm import RSSM


def _make_world_model(*, lidar: bool = False) -> RSSM:
    """Build a small deterministic RSSM (vision OFF; lidar optional)."""
    torch.manual_seed(0)
    cfg = ModelConfig(
        vision_dim=0,
        vision_proj_dim=0,
        hidden_dim=8,
        latent_dim=4,
        action_dim=3,
        obs_dim=8,
        lidar_dim=6 if lidar else 0,
        lidar_proj_dim=4 if lidar else 0,
    )
    wm = RSSM(cfg)
    wm.eval()
    return wm


def _make_vision_world_model() -> RSSM:
    """Build a small deterministic RSSM with vision ENABLED."""
    torch.manual_seed(0)
    cfg = ModelConfig(
        vision_dim=12,
        vision_proj_dim=6,
        hidden_dim=8,
        latent_dim=4,
        action_dim=3,
        obs_dim=8,
    )
    wm = RSSM(cfg)
    wm.eval()
    return wm


def _make_records(n: int, *, vision_dim: int = 0) -> list[MouseDroidExperienceRecord]:
    """Build ``n`` deterministic records with monotonically increasing fields."""
    records: list[MouseDroidExperienceRecord] = []
    for i in range(n):
        vision = (
            np.full(vision_dim, float(i), dtype=np.float32)
            if vision_dim > 0
            else np.zeros(0, dtype=np.float32)
        )
        records.append(
            MouseDroidExperienceRecord(
                timestamp=float(i),
                vision_features=vision,
                distance_m=0.5 + 0.1 * i,
                motor_state=np.array([0.1 * i, 0.0, 0.0, 12.0], dtype=np.float32),
                action=np.array([0.2 * i, 0.0, 0.0], dtype=np.float32),
                reward=float(i),
                surprise=0.0,
            )
        )
    return records


# ---------------------------------------------------------------------------
# record_to_observation — protocol conformance + mask synthesis
# ---------------------------------------------------------------------------


def test_record_to_observation_satisfies_protocol() -> None:
    """The adapter is a structural ``ObservationProtocol`` with all members."""
    rec = _make_records(1)[0]
    mask = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)

    obs = record_to_observation(rec, valid_mask=mask)

    assert isinstance(obs, ObservationProtocol)
    # Every member of the FULL protocol is present + correctly typed.
    assert obs.timestamp == rec.timestamp
    assert np.array_equal(obs.vision_features, rec.vision_features)
    assert obs.distance_m == rec.distance_m
    assert np.array_equal(obs.motor_state, rec.motor_state)
    assert obs.audio_chunk.shape == (0,)  # empty audio array
    assert obs.lidar_features is None  # no lidar in replay records
    assert np.array_equal(obs.valid_mask, mask)
    assert obs.n_modalities == len(mask)


def test_mask_length_equals_modality_count_no_lidar() -> None:
    """Mask length == 4 (vision, ultrasonic, motor, audio) when lidar is off."""
    from mousedroid.learning.on_device.seed_states import build_valid_mask

    wm = _make_world_model(lidar=False)
    rec = _make_records(1)[0]
    mask = build_valid_mask(rec, wm.encoder)

    assert mask.shape == (4,)
    assert mask.dtype == np.float32


def test_mask_length_equals_modality_count_with_lidar() -> None:
    """Mask length == 5 when the live encoder has lidar enabled."""
    from mousedroid.learning.on_device.seed_states import build_valid_mask

    wm = _make_world_model(lidar=True)
    assert wm.encoder.lidar_enabled
    rec = _make_records(1)[0]
    mask = build_valid_mask(rec, wm.encoder)

    assert mask.shape == (5,)
    # The lidar slot is 0 — replay records carry no lidar features.
    assert mask[4] == 0.0


def test_mask_vision_slot_zero_for_empty_vision() -> None:
    """Vision slot is 0 when ``vision_features`` is empty (record default)."""
    from mousedroid.learning.on_device.seed_states import build_valid_mask

    wm = _make_world_model()  # vision OFF
    rec = _make_records(1, vision_dim=0)[0]
    mask = build_valid_mask(rec, wm.encoder)

    # Slot order [vision, ultrasonic, motor, audio]; vision empty -> 0.
    assert mask[0] == 0.0
    # Motor is always present.
    assert mask[2] == 1.0


def test_mask_vision_slot_one_for_present_vision() -> None:
    """Vision slot is 1 when vision is enabled AND features are non-empty."""
    from mousedroid.learning.on_device.seed_states import build_valid_mask

    wm = _make_vision_world_model()
    assert wm.encoder.vision_enabled
    rec = _make_records(1, vision_dim=wm.cfg.vision_dim)[0]
    mask = build_valid_mask(rec, wm.encoder)

    assert mask[0] == 1.0


def test_mask_audio_and_ultrasonic_slots_for_enabled_encoder() -> None:
    """Audio + ultrasonic slots are 1 when the encoder enables those branches."""
    from mousedroid.learning.on_device.seed_states import build_valid_mask

    torch.manual_seed(0)
    cfg = ModelConfig(
        vision_dim=0,
        vision_proj_dim=0,
        ultrasonic_dim=1,
        ultrasonic_proj_dim=4,
        audio_dim=8,
        audio_proj_dim=4,
        hidden_dim=8,
        latent_dim=4,
        action_dim=3,
        obs_dim=8,
    )
    wm = RSSM(cfg)
    assert wm.encoder.audio_enabled
    assert wm.encoder.ultrasonic_enabled
    rec = _make_records(1)[0]

    mask = build_valid_mask(rec, wm.encoder)

    assert mask.shape == (4,)
    assert mask[1] == 1.0  # ultrasonic slot
    assert mask[3] == 1.0  # audio slot
