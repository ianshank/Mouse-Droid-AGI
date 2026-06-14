"""Unit tests for the WS-E1 replay-encoded seed-state encoder.

Pins the WS-E1 ENABLEMENT contracts:

* :func:`record_to_observation` produces an object satisfying the FULL
  :class:`~mousedroid.sensing.protocol.ObservationProtocol` (all 8 members) from
  a :class:`~mousedroid.experience.record.MouseDroidExperienceRecord` — vision /
  distance / motor copied from the record, audio an empty array, lidar ``None``,
  timestamp passed through;
* the synthesized ``valid_mask`` length + order is derived from the LIVE
  ``world_model.encoder`` enabled flags (NOT a literal): length ==
  the cfg modality count (4 with lidar off, 5 with lidar on), order
  ``[vision, ultrasonic, motor, audio, (lidar)]``, with the vision slot ``0``
  when ``vision_features`` is empty (or vision is encoder-disabled);
* :func:`encode_seed_states` rolls ``observe_step`` from a zero ``(h, z)`` across
  the records (``prev_action`` for step ``t`` = ``records[t-1].action``, zeros at
  ``t0``), is DETERMINISTIC given fixed records, places tensors on the requested
  ``device``, runs under ``eval()`` + ``no_grad`` (no grad leak, no train-mode
  side effect, base model untouched), and falls back to an empty list with a
  structured warning when given no records.
"""

from __future__ import annotations

import numpy as np
import torch

from mousedroid.config.schema import ModelConfig
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.learning.on_device.seed_states import (
    encode_seed_states,
    record_to_observation,
)
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


# ---------------------------------------------------------------------------
# encode_seed_states — determinism, device, eval/no_grad, fallback
# ---------------------------------------------------------------------------


def test_encode_seed_states_returns_h_z_pairs() -> None:
    """Encoding produces ``(h, z)`` seed-state pairs of the right shape."""
    wm = _make_world_model()
    records = _make_records(5)

    states = encode_seed_states(wm, records, n_seed=3, device=torch.device("cpu"))

    assert len(states) == 3
    for h, z in states:
        assert h.shape == (1, wm.cfg.hidden_dim)
        assert z.shape == (1, wm.cfg.latent_dim)


def test_encode_seed_states_caps_at_n_seed() -> None:
    """At most ``n_seed`` states are returned even with more records."""
    wm = _make_world_model()
    records = _make_records(10)

    states = encode_seed_states(wm, records, n_seed=4, device=torch.device("cpu"))

    assert len(states) == 4


def test_encode_seed_states_returns_all_when_fewer_records() -> None:
    """When records < n_seed, returns one state per record."""
    wm = _make_world_model()
    records = _make_records(2)

    states = encode_seed_states(wm, records, n_seed=10, device=torch.device("cpu"))

    assert len(states) == 2


def test_encode_seed_states_deterministic() -> None:
    """Same model + same records -> byte-identical seed states."""
    wm = _make_world_model()
    records = _make_records(5)

    a = encode_seed_states(wm, records, n_seed=3, device=torch.device("cpu"))
    b = encode_seed_states(wm, records, n_seed=3, device=torch.device("cpu"))

    assert len(a) == len(b)
    for (ha, za), (hb, zb) in zip(a, b, strict=True):
        assert torch.equal(ha, hb)
        assert torch.equal(za, zb)


def test_encode_seed_states_on_device() -> None:
    """Returned tensors live on the requested device."""
    wm = _make_world_model()
    records = _make_records(3)
    device = torch.device("cpu")

    states = encode_seed_states(wm, records, n_seed=3, device=device)

    for h, z in states:
        assert h.device == device
        assert z.device == device


def test_encode_seed_states_no_grad_and_eval_preserved() -> None:
    """Encoding leaks no autograd graph and restores the model's train mode."""
    wm = _make_world_model()
    wm.train()  # deliberately in train mode
    records = _make_records(3)

    states = encode_seed_states(wm, records, n_seed=3, device=torch.device("cpu"))

    # No grad graph attached to the returned tensors.
    for h, z in states:
        assert not h.requires_grad
        assert not z.requires_grad
    # Train-mode is restored (no eval() side effect leaks out).
    assert wm.training is True


def test_encode_seed_states_base_model_untouched() -> None:
    """Encoding never mutates the world-model parameters in place."""
    wm = _make_world_model()
    before = {k: v.clone() for k, v in wm.named_parameters()}
    records = _make_records(4)

    encode_seed_states(wm, records, n_seed=2, device=torch.device("cpu"))

    for k, v in wm.named_parameters():
        assert torch.equal(before[k], v), f"param {k} mutated"


def test_encode_seed_states_empty_returns_fallback_with_warning() -> None:
    """An empty record slice returns an empty list + a structured warning."""
    import structlog

    wm = _make_world_model()

    with structlog.testing.capture_logs() as captured:
        states = encode_seed_states(wm, [], n_seed=3, device=torch.device("cpu"))

    assert states == []
    events = [entry.get("event", "") for entry in captured]
    assert "on_device_seed_states_empty_replay" in events


def test_encode_seed_states_lidar_encoder_mask_does_not_indexerror() -> None:
    """A lidar-enabled encoder uses a 5-slot mask without IndexError."""
    wm = _make_world_model(lidar=True)
    records = _make_records(3)

    states = encode_seed_states(wm, records, n_seed=2, device=torch.device("cpu"))

    assert len(states) == 2
