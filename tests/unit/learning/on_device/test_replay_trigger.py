"""Unit tests for the replay-triggered on-device update coordinator (WS3).

Pins the WS3 coordination contract:

* an update cycle fires only when new-record count >= ``trigger_min_new_records``;
* below threshold is a no-op (no learner call, no slot persisted);
* the torch ``update()`` work is offloaded off the event loop (proved by
  driving the coordinator from a running loop and asserting it never blocks —
  the learner records the thread it ran on and it is NOT the loop thread);
* a fired cycle persists a stamped candidate slot and resets the new-record
  marker so the next cycle counts from the new baseline.

Plus the WS-E2 generalization: the coordinator is shape-agnostic — it flows a
``(B, T, ...)`` sequence-DICT batch to an
:class:`~mousedroid.learning.on_device.protocol.RSSMSequenceLearner` exactly as
it flows the #134 Tensor batch to an
:class:`~mousedroid.learning.on_device.protocol.OnDeviceLearner`, with a
dict-aware empty-check (empty ``{}`` OR a zero-length representative key) and an
injectable ``batch_is_empty`` override.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import torch
import torch.nn as nn

from mousedroid.config.schema import ExperienceConfig, OnDeviceLearningConfig
from mousedroid.learning.on_device.ewc_online import EWCOnlineLearner
from mousedroid.learning.on_device.protocol import OnDeviceUpdateResult
from mousedroid.learning.on_device.replay_trigger import ReplayTriggerCoordinator
from mousedroid.learning.on_device.slot_store import OnDeviceSlotStore

_INPUT_DIM = 4


def _make_store(tmp_path: Path, on_device: OnDeviceLearningConfig) -> OnDeviceSlotStore:
    experience = ExperienceConfig(path=str(tmp_path / "experience_root"))
    return OnDeviceSlotStore(experience_cfg=experience, on_device_cfg=on_device)


def _make_learner(cfg: OnDeviceLearningConfig) -> EWCOnlineLearner:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(_INPUT_DIM, 3))
    return EWCOnlineLearner(cfg, model)


def _batch_provider(n: int = 8) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(n, _INPUT_DIM)


def test_below_threshold_is_noop(tmp_path: Path) -> None:
    """No update fires while new records < trigger threshold."""
    cfg = OnDeviceLearningConfig(enabled=True, trigger_min_new_records=500, update_steps=1)
    coordinator = ReplayTriggerCoordinator(
        cfg=cfg,
        learner=_make_learner(cfg),
        slot_store=_make_store(tmp_path, cfg),
        count_new_records=lambda: 10,
        load_batch=_batch_provider,
    )

    fired = asyncio.run(coordinator.maybe_update())

    assert fired is None
    # No slot persisted.
    assert not list((tmp_path / "experience_root" / "on_device_slot").glob("*.pt"))


def test_fires_at_threshold_and_persists_slot(tmp_path: Path) -> None:
    """At >= threshold, an update fires and a stamped slot is persisted."""
    cfg = OnDeviceLearningConfig(enabled=True, trigger_min_new_records=5, update_steps=2)
    coordinator = ReplayTriggerCoordinator(
        cfg=cfg,
        learner=_make_learner(cfg),
        slot_store=_make_store(tmp_path, cfg),
        count_new_records=lambda: 5,
        load_batch=_batch_provider,
    )

    slot = asyncio.run(coordinator.maybe_update())

    assert slot is not None
    assert slot.path.is_file()
    assert len(slot.digest) == 64


def test_torch_update_is_offloaded_off_the_event_loop(tmp_path: Path) -> None:
    """The learner's ``update`` runs on a worker thread, not the loop thread."""
    cfg = OnDeviceLearningConfig(enabled=True, trigger_min_new_records=1, update_steps=1)

    update_thread: dict[str, int] = {}

    class _ThreadRecordingLearner:
        def update(self, batch: torch.Tensor) -> OnDeviceUpdateResult:
            update_thread["tid"] = threading.get_ident()
            return OnDeviceUpdateResult(
                candidate_state_dict={"w": torch.zeros(2)},
                train_loss=0.0,
                n_steps=cfg.update_steps,
            )

    async def _drive() -> None:
        loop_tid = threading.get_ident()
        coordinator = ReplayTriggerCoordinator(
            cfg=cfg,
            learner=_ThreadRecordingLearner(),
            slot_store=_make_store(tmp_path, cfg),
            count_new_records=lambda: 1,
            load_batch=_batch_provider,
        )
        await coordinator.maybe_update()
        assert update_thread["tid"] != loop_tid

    asyncio.run(_drive())


def test_replay_collaborators_offloaded_off_the_event_loop(tmp_path: Path) -> None:
    """``count_new_records`` and ``load_batch`` run off the loop thread.

    Both collaborators may perform blocking LMDB I/O (the WS3 factory wiring
    drives the async reader via a private-loop worker), so they must be invoked
    through ``asyncio.to_thread`` — never inline on the event loop thread.
    """
    cfg = OnDeviceLearningConfig(enabled=True, trigger_min_new_records=1, update_steps=1)
    observed: dict[str, int] = {}

    async def _drive() -> None:
        loop_tid = threading.get_ident()

        def _count() -> int:
            observed["count_tid"] = threading.get_ident()
            return 1

        def _load() -> torch.Tensor:
            observed["load_tid"] = threading.get_ident()
            return _batch_provider()

        coordinator = ReplayTriggerCoordinator(
            cfg=cfg,
            learner=_make_learner(cfg),
            slot_store=_make_store(tmp_path, cfg),
            count_new_records=_count,
            load_batch=_load,
        )
        await coordinator.maybe_update()
        assert observed["count_tid"] != loop_tid
        assert observed["load_tid"] != loop_tid

    asyncio.run(_drive())


def test_marker_resets_after_fire(tmp_path: Path) -> None:
    """After firing, the coordinator notifies the consume callback once."""
    cfg = OnDeviceLearningConfig(enabled=True, trigger_min_new_records=3, update_steps=1)
    consumed: list[int] = []
    coordinator = ReplayTriggerCoordinator(
        cfg=cfg,
        learner=_make_learner(cfg),
        slot_store=_make_store(tmp_path, cfg),
        count_new_records=lambda: 7,
        load_batch=_batch_provider,
        on_consumed=consumed.append,
    )

    asyncio.run(coordinator.maybe_update())

    assert consumed == [7]


def test_gate_runner_invoked_with_persisted_slot(tmp_path: Path) -> None:
    """A wired ``gate_runner`` is called with the persisted slot after persist."""
    cfg = OnDeviceLearningConfig(enabled=True, trigger_min_new_records=1, update_steps=1)
    seen: list[str] = []

    def _gate(slot: object) -> None:
        seen.append(slot.digest)  # type: ignore[attr-defined]

    coordinator = ReplayTriggerCoordinator(
        cfg=cfg,
        learner=_make_learner(cfg),
        slot_store=_make_store(tmp_path, cfg),
        count_new_records=lambda: 1,
        load_batch=_batch_provider,
        gate_runner=_gate,
    )

    slot = asyncio.run(coordinator.maybe_update())

    assert slot is not None
    assert seen == [slot.digest]


def test_no_gate_runner_is_byte_identical(tmp_path: Path) -> None:
    """Omitting ``gate_runner`` still persists a slot (pre-WS4 byte-identical)."""
    cfg = OnDeviceLearningConfig(enabled=True, trigger_min_new_records=1, update_steps=1)
    coordinator = ReplayTriggerCoordinator(
        cfg=cfg,
        learner=_make_learner(cfg),
        slot_store=_make_store(tmp_path, cfg),
        count_new_records=lambda: 1,
        load_batch=_batch_provider,
    )

    slot = asyncio.run(coordinator.maybe_update())

    assert slot is not None
    assert slot.path.is_file()


def test_empty_batch_skips_update(tmp_path: Path) -> None:
    """A threshold-met trigger with an empty batch is a safe no-op."""
    cfg = OnDeviceLearningConfig(enabled=True, trigger_min_new_records=1, update_steps=1)

    def _empty_batch() -> torch.Tensor:
        return torch.empty(0, _INPUT_DIM)

    coordinator = ReplayTriggerCoordinator(
        cfg=cfg,
        learner=_make_learner(cfg),
        slot_store=_make_store(tmp_path, cfg),
        count_new_records=lambda: 100,
        load_batch=_empty_batch,
    )

    slot = asyncio.run(coordinator.maybe_update())

    assert slot is None


# --------------------------------------------------------------------------- #
# WS-E2: the coordinator is shape-agnostic — a (B, T, ...) sequence DICT batch
# flows through the RSSM-refiner path identically to the #134 Tensor path.
# --------------------------------------------------------------------------- #
def _seq_batch(b: int = 2, t: int = 3) -> dict[str, torch.Tensor]:
    """A minimal (B, T, ...) sequence-dict batch (only the keys the test needs)."""
    return {
        "motor": torch.zeros(b, t, 4),
        "action": torch.zeros(b, t, 3),
        "valid_mask": torch.zeros(b, t, 4),
    }


class _DictRecordingLearner:
    """An :class:`RSSMSequenceLearner`-shaped learner recording its batch arg."""

    def __init__(self) -> None:
        self.seen: list[dict[str, torch.Tensor]] = []

    def update(self, batch: dict[str, torch.Tensor]) -> OnDeviceUpdateResult:
        self.seen.append(batch)
        return OnDeviceUpdateResult(
            candidate_state_dict={"w": torch.zeros(2)}, train_loss=0.5, n_steps=1
        )


def test_dict_batch_flows_through_to_learner(tmp_path: Path) -> None:
    """A sequence-dict batch is forwarded verbatim to ``learner.update``."""
    cfg = OnDeviceLearningConfig(enabled=True, trigger_min_new_records=1, update_steps=1)
    learner = _DictRecordingLearner()
    batch = _seq_batch()

    coordinator = ReplayTriggerCoordinator(
        cfg=cfg,
        learner=learner,
        slot_store=_make_store(tmp_path, cfg),
        count_new_records=lambda: 1,
        load_batch=lambda: batch,
    )

    slot = asyncio.run(coordinator.maybe_update())

    assert slot is not None
    assert learner.seen == [batch]


def test_empty_dict_batch_skips_update(tmp_path: Path) -> None:
    """An empty sequence-dict ({}) is treated as an empty batch (safe no-op)."""
    cfg = OnDeviceLearningConfig(enabled=True, trigger_min_new_records=1, update_steps=1)
    learner = _DictRecordingLearner()

    coordinator = ReplayTriggerCoordinator(
        cfg=cfg,
        learner=learner,
        slot_store=_make_store(tmp_path, cfg),
        count_new_records=lambda: 100,
        load_batch=dict,
    )

    slot = asyncio.run(coordinator.maybe_update())

    assert slot is None
    assert learner.seen == []


def test_zero_length_dict_batch_skips_update(tmp_path: Path) -> None:
    """A dict whose representative ``motor`` key has B == 0 is empty (no-op)."""
    cfg = OnDeviceLearningConfig(enabled=True, trigger_min_new_records=1, update_steps=1)
    learner = _DictRecordingLearner()

    def _zero_len() -> dict[str, torch.Tensor]:
        return {"motor": torch.zeros(0, 3, 4), "action": torch.zeros(0, 3, 3)}

    coordinator = ReplayTriggerCoordinator(
        cfg=cfg,
        learner=learner,
        slot_store=_make_store(tmp_path, cfg),
        count_new_records=lambda: 100,
        load_batch=_zero_len,
    )

    slot = asyncio.run(coordinator.maybe_update())

    assert slot is None
    assert learner.seen == []


def test_dict_batch_without_motor_key_uses_first_value_for_size(tmp_path: Path) -> None:
    """A non-empty dict lacking ``motor`` falls back to its first value's leading dim.

    Exercises the ``_batch_size`` fallback branch (``next(iter(values))``) so a
    custom sequence-dict that does not key on ``motor`` is still sized — and
    therefore neither false-empty-skipped nor IndexError'd.
    """
    cfg = OnDeviceLearningConfig(enabled=True, trigger_min_new_records=1, update_steps=1)
    learner = _DictRecordingLearner()
    batch = {"observation": torch.zeros(2, 3, 4)}

    coordinator = ReplayTriggerCoordinator(
        cfg=cfg,
        learner=learner,
        slot_store=_make_store(tmp_path, cfg),
        count_new_records=lambda: 1,
        load_batch=lambda: batch,
    )

    slot = asyncio.run(coordinator.maybe_update())

    assert slot is not None
    assert learner.seen == [batch]


def test_injected_batch_is_empty_override(tmp_path: Path) -> None:
    """A custom ``batch_is_empty`` override gates the cycle (here: always empty)."""
    cfg = OnDeviceLearningConfig(enabled=True, trigger_min_new_records=1, update_steps=1)
    learner = _DictRecordingLearner()

    coordinator = ReplayTriggerCoordinator(
        cfg=cfg,
        learner=learner,
        slot_store=_make_store(tmp_path, cfg),
        count_new_records=lambda: 100,
        load_batch=_seq_batch,
        batch_is_empty=lambda _batch: True,
    )

    slot = asyncio.run(coordinator.maybe_update())

    assert slot is None
    assert learner.seen == []
