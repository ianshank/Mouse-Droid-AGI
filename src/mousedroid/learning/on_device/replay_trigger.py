"""Replay-triggered on-device update coordinator (Phase 6 WS3).

Sits at the slow-cadence seam (mirrors the orchestrator's
``_consolidation_loop`` background task): once enough fresh experience records
have accumulated it reads a batch, runs the WS2 bounded learner ``update()``
OFF the event loop (``asyncio.to_thread``), and persists the resulting
candidate via the SHA-256-stamped :class:`OnDeviceSlotStore`.

The coordinator PRODUCES + PERSISTS a stamped candidate slot. It does NOT
activate / swap it into the live policy — promotion is WS4's safety-gated
decision. The 30 Hz reactive control loop is never touched: the only torch
work runs on a worker thread, and the trigger check + batch read are cheap
async operations on the slow cadence.

Collaborators are injected as plain callables so the same coordinator drives
both the real LMDB replay reader (wired by the factory/orchestrator) and the
fakes used by unit / integration tests:

* ``count_new_records() -> int`` — fresh-record count since the last consumed
  baseline (the trigger source; wired to a reader cursor/marker upstream);
* ``load_batch() -> Tensor`` — materialise one training batch tensor;
* ``on_consumed(count)`` — optional callback fired AFTER a successful cycle so
  the caller can advance its new-record marker to the consumed baseline.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from torch import Tensor

    from mousedroid.config.schema import OnDeviceLearningConfig
    from mousedroid.learning.on_device.protocol import OnDeviceLearner
    from mousedroid.learning.on_device.slot_store import CandidateSlot, OnDeviceSlotStore

_log = get_logger(__name__)


class ReplayTriggerCoordinator:
    """Replay-triggered, slow-cadence on-device update coordinator.

    Args:
        cfg: On-device-learning config (``trigger_min_new_records`` gates the
            cycle; ``update_steps`` is logged for operator visibility).
        learner: WS2 :class:`OnDeviceLearner` producing the candidate.
        slot_store: SHA-256-stamped candidate slot store.
        count_new_records: Returns fresh-record count since the last baseline.
        load_batch: Materialises one training batch tensor for the learner.
        on_consumed: Optional callback invoked with the consumed record count
            AFTER a successful persist so the caller advances its marker.
    """

    def __init__(
        self,
        *,
        cfg: OnDeviceLearningConfig,
        learner: OnDeviceLearner,
        slot_store: OnDeviceSlotStore,
        count_new_records: Callable[[], int],
        load_batch: Callable[[], Tensor],
        on_consumed: Callable[[int], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._learner = learner
        self._slot_store = slot_store
        self._count_new_records = count_new_records
        self._load_batch = load_batch
        self._on_consumed = on_consumed

    async def maybe_update(self) -> CandidateSlot | None:
        """Run one trigger check; produce + persist a candidate if armed.

        Returns:
            The persisted :class:`CandidateSlot` when a cycle fired, else
            ``None`` (below threshold, or an empty batch).
        """
        new_records = self._count_new_records()
        threshold = self._cfg.trigger_min_new_records
        if new_records < threshold:
            _log.debug(
                "on_device_trigger_below_threshold",
                new_records=new_records,
                threshold=threshold,
            )
            return None

        _log.info(
            "on_device_trigger_fired",
            new_records=new_records,
            threshold=threshold,
            update_steps=self._cfg.update_steps,
        )

        batch = self._load_batch()
        if batch.shape[0] == 0:
            _log.warning("on_device_trigger_empty_batch", new_records=new_records)
            return None

        # Offload the bounded torch update to a worker thread so the event
        # loop — and therefore the 30 Hz reactive control loop sharing it —
        # is NEVER blocked by gradient work.
        result = await asyncio.to_thread(self._learner.update, batch)

        _log.info(
            "on_device_candidate_produced",
            n_steps=result.n_steps,
            train_loss=result.train_loss,
            batch_size=int(batch.shape[0]),
        )

        slot = self._slot_store.persist(result.candidate_state_dict)
        _log.info(
            "on_device_candidate_persisted",
            digest=slot.digest,
            path=str(slot.path),
            consumed_records=new_records,
        )

        if self._on_consumed is not None:
            self._on_consumed(new_records)
        return slot


__all__ = ["ReplayTriggerCoordinator"]
