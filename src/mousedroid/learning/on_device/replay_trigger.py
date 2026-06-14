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
* ``load_batch() -> Tensor | Mapping[str, Tensor]`` — materialise one training
  batch. The #134 :class:`~mousedroid.learning.on_device.protocol.OnDeviceLearner`
  path returns a ``(n, input_dim)`` :class:`~torch.Tensor`; the WS-E2
  RSSM-refinement path returns a ``(B, T, ...)`` sequence dict consumed by
  :class:`~mousedroid.learning.on_device.protocol.RSSMSequenceLearner`. The
  coordinator flows whichever shape ``load_batch`` produces straight into
  ``learner.update`` — both are offloaded identically;
* ``batch_is_empty(batch) -> bool`` — optional empty-check override. The default
  :func:`_default_batch_is_empty` handles BOTH shapes (a ``Tensor`` via
  ``shape[0]`` and a ``Mapping`` via its representative ``"motor"`` key — or any
  first value when ``"motor"`` is absent), so the #134 Tensor callers stay
  byte-identical without supplying it;
* ``on_consumed(count)`` — optional callback fired AFTER a successful cycle so
  the caller can advance its new-record marker to the consumed baseline.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from torch import Tensor

    from mousedroid.config.schema import OnDeviceLearningConfig
    from mousedroid.learning.on_device.protocol import (
        OnDeviceLearner,
        OnDeviceUpdateResult,
        RSSMSequenceLearner,
    )
    from mousedroid.learning.on_device.slot_store import CandidateSlot, OnDeviceSlotStore

    # Either the #134 Tensor batch or the WS-E2 (B, T, ...) sequence dict.
    Batch = Tensor | Mapping[str, Tensor]
    # The bounded update both learner protocols expose (differing only in the
    # batch arg type they accept — unified here at the coordinator's call site).
    UpdateFn = Callable[[Batch], OnDeviceUpdateResult]

_log = get_logger(__name__)


def _batch_size(batch: Batch) -> int:
    """Return the leading (batch) dimension of a Tensor or sequence-dict batch.

    Args:
        batch: A ``(n, input_dim)`` :class:`~torch.Tensor` (#134 path) or a
            ``(B, T, ...)`` sequence dict (WS-E2 path).

    Returns:
        The leading dimension — ``batch.shape[0]`` for a Tensor, or the leading
        dimension of the representative ``"motor"`` value (falling back to the
        first value) for a Mapping.
    """
    if isinstance(batch, Mapping):
        representative = batch.get("motor")
        if representative is None:
            representative = next(iter(batch.values()))
        return int(representative.shape[0])
    return int(batch.shape[0])


def _default_batch_is_empty(batch: Batch) -> bool:
    """Default empty-check for a Tensor (#134) or sequence-dict (WS-E2) batch.

    An empty :class:`~torch.Tensor` (``shape[0] == 0``) or an empty Mapping (no
    keys) is empty; otherwise the batch is empty iff its leading dimension is 0.

    Args:
        batch: The materialised batch to test.

    Returns:
        ``True`` when the batch carries no samples.
    """
    if isinstance(batch, Mapping) and not batch:
        return True
    return _batch_size(batch) == 0


class ReplayTriggerCoordinator:
    """Replay-triggered, slow-cadence on-device update coordinator.

    Args:
        cfg: On-device-learning config (``trigger_min_new_records`` gates the
            cycle; ``update_steps`` is logged for operator visibility).
        learner: The bounded learner producing the candidate — either the #134
            Tensor-path :class:`OnDeviceLearner` or the WS-E2
            :class:`RSSMSequenceLearner` consuming a ``(B, T, ...)`` sequence
            dict. The coordinator is agnostic: it offloads ``learner.update`` on
            whatever ``load_batch`` produces.
        slot_store: SHA-256-stamped candidate slot store.
        count_new_records: Returns fresh-record count since the last baseline.
        load_batch: Materialises one training batch for the learner — a
            ``(n, input_dim)`` Tensor (#134) OR a ``(B, T, ...)`` sequence dict
            (WS-E2). The coordinator never inspects the shape beyond the
            empty-check + batch-size log.
        batch_is_empty: Optional override of the empty-batch check. ``None`` (the
            default) uses :func:`_default_batch_is_empty`, which handles BOTH the
            Tensor and the sequence-dict shapes — so the #134 callers keep their
            exact ``shape[0] == 0`` skip semantics without supplying it.
        on_consumed: Optional callback invoked with the consumed record count
            AFTER a successful persist so the caller advances its marker.
        gate_runner: Optional WS4 safety-regression gate callback invoked with
            the persisted candidate slot AFTER persist (and after ``on_consumed``).
            It scores the candidate vs the live baseline and promotes-or-reverts
            (marking the slot active on pass, incrementing the revert counter on
            fail). When ``None`` (the default, and whenever the gate is disabled)
            NO scoring runs and the cycle is byte-identical to pre-WS4. The
            gate's torch scoring is offloaded off the event loop with the same
            ``asyncio.to_thread`` discipline as the learner update so the 30 Hz
            hot loop is never blocked.
    """

    def __init__(
        self,
        *,
        cfg: OnDeviceLearningConfig,
        learner: OnDeviceLearner | RSSMSequenceLearner,
        slot_store: OnDeviceSlotStore,
        count_new_records: Callable[[], int],
        load_batch: Callable[[], Batch],
        batch_is_empty: Callable[[Batch], bool] | None = None,
        on_consumed: Callable[[int], None] | None = None,
        gate_runner: Callable[[CandidateSlot], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._learner = learner
        self._slot_store = slot_store
        self._count_new_records = count_new_records
        self._load_batch = load_batch
        self._batch_is_empty = batch_is_empty or _default_batch_is_empty
        self._on_consumed = on_consumed
        self._gate_runner = gate_runner

    async def maybe_update(self) -> CandidateSlot | None:
        """Run one trigger check; produce + persist a candidate if armed.

        Returns:
            The persisted :class:`CandidateSlot` when a cycle fired, else
            ``None`` (below threshold, or an empty batch).
        """
        # Offload the trigger probe: the wired collaborator may run blocking
        # LMDB I/O, so it must never execute inline on the event-loop thread
        # (mirrors the learner-update offload below).
        new_records = await asyncio.to_thread(self._count_new_records)
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

        # Offload the batch materialisation for the same reason — it reads the
        # replay store off the event loop. ``batch`` is a Tensor (#134) or a
        # (B, T, ...) sequence dict (WS-E2); the empty-check + batch-size helpers
        # are shape-agnostic so a dict batch never IndexErrors a missing key.
        batch = await asyncio.to_thread(self._load_batch)
        if self._batch_is_empty(batch):
            _log.warning("on_device_trigger_empty_batch", new_records=new_records)
            return None

        # Offload the bounded torch update to a worker thread so the event
        # loop — and therefore the 30 Hz reactive control loop sharing it —
        # is NEVER blocked by gradient work. The dict batch flows straight
        # through to the RSSM refiner's ``update`` here. ``self._learner`` is a
        # union of two protocols whose ``update`` differs ONLY in the batch arg
        # type; the runtime object is a single concrete learner that accepts the
        # batch ``load_batch`` produced, so the bound method is unified to the
        # ``Batch``-accepting ``UpdateFn`` (a cast, not a suppression).
        update_fn = cast("UpdateFn", self._learner.update)
        result = await asyncio.to_thread(update_fn, batch)

        _log.info(
            "on_device_candidate_produced",
            n_steps=result.n_steps,
            train_loss=result.train_loss,
            batch_size=_batch_size(batch),
        )

        # Offload the slot persistence too — it serialises the state dict to
        # disk and streams a SHA-256 over it, both blocking syscalls that must
        # not run on the event loop (CodeRabbit review).
        slot = await asyncio.to_thread(self._slot_store.persist, result.candidate_state_dict)
        _log.info(
            "on_device_candidate_persisted",
            digest=slot.digest,
            path=str(slot.path),
            consumed_records=new_records,
        )

        if self._on_consumed is not None:
            self._on_consumed(new_records)

        # WS4 safety-regression gate: score the candidate vs the live baseline
        # and promote-or-revert. Offloaded off the event loop (the gate runs the
        # torch rollout-return scoring) so the 30 Hz reactive loop is untouched.
        # Default-OFF: with no gate wired this is byte-identical to pre-WS4.
        if self._gate_runner is not None:
            await asyncio.to_thread(self._gate_runner, slot)

        return slot


__all__ = ["ReplayTriggerCoordinator"]
