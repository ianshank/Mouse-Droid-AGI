"""Off-loop growth-pillar coordinator — VLA knowledge distillation.

Wires the growth pillar (:class:`~mousedroid.growth.distillation.KnowledgeDistiller`)
into the runtime as a **default-OFF, slow-cadence** background operation, mirroring
the Phase-6 on-device
:class:`~mousedroid.learning.on_device.replay_trigger.ReplayTriggerCoordinator`:

- The 30 Hz reactive loop stays distillation-free. This coordinator runs on a
  dedicated slow-cadence background task and offloads EVERY torch / blocking op
  via :func:`asyncio.to_thread`, so no gradient work ever touches the event loop.
- It distils the wired VLA teacher policy into a compact
  :class:`~mousedroid.growth.student.StudentVLAPolicy` over real on-policy latents
  and persists the student to a SHA-256-stamped slot. It NEVER hot-swaps the
  student into the live policy — deploying a distilled student is a soak-gated
  operator decision (exactly like on-device WS4 promotion).
- The trigger arms on NEW experience records (an injected counter), so the loop
  disarms until fresh experience accumulates instead of re-distilling stale data.

Collaborators are injected as plain callables so tests exercise the whole cycle
with fakes and no hardware / world model.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import GrowthConfig
    from mousedroid.growth.distillation import KnowledgeDistiller
    from mousedroid.growth.slot_store import GrowthSlotStore, StudentSlot
    from mousedroid.growth.student import StudentVLAPolicy
    from mousedroid.telemetry.metrics import MetricsRegistry

_log = get_logger(__name__)

#: Injected sampler: returns one ``(h_batch, z_batch)`` latent minibatch, or
#: ``None`` when no observations are available (e.g. the world model has not
#: produced any latents yet). ``h_batch`` is ``[B, h_dim]``, ``z_batch`` is
#: ``[B, z_dim]``.
SampleBatchFn = Callable[[], "tuple[Tensor, Tensor] | None"]


class GrowthDistillationCoordinator:
    """Slow-cadence VLA-teacher -> compact-student knowledge distillation.

    Args:
        cfg: The growth config block (trigger, step count, etc.).
        distiller: A regression-objective :class:`KnowledgeDistiller` whose
            teacher wraps the live VLA policy and whose student is ``student``.
        student: The compact student being grown (its state-dict is persisted).
        sample_batch: Injected latent-minibatch sampler (see :data:`SampleBatchFn`).
        slot_store: SHA-256 slot store for the distilled student.
        count_new_records: Returns the number of new experience records since the
            last consumed baseline (the trigger signal).
        on_consumed: Called with the consumed count after a fired cycle so the
            trigger disarms until fresh experience accumulates. ``None`` = no-op.
        metrics: Optional registry for the gated distillation counter.
    """

    def __init__(
        self,
        *,
        cfg: GrowthConfig,
        distiller: KnowledgeDistiller,
        student: StudentVLAPolicy,
        sample_batch: SampleBatchFn,
        slot_store: GrowthSlotStore,
        count_new_records: Callable[[], int],
        on_consumed: Callable[[int], None] | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._cfg = cfg
        self._distiller = distiller
        self._student = student
        self._sample_batch = sample_batch
        self._slot_store = slot_store
        self._count_new_records = count_new_records
        self._on_consumed = on_consumed
        self._metrics = metrics

    async def maybe_distill(self) -> StudentSlot | None:
        """Run one bounded distillation cycle if the trigger is armed.

        Returns:
            The persisted :class:`StudentSlot`, or ``None`` when the trigger is
            below threshold or no latent batch was available.
        """
        # Offload the trigger probe (may hit blocking storage).
        new_records = await asyncio.to_thread(self._count_new_records)
        threshold = self._cfg.trigger_min_new_records
        if new_records < threshold:
            _log.debug(
                "growth_trigger_below_threshold",
                new_records=new_records,
                threshold=threshold,
            )
            return None

        _log.info(
            "growth_distill_start",
            new_records=new_records,
            distill_steps=self._cfg.distill_steps,
        )

        # Offload the whole bounded distillation epoch to a worker thread — all
        # torch autograd + world-model sampling happens off the event loop.
        losses = await asyncio.to_thread(self._run_distill_epoch)
        if not losses:
            _log.warning("growth_distill_skipped_no_batch", new_records=new_records)
            if self._metrics is not None:
                self._metrics.inc_growth_distilled("skipped_no_batch")
            return None

        # Offload slot persistence (serialise + SHA-256 stamp).
        slot = await asyncio.to_thread(self._slot_store.persist, self._student_state_dict())
        _log.info(
            "growth_distill_complete",
            digest=slot.digest,
            n_steps=len(losses),
            final_loss=losses[-1],
        )

        if self._on_consumed is not None:
            self._on_consumed(new_records)
        if self._metrics is not None:
            self._metrics.inc_growth_distilled("completed")
        return slot

    def _run_distill_epoch(self) -> list[float]:
        """Run up to ``distill_steps`` regression distill steps (sync, in-thread).

        Returns:
            Per-step scalar losses. Empty when the first sampled batch is ``None``
            (no latents available) — the caller treats that as a skip.
        """
        losses: list[float] = []
        for _ in range(self._cfg.distill_steps):
            batch = self._sample_batch()
            if batch is None:
                break
            h, z = batch
            x = torch.cat([h, z], dim=-1)
            # Regression self-distillation: student matches the frozen VLA
            # teacher's continuous action (no ground-truth hard labels).
            loss = self._distiller.distill_step(x)
            losses.append(float(loss.item()))
        return losses

    def _student_state_dict(self) -> dict[str, Tensor]:
        """Detached, cloned snapshot of the student parameters for persistence."""
        return {name: param.detach().clone() for name, param in self._student.state_dict().items()}


__all__ = ["GrowthDistillationCoordinator", "SampleBatchFn"]
