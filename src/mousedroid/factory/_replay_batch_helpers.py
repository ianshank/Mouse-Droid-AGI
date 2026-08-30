"""Private replay-batch plumbing shared by on_device_learning.py and growth.py.

Not part of the public factory surface.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, TypeVar

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    import torch
    from torch import Tensor

    from mousedroid.config.schema import (
        ModelConfig,
        Settings,
    )
    from mousedroid.experience.record import MouseDroidExperienceRecord
    from mousedroid.training.replay.lmdb_reader import LMDBReplayReader
    from mousedroid.world_model.encoder import MultimodalEncoder

_log = get_logger(__name__)
_CoroResult = TypeVar("_CoroResult")
_MAX_REPLAY_COUNT_CHUNK: int = 1000


def _make_consumed_offset_advancer(
    consumed_offset: list[int], *, log_event: str
) -> Callable[[int], None]:
    """Build an ``on_consumed`` callback advancing a shared consumed-offset cell.

    Both slow-cadence coordinator builders (on-device learning, growth
    distillation) use an identical in-memory "consumed beyond this baseline"
    trigger-disarm pattern, differing only in which structured-log event name
    they emit — preserved here via ``log_event`` so neither
    ``on_device_consumed_offset_advanced`` nor ``growth_consumed_offset_advanced``
    (both referenced by CLAUDE.md / operator grep runbooks) changes.
    """

    def _advance_consumed(n_new: int) -> None:
        consumed_offset[0] += n_new
        _log.debug(log_event, consumed_total=consumed_offset[0], advanced_by=n_new)

    return _advance_consumed


def _build_shared_replay_reader(cfg: Settings) -> LMDBReplayReader:
    """Construct the replay reader shared by both slow-cadence coordinators.

    Byte-identical across ``build_on_device_coordinator`` and
    ``build_growth_coordinator`` — both honour any
    ``cfg.training.replay.source_path`` override and the shared debug-log
    cadence. Deliberately NOT reused by the main replay-reader builder
    (~line 1393), which additionally threads ``metrics=metrics``: a third
    caller needing a parameter only it uses would repeat the exact "leaky
    kwargs" asymmetry ADR-014 already rejected for a similar merge.
    """
    from mousedroid.training.replay.lmdb_reader import LMDBReplayReader

    return LMDBReplayReader(
        cfg.experience,
        path_override=cfg.training.replay.source_path,
        debug_log_every_n=cfg.training.replay_mixer.debug_log_every_n,
    )


def _run_coro_blocking(coro: Coroutine[Any, Any, _CoroResult]) -> _CoroResult:
    """Drive ``coro`` to completion on a private loop in a dedicated thread.

    The WS3 coordinator's ``count_new_records`` / ``load_batch`` collaborators
    are SYNC callables, but the LMDB reader is async. ``asyncio.run`` cannot be
    called when a loop is already running (the orchestrator drives the
    coordinator inside the event loop), so we always run the coroutine on a
    fresh loop in a worker thread. That also keeps the (slow-cadence) blocking
    LMDB scan off the orchestrator's event-loop thread, preserving the hot-loop
    isolation contract.

    Args:
        coro: The coroutine to run to completion.

    Returns:
        The coroutine's result.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    def _worker() -> _CoroResult:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_worker).result()


def _count_replay_records(reader: LMDBReplayReader, cap: int) -> int:
    """Count up to ``cap`` replay records via the async reader (sync wrapper).

    Args:
        reader: The LMDB replay reader.
        cap: Stop counting once this many records are seen.

    Returns:
        The number of records read (capped at ``cap``).
    """

    async def _run() -> int:
        seen = 0
        async for chunk in reader.stream(chunk_size=cap):
            seen += len(chunk)
            if seen >= cap:
                break
        return seen

    return _run_coro_blocking(_run())


def _count_new_replay_records(reader: LMDBReplayReader, *, consumed: int, cap: int) -> int:
    """Count replay records BEYOND a consumed baseline (sync wrapper).

    The on-device trigger must arm on NEW experience — records that have arrived
    since the last fired cycle drained the store — not on the absolute store size.
    Counting the total re-fires the refine + gate every cadence on already-
    consumed (stale) data because the count never drops below the threshold.

    Streams chronologically, skips the first ``consumed`` records, and counts up
    to ``cap`` records after them (so the NEW window is bounded the same way the
    refine batch scan is). Returns ``0`` once ``consumed`` reaches/exceeds the
    store size (the trigger is disarmed), never a negative.

    Args:
        reader: The LMDB replay reader.
        consumed: Number of leading records already consumed by a prior cycle.
        cap: Maximum NEW (post-consumed) records to count.

    Returns:
        The number of records after the consumed baseline, capped at ``cap``.
    """

    async def _run() -> int:
        seen = 0  # records observed (including the consumed prefix)
        new = 0  # records strictly AFTER the consumed baseline
        # Stream in BOUNDED windows: ``consumed + cap`` is the ideal single-pass
        # window, but it grows without limit as ``consumed`` accrues across fired
        # cycles, so it is capped at ``_MAX_REPLAY_COUNT_CHUNK`` to bound the peak
        # decoded memory per yield (avoids a Jetson OOM). The ``async for`` iterates
        # EVERY chunk, so the consumed prefix is still skipped and the new-record
        # cap is still filled — just across multiple bounded windows instead of one.
        chunk_size = min(_MAX_REPLAY_COUNT_CHUNK, consumed + cap)
        async for chunk in reader.stream(chunk_size=chunk_size):
            for _record in chunk:
                if seen >= consumed:
                    new += 1
                    if new >= cap:
                        return new
                seen += 1
        return new

    return _run_coro_blocking(_run())


def _load_replay_batch(reader: LMDBReplayReader, input_dim: int, cap: int) -> Tensor:
    """Build a ``(n, input_dim)`` batch tensor from replay vision-features.

    Args:
        reader: The LMDB replay reader.
        input_dim: Target feature width; each record's vision-features vector
            is resized to this width.
        cap: Maximum records to draw into the batch.

    Returns:
        A ``(n, input_dim)`` float32 tensor (empty when the store has no
        records).
    """
    import numpy as np
    import torch

    async def _run() -> list[MouseDroidExperienceRecord]:
        rows: list[MouseDroidExperienceRecord] = []
        async for chunk in reader.stream(chunk_size=cap):
            rows.extend(chunk)
            if len(rows) >= cap:
                break
        return rows[:cap]

    records = _run_coro_blocking(_run())
    if not records:
        return torch.empty(0, input_dim)
    matrix = np.stack(
        [np.resize(np.asarray(rec.vision_features, dtype=np.float32), input_dim) for rec in records]
    )
    return torch.from_numpy(matrix)


def _load_replay_sequence_batch(
    reader: LMDBReplayReader,
    model_cfg: ModelConfig,
    encoder: MultimodalEncoder,
    *,
    sequence_length: int,
    n_episodes: int,
    cap: int,
    device: torch.device,
) -> dict[str, Tensor]:
    """Build a ``(B, T, ...)`` RSSM sequence batch from replay (WS-E2 path).

    Reads up to ``cap`` records off the event loop (via :func:`_run_coro_blocking`)
    and assembles them into the sequence-dict batch
    :meth:`mousedroid.world_model.rssm.RSSM.train_sequence` expects through
    :func:`mousedroid.learning.on_device.rssm_refiner.build_sequence_batch`.

    Returns an EMPTY dict when the store holds fewer than
    ``n_episodes * sequence_length`` records — too few to fill even one batch.
    The coordinator's dict-aware empty-check treats an empty dict as a safe skip
    (``on_device_trigger_empty_batch``), so a fresh / sparse Jetson never crashes
    the refiner.

    Args:
        reader: The LMDB replay reader.
        model_cfg: The model config supplying per-modality dims.
        encoder: The live world-model encoder whose ``*_enabled`` flags drive the
            mask length + assembled modality tensors.
        sequence_length: Temporal length ``T`` of each window.
        n_episodes: Batch dimension ``B`` (number of windows).
        cap: Maximum records to draw from the store.
        device: Device on which to place every assembled tensor (the refiner's
            candidate device).

    Returns:
        A ``(B, T, ...)`` sequence-dict batch, or an empty dict when the store
        holds too few records to fill one batch.
    """
    from mousedroid.learning.on_device.rssm_refiner import build_sequence_batch

    needed = n_episodes * sequence_length

    async def _run() -> list[MouseDroidExperienceRecord]:
        rows: list[MouseDroidExperienceRecord] = []
        async for chunk in reader.stream(chunk_size=cap):
            rows.extend(chunk)
            if len(rows) >= cap:
                break
        return rows[:cap]

    records = _run_coro_blocking(_run())
    if len(records) < needed:
        _log.debug(
            "on_device_replay_sequence_below_batch",
            have=len(records),
            needed=needed,
            n_episodes=n_episodes,
            sequence_length=sequence_length,
        )
        return {}
    return build_sequence_batch(
        records,
        model_cfg,
        encoder,
        sequence_length=sequence_length,
        n_episodes=n_episodes,
        device=device,
    )


def _build_held_out_sequence_batch(
    reader: LMDBReplayReader | None,
    model_cfg: ModelConfig,
    encoder: MultimodalEncoder,
    *,
    sequence_length: int,
    n_episodes: int,
    refine_offset: int,
    device: torch.device,
) -> dict[str, Tensor] | None:
    """Build the FIXED WS-E3 gate held-out batch DISJOINT from the refine batch.

    The refine batch consumes the FIRST ``refine_offset``
    (``= refine_n_episodes * refine_sequence_length``) records. To score the gate
    on data the refiner did NOT train on, the held-out batch is assembled from the
    records AFTER that window: this reads up to ``refine_offset + needed`` records
    and partitions the trailing ``needed`` of them into the held-out ``(B, T, ...)``
    windows.

    Args:
        reader: The LMDB replay reader (``None`` ⇒ ``None`` returned).
        model_cfg: The model config supplying per-modality dims.
        encoder: The live world-model encoder whose ``*_enabled`` flags drive the
            mask length + assembled modality tensors.
        sequence_length: Temporal length ``T`` of each held-out window.
        n_episodes: Batch dimension ``B`` of the held-out batch.
        refine_offset: Number of leading records the refine batch consumes; the
            held-out window starts immediately after these.
        device: Device on which to place every assembled tensor.

    Returns:
        A FIXED ``(B, T, ...)`` held-out sequence-dict batch, or ``None`` when no
        reader is wired or the store holds too few records for a disjoint window.
    """
    if reader is None:
        return None

    from mousedroid.learning.on_device.rssm_refiner import build_sequence_batch

    needed = n_episodes * sequence_length
    cap = refine_offset + needed

    async def _run() -> list[MouseDroidExperienceRecord]:
        rows: list[MouseDroidExperienceRecord] = []
        async for chunk in reader.stream(chunk_size=cap):
            rows.extend(chunk)
            if len(rows) >= cap:
                break
        return rows[:cap]

    records = _run_coro_blocking(_run())
    held_out = records[refine_offset : refine_offset + needed]
    if len(held_out) < needed:
        _log.warning(
            "on_device_gate_held_out_below_batch",
            have=len(held_out),
            needed=needed,
            refine_offset=refine_offset,
            total_records=len(records),
        )
        return None
    return build_sequence_batch(
        held_out,
        model_cfg,
        encoder,
        sequence_length=sequence_length,
        n_episodes=n_episodes,
        device=device,
    )
