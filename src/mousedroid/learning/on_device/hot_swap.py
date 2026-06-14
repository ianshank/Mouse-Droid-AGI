"""Off-loop hot-swap weight-update source (Phase 6 WS-E4).

Surfaces a PROMOTED on-device candidate slot to the orchestrator's C1 atomic
weight-swap seam as a :class:`~mousedroid.cloud.protocol.PendingWeightUpdate`,
gated by ``cfg.on_device_learning.enable_hot_swap`` (default-OFF: when the flag
is off this source is NEVER wired and the orchestrator is byte-identical to
#134).

SAFETY-LOCKED design (per the WS-E4 peer reviews):

* **Materialise OFF the hot loop.** The orchestrator's
  ``_weight_update_loader`` is called SYNCHRONOUSLY inside ``tick()``;
  ``torch.load`` + model construction there would blow ``tick_timeout_s`` and
  trip the emergency stop. So the disk-load + construct + device-place runs in
  this source's OWN slow-cadence background task (``refresh_once`` via
  :func:`asyncio.to_thread`), and the hot-loop loader is a PURE reference
  assignment — :meth:`take_materialized` just returns the already-constructed,
  device-correct engine.
* **Integrity = fail-closed + counted.** The injected ``materialize`` callable
  re-verifies the slot's SHA-256 (it reuses ``OnDeviceSlotStore.load``); on a
  :class:`~mousedroid.learning.on_device.slot_store.SlotIntegrityError` this
  source does NOT surface a pending update (the live model is untouched) AND
  increments ``inc_on_device_learning_reverted("integrity_mismatch")`` BEFORE
  any swap could occur — the C1 broad-except only logs
  ``cloud_weight_update_swap_failed`` and never touches the on-device counter.
* **Promotion stays separate from activation.** The WS-E3 gate marks a slot
  active (``slot_store.mark_active``); THIS source is the activation seam. With
  ``enable_hot_swap=False`` the factory never builds this source, so a marked-
  active slot is never swapped into the running model.

Conforms structurally to
:class:`~mousedroid.cloud.protocol.WeightUpdatePollerProtocol` PLUS the
``engine_type`` extension property (always ``world_model``), so it slots into
``orchestrator._weight_update_pollers`` and is driven by the existing
per-poller ``start``/``stop`` lifecycle + per-tick ``pending_update`` consume.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Protocol

from mousedroid.cloud.protocol import ENGINE_TYPE_WORLD_MODEL, EngineType, PendingWeightUpdate
from mousedroid.learning.on_device.slot_store import SlotIntegrityError
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from mousedroid.telemetry.metrics import MetricsRegistry

_log = get_logger(__name__)

# Synthetic ``repo_id``/``filename`` for the on-device pending update. The C1
# ``PendingWeightUpdate`` shape is HF-Hub-shaped (it was built for the cloud OTA
# poller), but an on-device slot has no repo. These constants make the slot's
# provenance greppable in the ``cloud_weight_update_swap_applied`` log + the
# ``engine_type`` Prometheus label without inventing a parallel update type.
_ON_DEVICE_REPO_ID: str = "on-device://active-slot"
_ON_DEVICE_FILENAME: str = "active_slot.pt"


class _SlotActiveSource(Protocol):
    """Structural slice of :class:`OnDeviceSlotStore` this source consumes."""

    def load_active(self) -> str | None:
        """Return the active (blessed) candidate's digest, or ``None``."""
        ...

    @property
    def slot_dir(self) -> Path:
        """Resolved slot directory (used only to stamp the update's local_path)."""
        ...


class OnDeviceWeightUpdateSource:
    """Surfaces the active on-device slot as a pre-materialised pending swap.

    Args:
        slot_store: The on-device slot store whose ``load_active`` digest this
            source watches and whose ``slot_dir`` stamps the update's
            ``local_path``.
        materialize: Off-loop materialiser. Given the active digest it must
            re-verify the slot (SHA-256) and return an ALREADY-constructed,
            device-correct engine — or raise
            :class:`~mousedroid.learning.on_device.slot_store.SlotIntegrityError`
            on a corrupt slot. Runs inside :func:`asyncio.to_thread` so the
            (blocking) ``torch.load`` + ``build_world_model`` never touches the
            event loop the 30 Hz hot loop shares.
        check_interval_s: Slow-cadence period (seconds) between active-slot
            probes in the background loop. Independent of the cloud OTA
            ``poll_interval_s``.
        metrics: Optional shared metrics registry. On a corrupt slot the source
            increments ``inc_on_device_learning_reverted("integrity_mismatch")``.
        engine_type: Engine discriminator the orchestrator dispatches on.
            Always ``world_model`` for the on-device RSSM refinement path.
    """

    def __init__(
        self,
        *,
        slot_store: _SlotActiveSource,
        materialize: Callable[[str], object],
        check_interval_s: float,
        metrics: MetricsRegistry | None = None,
        engine_type: EngineType = ENGINE_TYPE_WORLD_MODEL,
    ) -> None:
        self._slot_store = slot_store
        self._materialize = materialize
        self._check_interval_s = check_interval_s
        self._metrics = metrics
        self._engine_type: EngineType = engine_type

        self._pending_update: PendingWeightUpdate | None = None
        # The pre-materialised engine, keyed by ``id(update)`` so the hot-loop
        # loader retrieves it by identity (the frozen update carries no engine).
        self._engine_by_update: dict[int, object] = {}
        # The digest most recently MATERIALISED (or seen-and-rejected). Guards
        # against re-materialising an unchanged active slot every cadence AND
        # against retrying a known-corrupt digest in a tight loop.
        self._seen_digest: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Begin the background active-slot refresh loop."""
        if self._task is not None and not self._task.done():
            _log.debug("on_device_hot_swap_source_already_running")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="on_device_hot_swap_source")
        _log.info(
            "on_device_hot_swap_source_started",
            check_interval_s=self._check_interval_s,
            engine_type=self._engine_type,
        )

    async def stop(self) -> None:
        """Stop the refresh loop + cancel any in-flight materialisation."""
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        _log.info("on_device_hot_swap_source_stopped")

    async def _run(self) -> None:
        """Background loop — refreshes the active slot every ``check_interval_s``."""
        while not self._stop_event.is_set():
            try:
                await self.refresh_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # pylint: disable=broad-except
                _log.warning("on_device_hot_swap_refresh_failed", exc_info=True)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._check_interval_s)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise

    # ------------------------------------------------------------------
    # Off-loop refresh — the ONLY place the engine is constructed
    # ------------------------------------------------------------------

    async def refresh_once(self) -> bool:
        """Probe the active slot; materialise + surface a pending swap if new.

        Public so tests + a slow-cadence driver can run one refresh without the
        background loop. The heavy materialisation runs OFF the event loop via
        :func:`asyncio.to_thread`, so this coroutine never blocks the hot loop.

        Returns:
            ``True`` iff a NEW pre-materialised pending update was surfaced this
            call; ``False`` for the no-active-slot, unchanged-digest, and
            corrupt-slot (fail-closed) paths.
        """
        digest = self._slot_store.load_active()
        if digest is None:
            return False
        if digest == self._seen_digest:
            # Already materialised (pending) or already rejected as corrupt.
            return False

        try:
            engine = await asyncio.to_thread(self._materialize, digest)
        except SlotIntegrityError:
            # Fail-closed: do NOT surface a pending update (the live model is
            # untouched), and COUNT the revert here — the C1 swap path never
            # increments the on-device counter. Mark the digest seen so a
            # corrupt slot is not re-materialised every cadence.
            self._seen_digest = digest
            if self._metrics is not None:
                self._metrics.inc_on_device_learning_reverted("integrity_mismatch")
            _log.warning("on_device_hot_swap_slot_integrity_mismatch", digest=digest)
            return False

        update = PendingWeightUpdate(
            repo_id=_ON_DEVICE_REPO_ID,
            filename=_ON_DEVICE_FILENAME,
            revision=digest,
            sha256=digest,
            local_path=self._slot_store.slot_dir / f"{digest}.pt",
            downloaded_at=time.time(),
            engine_type=self._engine_type,
        )
        self._seen_digest = digest
        self._pending_update = update
        self._engine_by_update[id(update)] = engine
        _log.info(
            "on_device_hot_swap_pending",
            digest=digest,
            engine_type=self._engine_type,
        )
        return True

    # ------------------------------------------------------------------
    # Public surface for the orchestrator + loader
    # ------------------------------------------------------------------

    @property
    def engine_type(self) -> EngineType:
        """Engine discriminator the orchestrator dispatches on (always world_model)."""
        return self._engine_type

    @property
    def pending_update(self) -> PendingWeightUpdate | None:
        """Latest pre-materialised update awaiting the orchestrator swap (``None`` if none)."""
        return self._pending_update

    def take_materialized(self, update: PendingWeightUpdate) -> object:
        """Return the pre-materialised engine for ``update`` (PURE — no I/O).

        Invoked by the orchestrator's ``_weight_update_loader`` INSIDE ``tick()``.
        It must do NO disk I/O / model construction — the engine was built
        off-loop in :meth:`refresh_once`. A reference lookup by ``id(update)``.

        Args:
            update: The pending update the orchestrator is about to apply.

        Returns:
            The already-constructed, device-correct engine.

        Raises:
            KeyError: If no engine was materialised for ``update`` (the update
                was already acknowledged, or did not originate from this source).
        """
        return self._engine_by_update[id(update)]

    def owns(self, update: PendingWeightUpdate) -> bool:
        """Return ``True`` iff this source materialised ``update`` (loader dispatch)."""
        return id(update) in self._engine_by_update

    def acknowledge_swap(self, update: PendingWeightUpdate) -> None:
        """Clear the pending slot + evict the cached engine after the swap.

        Identity check (``is``) on the slot so a stale acknowledge never clears
        a freshly surfaced update.

        Args:
            update: The update the orchestrator has applied (or dead-lettered).
        """
        self._engine_by_update.pop(id(update), None)
        if self._pending_update is update:
            self._pending_update = None
            _log.debug("on_device_hot_swap_swap_acknowledged", digest=update.sha256)


__all__ = ["OnDeviceWeightUpdateSource"]
