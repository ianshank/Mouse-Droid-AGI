"""MouseDroid orchestrator — world model state mixin.

Handles latent state management, validation, and OTA weight updates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mousedroid.cloud.protocol import ENGINE_TYPE_POLICY, ENGINE_TYPE_WORLD_MODEL
from mousedroid.logging.setup import get_logger
from mousedroid.orchestrator._state import _OrchestratorState

if TYPE_CHECKING:
    from mousedroid.cloud.protocol import PendingWeightUpdate, WeightUpdatePollerProtocol
    from mousedroid.sensing.protocol import ObservationProtocol
    from mousedroid.vla.policy import VLAPolicyProtocol
    from mousedroid.world_model.protocol import WorldModelProtocol

_log = get_logger(__name__)


class _WorldModelStateMixin(_OrchestratorState):
    """World model state management for the orchestrator."""

    def _update_world_model(self, observation: ObservationProtocol) -> None:
        """Run world model observation step to update latent state.

        When the F-023 bounded-context memory is wired, healthy ticks store
        the RAW validated state and then blend the retrieved context into the
        carried ``(h, z)``. Unhealthy ticks (NaN detected — recovered or not)
        skip both calls so a transient NaN can never poison the memory's
        EMA/sink and today's self-healing path is preserved.

        Args:
            observation: Current sensor observation bundle.
        """
        with torch.no_grad():
            self._h, self._z, _, _ = self._world_model.observe_step(
                observation,
                self._prev_action,
                self._h,
                self._z,
            )
        self._h, self._z, healthy = self._validate_latent(self._h, self._z)
        if self._latent_context is not None and healthy:
            self._latent_context.observe(self._h, self._z)
            self._h, self._z = self._latent_context.contextualize(self._h, self._z)

    def _validate_latent(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Check latent state for NaN / saturation; recover from buffer on NaN.

        Performance: three independent scalars (NaN-in-h, NaN-in-z,
        h-norm) are computed on-device, stacked into a single rank-1
        tensor, and read back with one ``.tolist()`` call. This collapses
        three GPU→CPU syncs down to one on the 30 Hz hot path. Addresses
        review comments on GPU synchronization overhead.

        Args:
            h: Hidden state tensor from ``observe_step``.
            z: Latent state tensor from ``observe_step``.

        Returns:
            ``(h, z, healthy)`` — the state (possibly replaced by the last
            known-good values when NaN is detected and the recovery buffer is
            non-empty) and a ``healthy`` flag that is ``False`` on any
            NaN tick (recovered or unrecoverable). Downstream consumers that
            must never ingest a NaN-tick state (the F-023 bounded-context
            memory) gate on the flag.
        """
        # Single GPU→CPU sync covering all three diagnostics. ``stack``
        # forces consistent dtype/device so ``.tolist()`` returns Python
        # floats in one round-trip.
        diagnostics = torch.stack(
            [
                torch.isnan(h).any().to(torch.float32),
                torch.isnan(z).any().to(torch.float32),
                torch.linalg.norm(h.float()),
            ]
        )
        nan_h_float, nan_z_float, h_norm = diagnostics.tolist()
        has_nan = bool(nan_h_float) or bool(nan_z_float)

        if has_nan:
            self._failure_recorder.record(
                "world_model",
                "latent_nan",
                level="critical",
                extra={"tick": self._tick_count},
            )
            _log.critical("world_model_latent_nan", tick=self._tick_count)
            if self._latent_buffer:
                h_last, z_last = self._latent_buffer[-1]
                _log.info("world_model_latent_recovered", tick=self._tick_count)
                return h_last.clone(), z_last.clone(), False
            _log.critical("world_model_latent_unrecoverable", tick=self._tick_count)
            return h, z, False

        if h_norm > self._cfg.model.latent_norm_threshold:
            self._failure_recorder.record(
                "world_model",
                "latent_saturated",
                level="warning",
                extra={"h_norm": round(h_norm, 3)},
            )
            _log.warning("world_model_latent_saturated", h_norm=h_norm)

        self._latent_buffer.append((h.clone(), z.clone()))
        return h, z, True

    def _maybe_rearm_latent_sink(self, *, mission_completed: bool) -> None:
        """Re-arm the bounded-context sink at mission boundaries (F-023).

        Makes the sink a per-mission anchor rather than a stale boot
        snapshot. Only the sink re-captures; the ring and EMA summary are
        retained. Gated by ``world_model_memory.recapture_on_mission``
        (explicit ``None`` checks — no asserts; ``-O``-safe).

        Args:
            mission_completed: Snapshot of the dispatcher's
                ``mission_just_completed`` latch from the tick's
                centralised read.
        """
        if not mission_completed:
            return
        if self._latent_context is None:
            return
        memory_cfg = self._cfg.world_model_memory
        if memory_cfg is None or not memory_cfg.recapture_on_mission:
            return
        self._latent_context.rearm_sink()

    def _apply_pending_weight_update(self) -> bool:
        """Atomically swap policy / world-model if any poller has a verified update.

        Runs ONCE per tick, AFTER ``_select_action`` returns. Guarantees the
        current tick saw one consistent weight set for both
        ``_update_world_model`` and ``_select_action``. Reference assignment
        is atomic at the Python interpreter level; we hold no locks because
        the orchestrator's ``tick()`` is single-coroutine on the event loop.

        With Tier C1.2 the orchestrator holds a ``Mapping[str, poller]``
        keyed by ``engine_type``. Each tick this method iterates the mapping
        in the caller-provided insertion order of
        ``self._weight_update_pollers`` and delegates per-poller swap work
        to :meth:`_apply_one_pending_update`. The
        ``build_weight_update_pollers`` factory guarantees ``policy`` before
        ``world_model``; callers constructing ``MouseDroidOrchestrator``
        directly are responsible for the ordering they want. Iteration
        order matters: a world-model swap may zero the recurrent state on
        the same tick, so applying ``policy`` first prevents a stale-policy
        artefact from leaking into a freshly reset world model.

        Method is INTENTIONALLY synchronous: ``tick()`` is the only caller,
        the swap runs entirely in process memory (no I/O after the poller
        downloaded), and keeping it sync avoids scheduling churn between
        select_action and execute_action.

        Returns:
            ``True`` iff at least one world-model swap performed a recurrent-
            state reset (caller MUST skip its own ``_prev_action = action``
            assignment so the zero-state survives into the next tick).
            ``False`` for any other code path (empty mapping, no pendings,
            policy-only swap, loader failure, dead-letter, etc.).
        """
        if not self._weight_update_pollers:
            return False
        any_reset = False
        for poller in self._weight_update_pollers.values():
            update = poller.pending_update
            if update is None:
                continue
            if self._apply_one_pending_update(poller, update):
                any_reset = True
        return any_reset

    def _apply_one_pending_update(
        self,
        poller: WeightUpdatePollerProtocol,
        update: PendingWeightUpdate,
    ) -> bool:
        """Apply one pending update from one poller.

        Extracted from :meth:`_apply_pending_weight_update` so the multi-
        poller loop can delegate per-poller swap work uniformly. The body
        is the unchanged single-poller swap path from Tier C1 — it owns
        the loader invocation, atomic reference swap, engine-type dispatch,
        recurrent-state reset, metric increment, structured-log emission,
        and the final ``acknowledge_swap`` call.

        The new engine is fully materialised via ``self._weight_update_loader``
        BEFORE the reference swap, so a loader failure does NOT corrupt the
        live model — the helper logs the error, leaves the live model
        untouched, and clears the pending slot only on success.

        When ``cfg.cloud.weight_update.reset_state_on_swap`` is ``True`` (the
        default) the latent recurrent state ``(h, z)`` is reset to zeros
        after a world-model swap to avoid one-tick cross-model contamination
        (see ADR-010). The previous-action tensor and latent recovery buffer
        are also cleared in the same pass — they were produced by the OLD
        weights and would seed the new engine with stale context. Device +
        dtype are preserved via ``torch.zeros_like`` so a CUDA-resident
        world-model state survives the swap on its original device.

        Args:
            poller: The poller that surfaced ``update``. Used to invoke
                ``acknowledge_swap`` once the swap (or failure path) lands.
            update: The pending update to apply.

        Returns:
            ``True`` iff this swap zeroed the recurrent state (world-model
            engine only, gated by
            ``cfg.cloud.weight_update.reset_state_on_swap``). ``False`` for
            the no-loader branch, the loader-exception branch, a policy swap,
            and the unknown-engine-type dead-letter branch.
        """
        if self._weight_update_loader is None:
            # Acknowledge-and-warn-once: without ack the same pending update
            # would re-fire ``cloud_weight_update_swap_skipped_no_loader``
            # at 30 Hz forever (one log line per tick). Ack clears the slot
            # so the poller's next download cycle can surface a fresh update,
            # at which point the operator-visible warning fires again — once
            # per revision, not once per tick. (Copilot 3253293630.)
            _log.warning(
                "cloud_weight_update_swap_skipped_no_loader",
                repo_id=update.repo_id,
                revision=update.revision,
                engine_type=update.engine_type,
            )
            poller.acknowledge_swap(update)
            return False

        try:
            new_engine = self._weight_update_loader(update)
        except Exception:
            _log.error(
                "cloud_weight_update_swap_failed",
                repo_id=update.repo_id,
                revision=update.revision,
                engine_type=update.engine_type,
                exc_info=True,
            )
            # Ack the bad revision so we don't log-spam at 30 Hz. The
            # poller will surface a new PendingWeightUpdate on the next
            # cycle if the upstream artifact changes.
            poller.acknowledge_swap(update)
            return False

        # Atomic reference swap. Single-coroutine guarantee on tick() means
        # no concurrent reader observes a half-swapped state.
        reset_recurrent_state = False
        if update.engine_type == ENGINE_TYPE_WORLD_MODEL:
            self._world_model = cast("WorldModelProtocol", new_engine)
            reset_recurrent_state = self._cfg.cloud.weight_update.reset_state_on_swap
        elif update.engine_type == ENGINE_TYPE_POLICY:
            self._vla_policy = cast("VLAPolicyProtocol", new_engine)
        else:
            # Unknown engine type — acknowledge + dead-letter so the same
            # bad pending update doesn't stick around firing this warning
            # at 30 Hz. (Copilot 3253293637.)
            _log.warning(
                "cloud_weight_update_unknown_engine_type",
                engine_type=update.engine_type,
                repo_id=update.repo_id,
                revision=update.revision,
            )
            poller.acknowledge_swap(update)
            return False

        if reset_recurrent_state:
            # Use ``zeros_like`` so device + dtype are preserved. The live
            # world-model may run on CUDA; ``torch.zeros(...)`` with default
            # device would silently move state back to CPU and break the
            # next ``observe_step`` with a device-mismatch error.
            # (Copilot 3253293626 / 3253309982.)
            self._h = torch.zeros_like(self._h)
            self._z = torch.zeros_like(self._z)
            self._prev_action = torch.zeros_like(self._prev_action)
            self._latent_buffer.clear()
            if self._latent_context is not None:
                # A sink frozen under the pre-swap weights is stale under the
                # new weights: clear every store AND re-arm sink warmup so a
                # fresh anchor is captured post-swap (ADR-015).
                self._latent_context.reset()

        if self._metrics is not None:
            self._metrics.inc_cloud_weight_update_swap(update.engine_type)

        _log.info(
            "cloud_weight_update_swap_applied",
            repo_id=update.repo_id,
            revision=update.revision,
            engine_type=update.engine_type,
            reset_state=reset_recurrent_state,
        )
        poller.acknowledge_swap(update)
        return reset_recurrent_state
