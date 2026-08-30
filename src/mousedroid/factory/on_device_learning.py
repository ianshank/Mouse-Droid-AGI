"""Factory builders — on-device RSSM refinement coordinator and WS-E4 hot-swap source."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from mousedroid.factory._replay_batch_helpers import (
    _count_new_replay_records,
    _load_replay_sequence_batch,
    _make_consumed_offset_advancer,
)
from mousedroid.factory.world_model import build_world_model
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    import torch
    from torch import Tensor

    from mousedroid.config.schema import (
        ModelConfig,
        Settings,
    )
    from mousedroid.learning.on_device.hot_swap import OnDeviceWeightUpdateSource
    from mousedroid.learning.on_device.slot_store import CandidateSlot, OnDeviceSlotStore
    from mousedroid.telemetry.metrics import MetricsRegistry
    from mousedroid.training.replay.lmdb_reader import LMDBReplayReader
    from mousedroid.world_model.encoder import MultimodalEncoder
    from mousedroid.world_model.protocol import WorldModelProtocol
    from mousedroid.world_model.rssm import RSSM

from mousedroid.factory._replay_batch_helpers import (
    _build_held_out_sequence_batch,
    _build_shared_replay_reader,
)

_log = get_logger(__name__)
_SLOT_PT_SUFFIX: str = ".pt"


def build_on_device_coordinator(
    cfg: Settings,
    *,
    metrics: MetricsRegistry | None = None,
    world_model: WorldModelProtocol | None = None,
) -> object | None:
    """Build the Phase-6 WS3/WS4 replay-trigger on-device update coordinator.

    Returns ``None`` (so the orchestrator stays byte-identical to pre-WS3)
    whenever ``cfg.on_device_learning`` is absent or disabled. When enabled,
    wires the REUSED collaborators: the LMDB replay reader (new-record trigger
    + sequence-batch source), the WS-E2 :class:`RSSMRefiner` over the LIVE RSSM
    world model (deep-copied per update so the base stays bitwise-unchanged), and
    the SHA-256-stamped :class:`OnDeviceSlotStore` resolved under
    ``cfg.experience.path``.

    WS-E2 REPLACES the pre-ENABLEMENT ``EWCOnlineLearner``-over-``nn.Linear``
    stand-in: the learner now refines the rover's ACTUAL learned component (the
    RSSM dynamics MCTS plans through) via ``train_sequence`` over a ``(B, T, ...)``
    sequence-dict batch, and the persisted slot is a refined RSSM ``state_dict``
    that round-trips into a fresh ``build_world_model(cfg)``.

    The coordinator PRODUCES + PERSISTS the candidate slot, then the WS-E3
    safety-regression gate scores candidate-RSSM vs baseline-RSSM on a FIXED
    held-out replay batch via **reconstruction + KL loss** (``train_sequence``;
    lower-is-better) and promotes-or-reverts: PROMOTE iff ``candidate_loss <=
    baseline_loss + regression_tolerance`` (marks the slot active), else REVERT
    (increments the revert counter). The candidate is the persisted slot loaded
    into a deep-copy of the live RSSM, so the live model is bitwise-unchanged on
    both revert AND promote (activation is the separate ``enable_hot_swap`` seam).
    The retired self-gaming imagined-return metric is NOT used by the gate.

    Args:
        cfg: Root settings.
        metrics: Optional shared metrics registry, threaded keyword-only so the
            revert counter surfaces on ``/metrics``. ``None`` (the default)
            keeps the revert path working without recording a metric.
        world_model: Optional live world model (the orchestrator's already-built
            ``build_world_model(cfg)`` result), threaded keyword-only so the gate
            + refiner act on the REAL model the rover runs. ``None`` (the default)
            resolves the effective model via ``build_world_model(cfg)`` instead of
            the old wrong-arch ``RSSM(cfg.model)`` literal. NOTE: ``None`` is NOT
            unconditionally behaviour-preserving — when the effective engine lacks
            ``train_sequence`` (e.g. ``DualStreamRSSM`` / ``DualStreamRSSMOnnx``)
            the capability gate below returns ``None`` and logs
            ``on_device_refiner_unsupported_engine`` (on-device refinement is
            only supported for the classic ``RSSM`` engine).

    Returns:
        A ``ReplayTriggerCoordinator`` when enabled, else ``None``.
    """
    on_device_cfg = cfg.on_device_learning
    if on_device_cfg is None or not on_device_cfg.enabled:
        return None

    # Capability gate: on-device refinement requires ``train_sequence`` (present
    # on ``RSSM`` but NOT on ``DualStreamRSSM`` / ``DualStreamRSSMOnnx``). Resolve
    # the effective world model ONCE here (the injected instance, else the one
    # ``build_world_model(cfg)`` constructs) and thread it down to the gate runner
    # so it is never built twice. When the effective engine lacks the capability,
    # disable refinement (return ``None``) instead of wiring an unusable refiner.
    effective_wm = world_model if world_model is not None else build_world_model(cfg)
    if not hasattr(effective_wm, "train_sequence"):
        _log.warning(
            "on_device_refiner_unsupported_engine",
            engine_type=type(effective_wm).__name__,
            hint="active world-model engine lacks train_sequence; on-device refinement disabled",
        )
        return None

    import torch

    from mousedroid.learning.on_device.replay_trigger import ReplayTriggerCoordinator
    from mousedroid.learning.on_device.rssm_refiner import RSSMRefiner
    from mousedroid.learning.on_device.slot_store import OnDeviceSlotStore
    from mousedroid.world_model.rssm import RSSM

    # The capability gate above guarantees ``train_sequence`` — which only the
    # plain ``RSSM`` exposes (``DualStreamRSSM`` / the ONNX engine do not). Narrow
    # to the concrete ``RSSM`` so the refiner + sequence-batch builder are
    # statically typed (``.cfg`` / ``.encoder`` / ``train_sequence``) with no
    # suppression. A surprise non-RSSM engine that nonetheless carries a
    # ``train_sequence`` attribute disables refinement rather than crashing.
    if not isinstance(effective_wm, RSSM):
        _log.warning(
            "on_device_refiner_unsupported_engine",
            engine_type=type(effective_wm).__name__,
            hint="train_sequence present but engine is not a concrete RSSM; refinement disabled",
        )
        return None

    # Mirror the main replay path's reader construction (see build above) so the
    # on-device trigger honours any ``cfg.training.replay.source_path`` override
    # and the shared debug-log cadence instead of reading a different store.
    reader = _build_shared_replay_reader(cfg)
    slot_store = OnDeviceSlotStore(experience_cfg=cfg.experience, on_device_cfg=on_device_cfg)

    # WS-E2: the learner refines the LIVE RSSM world model (deep-copied per
    # update so the base is bitwise-unchanged) via ``train_sequence`` — replacing
    # the pre-ENABLEMENT ``nn.Linear`` stand-in over ``EWCOnlineLearner``. The
    # refined RSSM ``state_dict`` round-trips into a fresh ``build_world_model``.
    learner = RSSMRefiner(effective_wm, on_device_cfg)

    # The sequence batch is partitioned into ``refine_batch_episodes`` windows of
    # ``refine_sequence_length`` consecutive steps, so the loader must draw at
    # least that many records; ``trigger_min_new_records`` is the trigger gate,
    # not the batch size. Cap the scan at whichever is larger so the batch is
    # always fillable once the trigger fires.
    seq_len = on_device_cfg.refine_sequence_length
    n_episodes = on_device_cfg.refine_batch_episodes
    needed = n_episodes * seq_len
    cap = max(on_device_cfg.trigger_min_new_records, needed)

    # Resolve the model's device so the assembled batch lands where the refiner's
    # candidate (a deep-copy of ``effective_wm``) lives — never a hardcoded CPU.
    first_param = next(effective_wm.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cpu")
    encoder = effective_wm.encoder
    model_cfg = effective_wm.cfg

    # In-memory consumed offset: the count of replay records already consumed by
    # a FIRED cycle. The trigger arms on records BEYOND this baseline, never on
    # the total store size — otherwise a store that has crossed
    # ``trigger_min_new_records`` would re-fire the refine + gate every cadence on
    # byte-identical stale data (it never disarms). ``on_consumed`` advances it.
    # A list cell (not ``nonlocal``) keeps the two closures sharing one mutable
    # counter without rebinding gymnastics.
    consumed_offset = [0]

    def _count_new_records() -> int:
        """Count NEW replay records since the last consumed baseline (slow probe)."""
        return _count_new_replay_records(reader, consumed=consumed_offset[0], cap=cap)

    # Wired as the coordinator's ``on_consumed`` callback so the NEXT cycle
    # counts from the new baseline — the trigger disarms until fresh experience
    # accumulates past it again.
    _advance_consumed = _make_consumed_offset_advancer(
        consumed_offset, log_event="on_device_consumed_offset_advanced"
    )

    def _load_batch() -> dict[str, Tensor]:
        """Materialise one ``(B, T, ...)`` sequence-dict batch from replay.

        Returns an EMPTY dict when the store holds fewer than
        ``refine_batch_episodes * refine_sequence_length`` records — the
        coordinator's dict-aware empty-check treats that as a safe skip.
        """
        return _load_replay_sequence_batch(
            reader,
            model_cfg,
            encoder,
            sequence_length=seq_len,
            n_episodes=n_episodes,
            cap=cap,
            device=device,
        )

    gate_runner = _build_on_device_gate_runner(
        cfg,
        slot_store=slot_store,
        metrics=metrics,
        world_model=effective_wm,
        reader=reader,
        model_cfg=model_cfg,
        encoder=encoder,
        device=device,
        refine_sequence_length=seq_len,
        refine_n_episodes=n_episodes,
    )

    return ReplayTriggerCoordinator(
        cfg=on_device_cfg,
        learner=learner,
        slot_store=slot_store,
        count_new_records=_count_new_records,
        load_batch=_load_batch,
        on_consumed=_advance_consumed,
        gate_runner=gate_runner,
    )


def build_on_device_hot_swap_source(
    cfg: Settings,
    *,
    world_model: WorldModelProtocol | None = None,
    metrics: MetricsRegistry | None = None,
) -> OnDeviceWeightUpdateSource | None:
    """Build the WS-E4 off-loop hot-swap source for a promoted on-device slot.

    Returns ``None`` (so the orchestrator is byte-identical to #134 — NO swap
    path wired AT ALL) whenever:

    * ``cfg.on_device_learning`` is absent / disabled, OR
    * ``cfg.on_device_learning.enable_hot_swap`` is ``False`` (the default —
      promotion via ``mark_active`` stays SEPARATE from activation), OR
    * the live world-model engine is not a concrete ``RSSM`` (a same-cfg strict
      ``load_state_dict`` round-trip is only defined for the engine the slot was
      refined from; ``DualStreamRSSM`` / the ONNX engine cannot accept an RSSM
      ``state_dict``, mirroring the coordinator's capability gate).

    When enabled, wires a :class:`OnDeviceWeightUpdateSource` whose injected
    materialiser does the SLOW work OFF the hot loop: it reconstructs the
    ``CandidateSlot`` from the active digest, ``slot_store.load`` re-verifies the
    SHA-256 (raising :class:`SlotIntegrityError` on a corrupt slot — fail-closed),
    builds a fresh engine via :func:`build_world_model`, strict-loads the slot's
    weights, and places it on the SAME device as the live ``world_model`` (NEVER
    ``cuda-if-available`` like ``load_rssm_with_migration`` — else the
    orchestrator's ``zeros_like(self._h)`` reset is a cross-device op). The
    hot-loop loader then only returns this already-constructed engine.

    Args:
        cfg: Root settings.
        world_model: The live world model (the orchestrator's already-built
            ``build_world_model(cfg)`` result), threaded so the swap engine is
            materialised on the SAME device + architecture. ``None`` resolves it
            via ``build_world_model(cfg)`` (used by unit tests).
        metrics: Optional shared metrics registry, threaded so the off-loop
            ``integrity_mismatch`` revert counter surfaces on ``/metrics``.

    Returns:
        An :class:`OnDeviceWeightUpdateSource` when hot-swap is enabled + the
        engine is a concrete ``RSSM``, else ``None``.
    """
    on_device_cfg = cfg.on_device_learning
    if on_device_cfg is None or not on_device_cfg.enabled or not on_device_cfg.enable_hot_swap:
        return None

    import torch

    from mousedroid.learning.on_device.hot_swap import OnDeviceWeightUpdateSource
    from mousedroid.learning.on_device.slot_store import CandidateSlot, OnDeviceSlotStore
    from mousedroid.world_model.rssm import RSSM

    effective_wm = world_model if world_model is not None else build_world_model(cfg)
    if not isinstance(effective_wm, RSSM):
        # A same-cfg strict state_dict round-trip is only defined for the classic
        # RSSM engine; disable activation rather than wiring an unusable swap.
        _log.warning(
            "on_device_hot_swap_unsupported_engine",
            engine_type=type(effective_wm).__name__,
            hint="hot-swap activation requires a concrete RSSM engine; disabled",
        )
        return None

    slot_store = OnDeviceSlotStore(experience_cfg=cfg.experience, on_device_cfg=on_device_cfg)

    # Resolve the live model's device ONCE so every materialised engine lands
    # where the live recurrent state lives (device parity contract).
    first_param = next(effective_wm.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cpu")

    def _materialize(digest: str) -> object:
        """OFF-loop: reconstruct the slot, re-verify, build + device-place the engine.

        Runs inside the source's ``asyncio.to_thread`` so the blocking
        ``torch.load`` + ``build_world_model`` never touches the event loop.
        Raises :class:`SlotIntegrityError` (out of ``slot_store.load``) on a
        corrupt slot so the source fails closed + counts ``integrity_mismatch``.
        """
        # load_active() returns a DIGEST STRING — reconstruct the content-
        # addressed CandidateSlot and load via the store (re-verifies SHA-256).
        slot = CandidateSlot(path=slot_store.slot_dir / f"{digest}{_SLOT_PT_SUFFIX}", digest=digest)
        state_dict = slot_store.load(slot)
        engine = build_world_model(cfg)
        # Narrow to the concrete ``RSSM`` so ``load_state_dict`` / ``to`` / ``eval``
        # (nn.Module surface) type-check WITHOUT a suppression. Same engine the
        # source's capability gate already confirmed for the live model, so this
        # only differs if ``build_world_model`` is non-deterministic (a real bug).
        if not isinstance(engine, RSSM):
            msg = f"hot-swap materialise built a non-RSSM engine: {type(engine).__name__}"
            raise RuntimeError(msg)
        # Same-cfg slot ⇒ STRICT load (NOT load_rssm_with_migration, which forces
        # cuda-if-available + tolerates dim drift). A dim mismatch here is a real
        # bug that must surface, not be silently migrated.
        engine.load_state_dict(state_dict, strict=True)
        engine.to(device)
        engine.eval()
        return engine

    _log.info(
        "on_device_hot_swap_source_wired",
        check_interval_s=on_device_cfg.check_interval_s,
        device=str(device),
    )
    return OnDeviceWeightUpdateSource(
        slot_store=slot_store,
        materialize=_materialize,
        check_interval_s=on_device_cfg.check_interval_s,
        metrics=metrics,
    )


def _build_on_device_gate_runner(
    cfg: Settings,
    *,
    slot_store: OnDeviceSlotStore,
    metrics: MetricsRegistry | None,
    world_model: RSSM,
    reader: LMDBReplayReader | None = None,
    model_cfg: ModelConfig,
    encoder: MultimodalEncoder,
    device: torch.device,
    refine_sequence_length: int,
    refine_n_episodes: int,
) -> Callable[[CandidateSlot], None]:
    """Build the WS-E3 RSSM-vs-RSSM recon-loss safety-gate runner closure.

    Scores a refined candidate **RSSM** against the live baseline **RSSM** by
    their held-out reconstruction+KL loss (``score_dynamics``) on a SHARED FIXED
    held-out ``(B, T, ...)`` batch with SHARED reconstruction heads. This REPLACES
    the pre-ENABLEMENT path which scored config-sized policy stand-ins by their
    imagined return — a metric that SELF-GAMED on reward-head inflation (proven in
    the WS-E-SPIKE; the imagined-return metric is retired).

    The candidate is the persisted slot's refined weights loaded (per evaluation)
    into a DEEP COPY of the live RSSM; the baseline is the live RSSM's current
    weights. Loading into a copy keeps the live model bitwise-unchanged — a revert
    leaves the running brain untouched, a promote only marks the slot active
    (activation is the separate ``enable_hot_swap`` WS-E4 seam).

    The held-out batch is built ONCE here over a held-out replay slice DISJOINT
    from the refine batch (the refine batch reads the FIRST
    ``refine_n_episodes * refine_sequence_length`` records; the held-out window is
    drawn from the records AFTER it). When the store holds too few records for a
    disjoint held-out window the gate is a NO-OP (logged) so a fresh/sparse Jetson
    never crashes — promotion simply waits for more experience.

    Args:
        cfg: Root settings.
        slot_store: The slot store whose ``mark_active`` is called on PROMOTE and
            whose ``load`` deserialises the persisted candidate slot.
        metrics: Optional revert counter.
        world_model: The live concrete ``RSSM`` baseline (narrowed by the
            ``build_on_device_coordinator`` capability gate). Deep-copied per
            evaluation to materialise the candidate so the live model is never
            mutated.
        reader: The LMDB replay reader used to source the FIXED held-out batch.
            ``None`` ⇒ no held-out batch ⇒ the gate runner is a logged no-op.
        model_cfg: The live model config supplying per-modality dims.
        encoder: The live world-model encoder whose ``*_enabled`` flags drive the
            held-out batch's mask length + modality tensors.
        device: The device the live model lives on (where the batch + decoders +
            candidate copy are placed).
        refine_sequence_length: ``T`` of the refine batch windows (so the held-out
            window is drawn DISJOINT, after the refine window).
        refine_n_episodes: ``B`` of the refine batch (the refine window spans the
            first ``refine_n_episodes * refine_sequence_length`` records).

    Returns:
        A ``(CandidateSlot) -> None`` closure invoked by the coordinator after
        persist (offloaded off the event loop).
    """
    import copy

    import torch

    from mousedroid.learning.on_device.regression_gate import RegressionGate
    from mousedroid.learning.on_device.slot_store import SlotIntegrityError
    from mousedroid.world_model.rssm import RawModalityDecoders

    on_device_cfg = cfg.on_device_learning
    if on_device_cfg is None:
        # Caller-guarded; explicit check (no assert — stripped under -O).
        msg = "on_device_learning config required to build the WS-E3 gate runner"
        raise ValueError(msg)

    world_model.eval()

    # Build the FIXED held-out batch ONCE over a slice DISJOINT from the refine
    # batch. ``score_dynamics`` correctness depends on a representative, fixed
    # held-out set, so the gate is scored against the SAME batch every cycle.
    held_out_batch = _build_held_out_sequence_batch(
        reader,
        model_cfg,
        encoder,
        sequence_length=refine_sequence_length,
        n_episodes=refine_n_episodes,
        refine_offset=refine_n_episodes * refine_sequence_length,
        device=device,
    )

    # SHARED reconstruction heads: the SAME instance scores baseline AND candidate
    # (recon heads are external to the RSSM ``state_dict``; scoring against
    # different heads is meaningless). Seed the head init so the held-out gate is
    # reproducible across process restarts — but confine the seed to a
    # ``fork_rng`` so it never leaks into the caller's process RNG stream (the
    # build runs at orchestrator construction; an unrestored ``manual_seed`` would
    # silently shift every subsequent draw). ``manual_seed`` reseeds CPU AND EVERY
    # CUDA generator whenever CUDA is available — regardless of the model's device —
    # so the fork must cover ALL CUDA devices, not just the model's, else the
    # reseed leaks onto the unforked generators. Pass the explicit full device list
    # (never ``devices=None``) so every CUDA generator is restored without tripping
    # the ``devices=None`` multi-GPU UserWarning.
    fork_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(on_device_cfg.scoring_seed)
        decoders = RawModalityDecoders(model_cfg).to(device)

    if held_out_batch is None:
        # No disjoint held-out window available (no reader / too few records): the
        # gate cannot score, so it is a safe no-op. Promotion waits for more
        # experience rather than scoring against an unrepresentative / empty batch.
        def _noop_gate(slot: CandidateSlot) -> None:
            _log.warning(
                "on_device_gate_skipped_no_held_out_batch",
                digest=slot.digest,
                hint="no disjoint held-out replay window available; candidate left unpromoted",
            )

        return _noop_gate

    gate = RegressionGate(
        cfg=on_device_cfg,
        slot_store=slot_store,
        metrics=metrics,
        held_out_batch=held_out_batch,
        decoders=decoders,
    )

    def _run_gate(slot: CandidateSlot) -> None:
        # Materialise the candidate by loading the refined slot weights into a
        # DEEP COPY of the live RSSM — the live baseline stays bitwise-unchanged.
        # A corrupt slot fails its SHA-256 check on load → count integrity_mismatch
        # and leave the slot unpromoted (fail-closed; the C1 swap is never reached).
        try:
            candidate_state_dict = slot_store.load(slot)
        except SlotIntegrityError:
            if metrics is not None:
                metrics.inc_on_device_learning_reverted("integrity_mismatch")
            _log.warning("on_device_gate_slot_integrity_mismatch", digest=slot.digest)
            return
        candidate = copy.deepcopy(world_model)
        candidate.load_state_dict(candidate_state_dict, strict=True)
        candidate.to(device)
        candidate.eval()
        gate.evaluate(candidate_world_model=candidate, baseline_world_model=world_model, slot=slot)

    return _run_gate
