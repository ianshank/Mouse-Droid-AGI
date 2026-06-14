"""Unit tests for the WS-E4 off-loop hot-swap weight-update source.

Pins the SAFETY-LOCKED WS-E4 contracts for
:class:`mousedroid.learning.on_device.hot_swap.OnDeviceWeightUpdateSource`:

* it satisfies the FULL ``WeightUpdatePollerProtocol`` (start/stop/pending_update/
  acknowledge_swap) PLUS the ``engine_type`` extension property
  (``ENGINE_TYPE_WORLD_MODEL``);
* ``refresh_once`` materialises the active slot OFF the hot loop (via the injected
  ``materialize`` callable) and surfaces a ``PendingWeightUpdate`` carrying ALL 7
  fields with ``engine_type=world_model``;
* the loader seam (``take_materialized``) is a PURE reference return of the
  already-constructed engine — no I/O, no materialise call;
* a corrupt slot (``SlotIntegrityError`` out of ``materialize``) is FAIL-CLOSED:
  no pending update is surfaced AND ``inc_on_device_learning_reverted(
  "integrity_mismatch")`` is incremented BEFORE any swap could occur;
* an unchanged active digest is a no-op (no re-materialise, no duplicate pending);
* ``acknowledge_swap`` clears the pending slot + evicts the cached engine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.cloud.protocol import (
    ENGINE_TYPE_WORLD_MODEL,
    PendingWeightUpdate,
    WeightUpdatePollerProtocol,
)
from mousedroid.learning.on_device.hot_swap import OnDeviceWeightUpdateSource
from mousedroid.learning.on_device.slot_store import SlotIntegrityError

_CHECK_INTERVAL_S = 0.01


class _FakeSlotStore:
    """Minimal slot-store stub exposing ``load_active`` + ``slot_dir``."""

    def __init__(self, active_digest: str | None) -> None:
        self.active_digest = active_digest
        self.slot_dir = Path("/tmp/slot")  # test stub, never written

    def load_active(self) -> str | None:
        return self.active_digest


class _SpyMetrics:
    """Records ``inc_on_device_learning_reverted`` reasons."""

    def __init__(self) -> None:
        self.reverts: list[str] = []

    def inc_on_device_learning_reverted(self, reason: str, amount: int = 1) -> None:
        self.reverts.append(reason)


_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64


def _make_source(
    *,
    active_digest: str | None,
    materialize_returns: object | None = None,
    materialize_raises: Exception | None = None,
    metrics: _SpyMetrics | None = None,
) -> tuple[OnDeviceWeightUpdateSource, list[str]]:
    """Build a source with a spy materialiser; returns (source, materialise_calls)."""
    store = _FakeSlotStore(active_digest)
    calls: list[str] = []
    sentinel = materialize_returns if materialize_returns is not None else object()

    def _materialize(digest: str) -> object:
        calls.append(digest)
        if materialize_raises is not None:
            raise materialize_raises
        return sentinel

    source = OnDeviceWeightUpdateSource(
        slot_store=store,
        materialize=_materialize,
        check_interval_s=_CHECK_INTERVAL_S,
        metrics=metrics,
    )
    return source, calls


def test_source_satisfies_poller_protocol() -> None:
    """The source structurally satisfies ``WeightUpdatePollerProtocol``."""
    source, _ = _make_source(active_digest=None)
    assert isinstance(source, WeightUpdatePollerProtocol)


def test_engine_type_is_world_model() -> None:
    """The source declares the world-model engine discriminator."""
    source, _ = _make_source(active_digest=None)
    assert source.engine_type == ENGINE_TYPE_WORLD_MODEL


@pytest.mark.asyncio
async def test_refresh_with_no_active_slot_is_noop() -> None:
    """No active slot ⇒ no materialise, no pending update."""
    source, calls = _make_source(active_digest=None)
    surfaced = await source.refresh_once()
    assert surfaced is False
    assert calls == []
    assert source.pending_update is None


@pytest.mark.asyncio
async def test_refresh_materialises_active_slot_and_surfaces_pending() -> None:
    """An active slot is materialised off-loop and surfaced as a full PendingWeightUpdate."""
    engine = object()
    source, calls = _make_source(active_digest=_DIGEST_A, materialize_returns=engine)

    surfaced = await source.refresh_once()

    assert surfaced is True
    assert calls == [_DIGEST_A]
    update = source.pending_update
    assert update is not None
    # ALL 7 fields populated; engine_type targets the world model.
    assert update.engine_type == ENGINE_TYPE_WORLD_MODEL
    assert update.sha256 == _DIGEST_A
    assert update.revision == _DIGEST_A
    assert isinstance(update, PendingWeightUpdate)
    assert update.repo_id
    assert update.filename
    assert update.local_path is not None
    assert update.downloaded_at > 0.0


@pytest.mark.asyncio
async def test_take_materialized_is_pure_return_no_io() -> None:
    """The loader seam returns the pre-materialised engine with NO materialise call."""
    engine = object()
    source, calls = _make_source(active_digest=_DIGEST_A, materialize_returns=engine)
    await source.refresh_once()
    update = source.pending_update
    assert update is not None
    calls.clear()

    returned = source.take_materialized(update)

    # Pure reference return — identity preserved, materialise NOT re-invoked.
    assert returned is engine
    assert calls == []


@pytest.mark.asyncio
async def test_corrupt_slot_is_fail_closed_and_counted() -> None:
    """A corrupt slot increments integrity_mismatch and surfaces NO pending update."""
    metrics = _SpyMetrics()
    source, calls = _make_source(
        active_digest=_DIGEST_A,
        materialize_raises=SlotIntegrityError("bad slot"),
        metrics=metrics,
    )

    surfaced = await source.refresh_once()

    assert surfaced is False
    assert calls == [_DIGEST_A]
    # Fail-closed: nothing to swap.
    assert source.pending_update is None
    # Counted BEFORE any swap could occur.
    assert metrics.reverts == ["integrity_mismatch"]


@pytest.mark.asyncio
async def test_unchanged_active_digest_does_not_rematerialise() -> None:
    """A second refresh with the same active digest is a no-op (no duplicate work)."""
    source, calls = _make_source(active_digest=_DIGEST_A)
    await source.refresh_once()
    first = source.pending_update
    assert first is not None

    surfaced = await source.refresh_once()

    assert surfaced is False
    # materialise fired exactly once.
    assert calls == [_DIGEST_A]
    # Same pending object (not replaced).
    assert source.pending_update is first


@pytest.mark.asyncio
async def test_acknowledge_swap_clears_pending_and_evicts_engine() -> None:
    """acknowledge_swap clears the slot and evicts the cached engine."""
    engine = object()
    source, _ = _make_source(active_digest=_DIGEST_A, materialize_returns=engine)
    await source.refresh_once()
    update = source.pending_update
    assert update is not None

    source.acknowledge_swap(update)

    assert source.pending_update is None
    # The cached engine is gone — a stale loader lookup must not resurrect it.
    with pytest.raises(KeyError):
        source.take_materialized(update)


@pytest.mark.asyncio
async def test_new_digest_after_ack_materialises_again() -> None:
    """After ack, a NEW active digest is materialised + surfaced afresh."""
    store = _FakeSlotStore(_DIGEST_A)
    calls: list[str] = []

    def _materialize(digest: str) -> object:
        calls.append(digest)
        return object()

    source = OnDeviceWeightUpdateSource(
        slot_store=store,
        materialize=_materialize,
        check_interval_s=_CHECK_INTERVAL_S,
    )
    await source.refresh_once()
    first = source.pending_update
    assert first is not None
    source.acknowledge_swap(first)

    # Operator/gate promotes a new candidate.
    store.active_digest = _DIGEST_B
    surfaced = await source.refresh_once()

    assert surfaced is True
    assert calls == [_DIGEST_A, _DIGEST_B]
    new_update = source.pending_update
    assert new_update is not None
    assert new_update.sha256 == _DIGEST_B


@pytest.mark.asyncio
async def test_take_materialized_unknown_update_raises() -> None:
    """A loader lookup for an unknown update raises (never silently returns None)."""
    source, _ = _make_source(active_digest=None)
    bogus = PendingWeightUpdate(
        repo_id="x",
        filename="y",
        revision=_DIGEST_A,
        sha256=_DIGEST_A,
        local_path=Path("/tmp/x.pt"),  # stub
        downloaded_at=1.0,
        engine_type=ENGINE_TYPE_WORLD_MODEL,
    )
    with pytest.raises(KeyError):
        source.take_materialized(bogus)


@pytest.mark.asyncio
async def test_start_stop_lifecycle_drains_background_task() -> None:
    """start spawns the refresh loop; stop cancels it cleanly (no leaked task)."""
    source, calls = _make_source(active_digest=_DIGEST_A)
    await source.start()
    # Give the loop a moment to run at least one refresh.
    import asyncio

    for _ in range(50):
        if calls:
            break
        await asyncio.sleep(_CHECK_INTERVAL_S)
    await source.stop()
    assert calls  # the background loop drove at least one refresh
    # Idempotent stop is safe.
    await source.stop()


@pytest.mark.asyncio
async def test_start_is_idempotent_no_duplicate_task() -> None:
    """A second ``start`` while running is a no-op (does not spawn a duplicate task)."""
    source, _ = _make_source(active_digest=None)
    await source.start()
    first_task = source._task  # type: ignore[attr-defined]
    await source.start()  # guarded: already running
    assert source._task is first_task  # type: ignore[attr-defined]
    await source.stop()


@pytest.mark.asyncio
async def test_background_loop_survives_a_failing_refresh() -> None:
    """A refresh exception is logged + swallowed; the loop keeps running."""
    import asyncio

    store = _FakeSlotStore(_DIGEST_A)
    calls: list[str] = []

    def _materialize(digest: str) -> object:
        calls.append(digest)
        # First call raises a NON-integrity error (exercises the broad-except);
        # subsequent refreshes see the same seen-digest and short-circuit, so a
        # second active digest proves the loop survived.
        raise RuntimeError("transient materialise failure")

    source = OnDeviceWeightUpdateSource(
        slot_store=store,
        materialize=_materialize,
        check_interval_s=_CHECK_INTERVAL_S,
    )
    await source.start()
    for _ in range(50):
        if calls:
            break
        await asyncio.sleep(_CHECK_INTERVAL_S)
    # The loop survived the raising refresh (task still alive, not crashed).
    assert source._task is not None  # type: ignore[attr-defined]
    assert not source._task.done()  # type: ignore[attr-defined]
    await source.stop()
    assert calls  # at least one (failing) refresh ran
