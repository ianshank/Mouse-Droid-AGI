"""Unit tests for the WS-E4 composed hot-loop weight-update loader.

Pins :func:`mousedroid.factory._compose_weight_update_loader`:

* an on-device-owned update returns the source's PRE-materialised engine (PURE
  reference return — the cloud loader is NOT consulted);
* a non-owned update delegates to the cloud loader when one is wired;
* a non-owned update with NO cloud loader raises (fail-closed) so the
  orchestrator's broad-except leaves the live model untouched rather than
  swapping a bogus engine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.cloud.protocol import ENGINE_TYPE_WORLD_MODEL, PendingWeightUpdate
from mousedroid.factory import _compose_weight_update_loader


class _FakeSource:
    """Stub exposing the ``owns`` / ``take_materialized`` loader surface."""

    def __init__(self, owned: PendingWeightUpdate, engine: object) -> None:
        self._owned = owned
        self._engine = engine

    def owns(self, update: PendingWeightUpdate) -> bool:
        return update is self._owned

    def take_materialized(self, update: PendingWeightUpdate) -> object:
        if update is not self._owned:
            raise KeyError(update)
        return self._engine


def _update(revision: str) -> PendingWeightUpdate:
    return PendingWeightUpdate(
        repo_id="r",
        filename="f",
        revision=revision,
        sha256="0" * 64,
        local_path=Path("/tmp/x.pt"),  # stub
        downloaded_at=1.0,
        engine_type=ENGINE_TYPE_WORLD_MODEL,
    )


def test_owned_update_returns_premade_engine_no_cloud_call() -> None:
    """An on-device-owned update returns the pre-materialised engine purely."""
    owned = _update("owned")
    engine = object()
    source = _FakeSource(owned, engine)

    cloud_calls: list[PendingWeightUpdate] = []

    def _cloud(update: PendingWeightUpdate) -> object:
        cloud_calls.append(update)
        return object()

    loader = _compose_weight_update_loader(_cloud, source)  # type: ignore[arg-type]
    assert loader(owned) is engine
    assert cloud_calls == []  # cloud loader never consulted for an owned update


def test_non_owned_update_delegates_to_cloud_loader() -> None:
    """A non-owned update is delegated to the cloud loader."""
    owned = _update("owned")
    source = _FakeSource(owned, object())
    other = _update("other")
    cloud_engine = object()

    def _cloud(update: PendingWeightUpdate) -> object:
        return cloud_engine

    loader = _compose_weight_update_loader(_cloud, source)  # type: ignore[arg-type]
    assert loader(other) is cloud_engine


def test_non_owned_update_no_cloud_loader_raises_fail_closed() -> None:
    """A non-owned update with no cloud loader raises (fail-closed)."""
    owned = _update("owned")
    source = _FakeSource(owned, object())
    other = _update("other")

    loader = _compose_weight_update_loader(None, source)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="no loader wired"):
        loader(other)
