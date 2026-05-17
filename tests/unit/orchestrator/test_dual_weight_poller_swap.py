"""C1.2: orchestrator drives both policy and world-model pollers per tick."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

from mousedroid.cloud.protocol import PendingWeightUpdate
from mousedroid.config.schema import Settings

# Reuse the existing stub helpers from the single-poller swap test.
# _build_orch was extended in Step 3.0 to accept weight_update_pollers= and
# weight_update_loader= kwargs additive to the legacy poller=/loader= pair.
from tests.unit.orchestrator.test_weight_update_swap import _build_orch, _StubPoller


def _pending(engine_type: str, rev: str) -> PendingWeightUpdate:
    """Build a stub PendingWeightUpdate for the given engine + revision."""
    return PendingWeightUpdate(
        repo_id="ianshank/mousedroid-test",
        filename=f"{engine_type}.bin",
        revision=rev,
        sha256="0" * 64,
        local_path=Path(f"/tmp/{engine_type}.bin"),
        downloaded_at=time.time(),
        engine_type=engine_type,
    )


def test_dual_pollers_swap_independently() -> None:
    """Both pollers' pending updates are consumed in one _apply call."""
    cfg = Settings(mock_hardware=True)
    policy_poller = _StubPoller([_pending("policy", "rev-p1")])
    wm_poller = _StubPoller([_pending("world_model", "rev-w1")])
    loader = MagicMock(side_effect=lambda u: MagicMock(name=f"engine-{u.engine_type}"))

    orch, _wm_mock, _agent = _build_orch(
        cfg,
        weight_update_pollers={"policy": policy_poller, "world_model": wm_poller},
        weight_update_loader=loader,
    )

    swap_reset = orch._apply_pending_weight_update()
    assert swap_reset is True, "world_model swap must zero recurrent state"
    assert len(policy_poller.ack_calls) == 1
    assert len(wm_poller.ack_calls) == 1
    assert loader.call_count == 2


def test_empty_pollers_mapping_is_noop() -> None:
    """Empty pollers mapping short-circuits with no state mutation."""
    cfg = Settings(mock_hardware=True)
    orch, *_ = _build_orch(cfg, weight_update_pollers={})
    assert orch._apply_pending_weight_update() is False


def test_both_pollers_pending_same_tick_iteration_order() -> None:
    """Policy-before-world-model is the documented contract — assert insertion order."""
    cfg = Settings(mock_hardware=True)
    policy_poller = _StubPoller([_pending("policy", "rev-p2")])
    wm_poller = _StubPoller([_pending("world_model", "rev-w2")])
    swap_call_order: list[str] = []

    def _trace_loader(u: PendingWeightUpdate) -> MagicMock:
        swap_call_order.append(u.engine_type)
        return MagicMock(name=f"engine-{u.engine_type}")

    orch, _wm, _agent = _build_orch(
        cfg,
        weight_update_pollers={"policy": policy_poller, "world_model": wm_poller},
        weight_update_loader=_trace_loader,
    )
    swap_reset = orch._apply_pending_weight_update()
    assert swap_reset is True
    assert swap_call_order == ["policy", "world_model"], (
        "iteration order must be policy-before-world-model so the "
        "world-model swap's recurrent-state reset doesn't clobber a "
        "policy swap from the same tick"
    )
