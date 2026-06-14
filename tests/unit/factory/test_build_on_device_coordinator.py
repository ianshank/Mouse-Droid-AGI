"""Unit tests for the Phase-6 WS3 on-device coordinator factory helpers.

Covers the factory wiring branches not exercised by the integration path:

* ``_load_replay_batch`` returns an empty ``(0, input_dim)`` tensor when the
  replay store has no records (the safe empty-store branch the coordinator's
  ``load_batch`` callable relies on);
* ``_count_replay_records`` reports zero for an empty store;
* ``build_on_device_coordinator`` returns a non-None coordinator when enabled.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog
import torch

from mousedroid.config.schema import Settings
from mousedroid.factory import (
    _count_new_replay_records,
    _count_replay_records,
    _load_replay_batch,
    build_on_device_coordinator,
)
from mousedroid.training.replay.lmdb_reader import LMDBReplayReader

_INPUT_DIM = 16


def _empty_reader(tmp_path: Path) -> LMDBReplayReader:
    """Build a replay reader over an empty experience store."""
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "experience": {"path": str(tmp_path / "empty_root"), "map_size_gb": 0.01},
        }
    )
    return LMDBReplayReader(cfg.experience)


def test_load_replay_batch_empty_store_returns_empty_tensor(tmp_path: Path) -> None:
    """An empty replay store yields a ``(0, input_dim)`` batch, not an error."""
    reader = _empty_reader(tmp_path)

    batch = _load_replay_batch(reader, _INPUT_DIM, cap=8)

    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (0, _INPUT_DIM)


def test_count_replay_records_empty_store_is_zero(tmp_path: Path) -> None:
    """An empty replay store counts zero new records."""
    reader = _empty_reader(tmp_path)

    assert _count_replay_records(reader, cap=8) == 0


def test_build_coordinator_returns_coordinator_when_enabled(tmp_path: Path) -> None:
    """An enabled on-device block wires a non-None coordinator."""
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "experience": {"path": str(tmp_path / "root"), "map_size_gb": 0.01},
            "on_device_learning": {"enabled": True, "trigger_min_new_records": 5},
        }
    )

    coordinator = build_on_device_coordinator(cfg)

    assert coordinator is not None


# --------------------------------------------------------------------------- #
# WS-E0/E3: thread the LIVE world model into the gate runner
# --------------------------------------------------------------------------- #
def _extract_gate(coordinator: object) -> object:
    """Pull the ``RegressionGate`` the gate-runner closure closes over.

    The WS-E3 recon-loss gate-runner closes over the ``RegressionGate`` instance
    (alongside the live baseline RSSM); the cell index is an implementation
    detail, so locate the cell holding a gate by type rather than position.
    """
    from mousedroid.learning.on_device.regression_gate import RegressionGate

    runner = coordinator._gate_runner  # type: ignore[attr-defined]
    for cell in runner.__closure__ or ():
        contents = cell.cell_contents
        if isinstance(contents, RegressionGate):
            return contents
    raise AssertionError("gate runner closure does not hold a RegressionGate")


def _extract_baseline_world_model(coordinator: object) -> object:
    """Pull the live baseline RSSM the WS-E3 gate-runner closure scores against.

    The recon-loss gate-runner loads the candidate slot into a deep COPY of the
    live RSSM and scores it against the live RSSM held in the closure — so the
    baseline world model lives in a closure cell, not on the gate object.
    """
    from mousedroid.world_model.rssm import RSSM

    runner = coordinator._gate_runner  # type: ignore[attr-defined]
    for cell in runner.__closure__ or ():
        contents = cell.cell_contents
        if isinstance(contents, RSSM):
            return contents
    raise AssertionError("gate runner closure does not hold a baseline RSSM")


def _enabled_cfg(tmp_path: Path, *, tag: str = "root") -> Settings:
    return Settings.model_validate(
        {
            "mock_hardware": True,
            "experience": {"path": str(tmp_path / tag), "map_size_gb": 0.01},
            "on_device_learning": {
                "enabled": True,
                "trigger_min_new_records": 5,
                # Small refine geometry so a modest seeded store yields BOTH the
                # refine window AND a disjoint held-out window (so a real recon-loss
                # gate — not the no-op fallback — is wired + inspectable).
                "refine_sequence_length": 3,
                "refine_batch_episodes": 2,
            },
        }
    )


def _seed_records(cfg: Settings, n: int) -> None:
    """Seed ``n`` replay records so the gate can build a disjoint held-out batch."""
    from mousedroid.experience.logger import ExperienceLogger
    from mousedroid.experience.record import MouseDroidExperienceRecord

    logger = ExperienceLogger(cfg.experience)
    logger.open()
    try:
        for _ in range(n):
            logger.log(MouseDroidExperienceRecord())
    finally:
        logger.close()


def test_coordinator_uses_injected_world_model(tmp_path: Path) -> None:
    """When a live world model is injected, the gate scores against THAT model."""
    from mousedroid.factory import build_world_model
    from mousedroid.learning.on_device.regression_gate import RegressionGate

    cfg = _enabled_cfg(tmp_path)
    _seed_records(cfg, 16)  # refine(6) + disjoint held-out(6), with headroom
    wm = build_world_model(cfg)

    coordinator = build_on_device_coordinator(cfg, world_model=wm)

    assert coordinator is not None
    # The gate runner closes over the injected live RSSM as the scoring baseline.
    assert _extract_baseline_world_model(coordinator) is wm
    # ...and a real recon-loss RegressionGate is wired.
    assert isinstance(_extract_gate(coordinator), RegressionGate)


def test_coordinator_builds_world_model_when_none(tmp_path: Path) -> None:
    """``world_model=None`` builds a working gate via ``build_world_model(cfg)``."""
    from mousedroid.world_model.rssm import RSSM

    cfg = _enabled_cfg(tmp_path)
    _seed_records(cfg, 16)

    coordinator = build_on_device_coordinator(cfg, world_model=None)

    assert coordinator is not None
    # A real RSSM (the default config builds a plain RSSM) was constructed and is
    # the gate-runner's scoring baseline.
    baseline = _extract_baseline_world_model(coordinator)
    assert isinstance(baseline, RSSM)
    assert hasattr(baseline, "train_sequence")


def test_gate_does_not_construct_bare_rssm_literal(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """WS-E0 fix: the gate never constructs ``RSSM(cfg.model)`` directly.

    The pre-WS-E0 code built a fresh ``RSSM(cfg.model)`` in the gate runner,
    divorced from the live model (a DualStreamRSSM-arch bug). Threading the
    live model (or ``build_world_model(cfg)``) must replace that literal — so
    constructing the gate must NOT call ``RSSM.__init__`` directly.
    """
    import mousedroid.world_model.rssm as rssm_mod

    cfg = _enabled_cfg(cfg_tmp := tmp_path)
    assert cfg_tmp is not None

    calls: list[int] = []
    real_init = rssm_mod.RSSM.__init__

    def _spy_init(self: object, *args: object, **kwargs: object) -> None:
        calls.append(1)
        real_init(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(rssm_mod.RSSM, "__init__", _spy_init)

    # Inject a live world model so the gate has no excuse to build its own RSSM.
    from mousedroid.factory import build_world_model

    # Build the WM BEFORE installing the spy so only gate-internal construction
    # would register.
    monkeypatch.undo()
    wm = build_world_model(cfg)
    monkeypatch.setattr(rssm_mod.RSSM, "__init__", _spy_init)
    calls.clear()

    coordinator = build_on_device_coordinator(cfg, world_model=wm)

    assert coordinator is not None
    assert calls == [], "gate runner must NOT construct a bare RSSM(cfg.model) literal"


# --------------------------------------------------------------------------- #
# WS-E0: capability gate — refinement requires ``train_sequence``
# --------------------------------------------------------------------------- #
class _NoTrainSequenceWorldModel:
    """A minimal world-model stand-in that lacks ``train_sequence``.

    Mirrors ``DualStreamRSSM`` / ``DualStreamRSSMOnnx``, which expose
    ``imagine_step`` / ``eval`` but not the ``train_sequence`` the on-device
    refiner requires. Used to drive the capability-gate guard.
    """

    def eval(self) -> _NoTrainSequenceWorldModel:
        return self

    def imagine_step(self, *args: object, **kwargs: object) -> None:
        return None


def test_coordinator_disabled_for_engine_without_train_sequence(tmp_path: Path) -> None:
    """An effective world model lacking ``train_sequence`` disables refinement.

    The on-device refiner calls ``train_sequence`` (present on ``RSSM`` but not
    on ``DualStreamRSSM`` / ``DualStreamRSSMOnnx``). When the EFFECTIVE engine
    lacks that capability the coordinator returns ``None`` instead of building
    an unusable refiner, and logs ``on_device_refiner_unsupported_engine``.
    """
    cfg = _enabled_cfg(tmp_path)
    unsupported = _NoTrainSequenceWorldModel()

    with structlog.testing.capture_logs() as logs:
        coordinator = build_on_device_coordinator(cfg, world_model=unsupported)

    assert coordinator is None
    events = [entry for entry in logs if entry["event"] == "on_device_refiner_unsupported_engine"]
    assert len(events) == 1
    assert events[0]["log_level"] == "warning"
    assert events[0]["engine_type"] == "_NoTrainSequenceWorldModel"


def test_coordinator_builds_for_engine_with_train_sequence(tmp_path: Path) -> None:
    """A real ``RSSM`` (default config) HAS ``train_sequence`` → still builds."""
    from mousedroid.factory import build_world_model

    cfg = _enabled_cfg(tmp_path)
    wm = build_world_model(cfg)
    assert hasattr(wm, "train_sequence")

    with structlog.testing.capture_logs() as logs:
        coordinator = build_on_device_coordinator(cfg, world_model=wm)

    assert coordinator is not None
    unsupported = [e for e in logs if e["event"] == "on_device_refiner_unsupported_engine"]
    assert unsupported == []


def test_coordinator_none_world_model_builds_default_rssm_with_train_sequence(
    tmp_path: Path,
) -> None:
    """``world_model=None`` resolves the default plain-RSSM (has ``train_sequence``).

    The default config builds a plain ``RSSM`` (no ``cfc_hidden_dim``), which
    HAS ``train_sequence`` — so the None path still wires a coordinator and
    never trips the capability gate.
    """
    cfg = _enabled_cfg(tmp_path)

    with structlog.testing.capture_logs() as logs:
        coordinator = build_on_device_coordinator(cfg, world_model=None)

    assert coordinator is not None
    unsupported = [e for e in logs if e["event"] == "on_device_refiner_unsupported_engine"]
    assert unsupported == []


# --------------------------------------------------------------------------- #
# WS-E6: trigger arming — count NEW-since-consumed, disarm after a fired cycle
# --------------------------------------------------------------------------- #
def test_count_new_replay_records_skips_consumed_offset(tmp_path: Path) -> None:
    """``_count_new_replay_records`` counts only records AFTER the consumed offset.

    The TOTAL-count probe re-armed the trigger every cadence on stale data. The
    fix counts records BEYOND a consumed baseline; with ``consumed == total`` the
    new count is 0 (disarmed) and with a partial offset it is the remainder.
    """
    cfg = _enabled_cfg(tmp_path)
    _seed_records(cfg, 10)
    reader = LMDBReplayReader(cfg.experience)

    # No records consumed yet → all 10 are new (capped well above 10).
    assert _count_new_replay_records(reader, consumed=0, cap=64) == 10
    # Everything consumed → trigger disarmed.
    assert _count_new_replay_records(reader, consumed=10, cap=64) == 0
    # Partial consume → only the remainder is new.
    assert _count_new_replay_records(reader, consumed=7, cap=64) == 3
    # A consumed offset past the store size never underflows below zero.
    assert _count_new_replay_records(reader, consumed=99, cap=64) == 0


def test_count_new_replay_records_caps_the_new_window(tmp_path: Path) -> None:
    """The cap bounds the NEW (post-consumed) scan, not the absolute total."""
    cfg = _enabled_cfg(tmp_path)
    _seed_records(cfg, 20)
    reader = LMDBReplayReader(cfg.experience)

    # 20 records, 5 consumed, cap the new window at 8 → counts at most 8 new.
    assert _count_new_replay_records(reader, consumed=5, cap=8) == 8


def test_coordinator_trigger_disarms_after_fire_no_regate_on_stale(tmp_path: Path) -> None:
    """A fired cycle advances the consumed offset → no re-fire on the SAME records.

    The pre-WS-E6 wiring counted the TOTAL store size and never advanced a
    consumed marker, so once the store crossed ``trigger_min_new_records`` the
    coordinator re-ran the refine + gate every cadence on byte-identical stale
    data. After the fix the first ``maybe_update`` fires + records the consumed
    baseline; a second ``maybe_update`` with no fresh records sees ZERO new and
    is a no-op (returns ``None``).
    """
    cfg = _enabled_cfg(tmp_path)
    # Refine geometry is 2*3=6; seed enough for the refine batch but FEWER than
    # 2x so a second window's worth of NEW records cannot exist post-consume.
    _seed_records(cfg, 8)

    coordinator = build_on_device_coordinator(cfg)
    assert coordinator is not None

    first = asyncio.run(coordinator.maybe_update())  # type: ignore[attr-defined]
    assert first is not None, "first cycle should fire (store >= trigger threshold)"

    second = asyncio.run(coordinator.maybe_update())  # type: ignore[attr-defined]
    assert second is None, "trigger must disarm — no re-gate on stale (already-consumed) data"


def test_coordinator_rearms_when_fresh_records_arrive(tmp_path: Path) -> None:
    """After consuming, NEW records crossing the threshold re-arm the trigger."""
    cfg = _enabled_cfg(tmp_path)
    _seed_records(cfg, 8)

    coordinator = build_on_device_coordinator(cfg)
    assert coordinator is not None

    first = asyncio.run(coordinator.maybe_update())  # type: ignore[attr-defined]
    assert first is not None

    # No new records yet → disarmed.
    assert asyncio.run(coordinator.maybe_update()) is None  # type: ignore[attr-defined]

    # Fresh experience arrives (>= trigger_min_new_records=5 new records).
    _seed_records(cfg, 6)
    third = asyncio.run(coordinator.maybe_update())  # type: ignore[attr-defined]
    assert third is not None, "fresh records beyond the consumed offset must re-arm the trigger"


# --------------------------------------------------------------------------- #
# WS-E6: gate-runner decoder init must NOT perturb the global RNG (fork_rng)
# --------------------------------------------------------------------------- #
def test_gate_runner_build_does_not_perturb_global_cpu_rng(tmp_path: Path) -> None:
    """Building the WS-E3 gate runner restores the global CPU RNG (fork_rng).

    The gate runner seeds the global RNG (``torch.manual_seed(scoring_seed)``)
    to make the shared decoder init reproducible, then builds
    ``RawModalityDecoders(...).to(device)``. Without restoring, that seed leaks
    into the caller's RNG stream — every subsequent draw on the process RNG is
    silently shifted. Wrapping the seed + decoder init in ``torch.random.fork_rng``
    must leave the global RNG byte-identical across the build.

    A CPU world model is injected so the build does NOT call ``build_world_model``
    internally (which would legitimately consume CPU RNG constructing a fresh
    RSSM) — this isolates the gate-runner seed leak from model construction.
    """
    from mousedroid.factory import build_world_model

    cfg = _enabled_cfg(tmp_path)
    _seed_records(cfg, 16)
    wm = build_world_model(cfg).to("cpu")  # type: ignore[attr-defined]

    # Pin a known global RNG state AFTER building the model, capture it, then build
    # the coordinator (which builds the gate runner + its seeded decoders) and
    # assert the state is byte-identical — the seed was confined to the fork.
    torch.manual_seed(2024)
    before = torch.get_rng_state().clone()

    coordinator = build_on_device_coordinator(cfg, world_model=wm)

    assert coordinator is not None
    after = torch.get_rng_state()
    assert torch.equal(before, after), "gate-runner decoder init leaked its seed into global RNG"


def test_gate_runner_build_does_not_perturb_global_cuda_rng(tmp_path: Path) -> None:
    """On a CUDA box, the gate-runner build also restores the CUDA RNG.

    ``torch.manual_seed`` seeds CPU AND every CUDA generator, so the fork must
    cover the model's CUDA device too. Skipped on a CPU-only host (CI); the guard
    structure is exercised by the CPU test above. On the dev box (RTX) this proves
    the real CUDA RNG round-trips.
    """
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("no CUDA device — CUDA RNG restore path covered structurally on CPU")

    from mousedroid.factory import build_world_model

    cfg = _enabled_cfg(tmp_path)
    _seed_records(cfg, 16)
    wm = build_world_model(cfg).to("cuda")  # type: ignore[attr-defined]

    torch.cuda.manual_seed_all(4096)
    before = torch.cuda.get_rng_state(0).clone()

    coordinator = build_on_device_coordinator(cfg, world_model=wm)

    assert coordinator is not None
    after = torch.cuda.get_rng_state(0)
    assert torch.equal(before, after), "gate-runner decoder init leaked its seed into the CUDA RNG"
